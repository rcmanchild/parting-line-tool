"""The parting-line pipeline: classify -> edges -> silhouettes -> chain -> undercuts."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
from OCP.TopAbs import TopAbs_EDGE
from OCP.GC import GC_MakeSegment
from OCP.GeomAPI import GeomAPI_ProjectPointOnCurve
from OCP.gp import gp_Dir, gp_Pln, gp_Pnt, gp_Vec

from . import segments as sg
from .chain import Chaining, build_wires, chain
from .edges import EDGE, VBOT, VTOP, EdgeCtx, edge_curve
from .faces import CAVITY, CORE, MIXED, VERTICAL, FaceCtx
from .model import Model
from .occ import indexed_map, make_compound, map_list, pnt
from .silhouette import face_silhouettes, polyline_length
from .bands import BandSolver, arrival_points
from .undercut import RayCaster, mesh_arrays, mesh_deflection

VERTICAL_RULES = ("top", "bottom", "middle")


@dataclass
class Options:
    draft_tol_deg: float = 0.5       # |draft| below this counts as vertical
    vertical: str = "top"            # where the line goes on vertical bands
    gap_tol: float = 0.01            # chaining tolerance, mm
    chord_tol: float = 1e-3          # silhouette sagitta tolerance, mm
    point_defl: float = 1e-3         # deflection of output polylines, mm
    undercut: bool = True
    undercut_min_frac: float = 0.05  # face flagged when this share of its area is blocked ...
    undercut_min_area: float = 0.5   # ... and the blocked area is at least this (mm2)
    low_draft_deg: float = 1.0       # "near-zero draft" threshold for reporting
    bridge_tol: float = 0.5          # close gaps up to this (mm) with straight bridges, reported
    mesh_defl: float | None = None   # ray-cast mesh deflection; default from model size


@dataclass
class FaceReport:
    number: int
    type: str
    cls: str
    area: float
    min_draft_deg: float | None
    max_draft_deg: float | None
    needs_draft: bool
    undercut_frac: float = 0.0
    undercut: bool = False


@dataclass
class Result:
    pull: np.ndarray
    options: Options
    faces: list[FaceReport]
    segments: list[sg.Segment]
    chaining: Chaining
    wires: list
    stats: dict
    warnings: list[str] = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    split: object | None = None     # split.SplitResult, filled on request
    surface: object | None = None   # surface.SurfaceResult, filled on request

    @property
    def loops(self):
        return self.chaining.loops


def _draft_deg(f: float) -> float:
    return math.degrees(math.asin(max(-1.0, min(1.0, f))))


def analyze(model: Model, pull, opts: Options | None = None) -> Result:
    opts = opts or Options()
    if opts.vertical not in VERTICAL_RULES:
        raise ValueError(f"vertical rule must be one of {VERTICAL_RULES}")
    d = np.asarray(pull, dtype=float)
    d = d / np.linalg.norm(d)
    t = math.sin(math.radians(opts.draft_tol_deg))
    timings: dict[str, float] = {}
    warnings: list[str] = list(model.heal_info.warnings)
    clock = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        timings[name] = round(now - clock, 3)
        clock = now

    # 1. sample faces
    fcs = [FaceCtx(i, f) for i, f in enumerate(model.faces)]
    for fc in fcs:
        fc.set_pull(d)
        fc.sample_grid()
        fc.update_f(d)
    lap("sample_faces")

    # 2. sample edges (adds boundary n.d values), then classify faces
    ecs = [EdgeCtx(i, e, model.edge_uses[i], fcs) for i, e in enumerate(model.edges)]
    for fc in fcs:
        fc.classify(t)
    lap("sample_edges_classify")

    n_nonmanifold = sum(1 for ec in ecs if not ec.degenerate and len(ec.uses) > 2)
    if n_nonmanifold:
        warnings.append(f"{n_nonmanifold} non-manifold edges (shared by >2 faces) were skipped")

    # 3. smooth silhouettes on mixed faces.  Done before the edge pass: where a
    #    silhouette ends on an edge, the parting status of that edge changes.
    sil_segs: list[sg.Segment] = []
    for fc in fcs:
        if fc.cls != MIXED:
            continue
        for piece in face_silhouettes(fc, opts.chord_tol, max(0.2, 5e-3 * model.diag)):
            if polyline_length(piece.xyz) < opts.gap_tol:
                continue
            pts = np.vstack([piece.xyz, piece.xyz[:1]]) if piece.closed else piece.xyz
            curve = sg.interpolate(piece.xyz, closed=piece.closed)
            if curve is None:
                continue
            t0, t1 = curve.FirstParameter(), curve.LastParameter()
            sil_segs.append(sg.Segment(sg.SILHOUETTE, curve, t0, t1, pts, sg.curve_length(curve, t0, t1),
                                       {"face": fc.index + 1}, piece.uv))
    lap("silhouettes")

    # 4. parting status along edges
    extra = _silhouette_edge_params(sil_segs, model, ecs, opts.gap_tol)
    delta = max(1e-3, 1e-5 * model.diag)
    for ec in ecs:
        ec.compute_intervals(delta, extra.get(ec.index, ()))
    lap("edge_status")

    # 5. vertical regions: connected vertical faces
    parent = list(range(len(fcs)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for ec in ecs:
        if ec.degenerate or len(ec.uses) != 2:
            continue
        a, b = ec.uses
        if a.fc.cls == VERTICAL and b.fc.cls == VERTICAL:
            ra, rb = find(a.fc.index), find(b.fc.index)
            if ra != rb:
                parent[rb] = ra
    has_top: set[int] = set()
    has_bot: set[int] = set()
    for ec in ecs:
        for _, _, st, vf in ec.intervals:
            if st == VTOP:
                has_top.add(find(vf))
            elif st == VBOT:
                has_bot.add(find(vf))
    bands = has_top & has_bot  # vertical regions that separate cavity from core

    # 6. parting segments along model edges (sharp edges, edge-coincident silhouettes)
    segs: list[sg.Segment] = []
    for ec in ecs:
        curve = None
        for ta, tb, st, vf in ec.intervals:
            if st != EDGE:
                continue
            curve = curve or edge_curve(ec.edge)
            if curve is None:
                warnings.append(f"edge {ec.index + 1} has no 3D curve; skipped")
                break
            segs.append(sg.from_curve(sg.EDGE, curve, ta, tb, opts.point_defl, edge=ec.index + 1))
    # a silhouette running exactly along a parting edge is already covered by it
    n_dupes = 0
    if segs:
        edge_polys = [sgm.points for sgm in segs]
        for sgm in sil_segs:
            if _within(sgm.points, edge_polys, opts.gap_tol):
                n_dupes += 1
            else:
                segs.append(sgm)
    else:
        segs.extend(sil_segs)
    lap("edge_segments")

    # 6b. ray caster (mesh) for undercuts and vertical bands
    lin = opts.mesh_defl or mesh_deflection(model.diag)
    caster = None
    if opts.undercut or (bands and opts.vertical != "middle"):
        V, T, owner = mesh_arrays(model.shape, model.faces, lin)
        caster = RayCaster(V, T, d, owner)
    lap("mesh")

    # 6c. vertical bands
    if bands and opts.vertical == "middle":
        segs.extend(_band_sections(model, fcs, ecs, bands, find, d, opts))
    elif bands:
        solver = BandSolver(model, fcs, ecs, caster, d, opts.vertical, lin, opts)
        cache: dict[int, np.ndarray] = {}
        found = list(segs)
        for root in bands:
            region = {fc.index for fc in fcs if fc.cls == VERTICAL and find(fc.index) == root}
            arrivals = arrival_points(found, region, model, ecs, opts.gap_tol, cache)
            segs.extend(solver.solve(region, arrivals))
        warnings.extend(solver.warnings)
    lap("bands")

    # 7. chain into loops; drop slivers shorter than the gap tolerance and
    #    short dead-end spurs left by numerical noise at corners
    n_short = sum(1 for sgm in segs if sgm.length < opts.gap_tol)
    segs = [sgm for sgm in segs if sgm.length >= opts.gap_tol]
    ch = chain(segs, opts.gap_tol)
    n_spurs = 0
    for _ in range(5):
        spurs = _spurs(ch, segs, 10.0 * opts.gap_tol)
        if not spurs:
            break
        n_spurs += len(spurs)
        segs = [sgm for i, sgm in enumerate(segs) if i not in spurs]
        ch = chain(segs, opts.gap_tol)
    # close what is left open across small gaps (micro-faces at tips and
    # corners the sampling cannot resolve); every bridge is reported
    bridges = _bridges(ch, opts.bridge_tol, opts.point_defl)
    if bridges:
        segs.extend(bridges)
        ch = chain(segs, opts.gap_tol)
        warnings.append(f"{len(bridges)} gap(s) closed with straight bridges (longest "
                        f"{max(b.length for b in bridges):.3f} mm): "
                        + "; ".join(np.array2string(b.start, precision=3) for b in bridges[:5]))
    wires = build_wires(segs, ch)
    lap("chain")

    # 8. undercuts
    blocked_frac = np.zeros(len(fcs))
    if opts.undercut:
        blocked_frac = _undercuts(fcs, caster, t, lin)
    lap("undercut")

    # 9. statistics
    faces_out: list[FaceReport] = []
    sin_low = math.sin(math.radians(opts.low_draft_deg))
    stats = {
        "face_counts": {CAVITY: 0, CORE: 0, VERTICAL: 0, MIXED: 0},
        "face_area": {CAVITY: 0.0, CORE: 0.0, VERTICAL: 0.0, MIXED: 0.0},
    }
    min_draft = {CAVITY: math.inf, CORE: math.inf}
    low_draft_area = 0.0
    undercut_area = 0.0
    for fc, frac in zip(fcs, blocked_frac):
        area = fc.area
        stats["face_counts"][fc.cls] += 1
        stats["face_area"][fc.cls] += area
        fmin, fmax = fc.fmin, fc.fmax
        if fc.cls == CAVITY:
            min_draft[CAVITY] = min(min_draft[CAVITY], _draft_deg(max(fmin, 0.0)))
        elif fc.cls == CORE:
            min_draft[CORE] = min(min_draft[CORE], _draft_deg(max(-fmax, 0.0)))
        if fc.cls == VERTICAL:
            low_draft_area += area
        elif len(fc.samples_f):
            low_draft_area += area * float(np.mean(np.abs(fc.samples_f) < sin_low))
        flagged = bool(frac >= opts.undercut_min_frac and area * frac >= opts.undercut_min_area)
        if flagged:
            undercut_area += area * float(frac)
        fin = math.isfinite(fmin) and math.isfinite(fmax)
        faces_out.append(FaceReport(
            number=fc.index + 1,
            type=fc.type_name,
            cls=fc.cls,
            area=area,
            min_draft_deg=_draft_deg(fmin) if fin else None,
            max_draft_deg=_draft_deg(fmax) if fin else None,
            needs_draft=fc.cls == VERTICAL,
            undercut_frac=float(frac),
            undercut=flagged,
        ))
    stats["min_draft_deg"] = {k: (None if math.isinf(v) else v) for k, v in min_draft.items()}
    stats["vertical_area"] = stats["face_area"][VERTICAL]
    stats["vertical_faces"] = [f.number for f in faces_out if f.needs_draft]
    stats["vertical_bands"] = len(bands)
    stats["low_draft_area"] = low_draft_area
    stats["undercut_area"] = undercut_area
    stats["undercut_faces"] = [f.number for f in faces_out if f.undercut]
    loops = ch.loops
    stats["loops"] = {
        "count": len(loops),
        "closed": sum(1 for lp in loops if lp.closed),
        "open": sum(1 for lp in loops if not lp.closed),
        "lengths": [lp.length for lp in loops],
        "total_length": float(sum(lp.length for lp in loops)),
        "junctions": len(ch.junctions),
        "dangling_ends": len(ch.dangling),
        "max_gap_closed": ch.max_gap,
    }
    stats["segments"] = {k: sum(1 for s in segs if s.kind == k)
                         for k in (sg.EDGE, sg.VERTICAL, sg.SILHOUETTE, sg.SECTION, sg.BRIDGE)}
    stats["duplicate_silhouettes_dropped"] = n_dupes
    stats["short_segments_dropped"] = n_short
    stats["spurs_pruned"] = n_spurs
    stats["bridges"] = [{"start": b.start, "end": b.end, "length": b.length} for b in bridges]
    if stats["loops"]["open"]:
        warnings.append(f"{stats['loops']['open']} open parting-line chain(s): the line is not closed")
    if ch.junctions:
        warnings.append(f"{len(ch.junctions)} junction(s) where more than two parting segments meet")
    if not loops:
        warnings.append("no parting line found for this pull direction")
    timings["total"] = round(sum(timings.values()), 3)
    return Result(d, opts, faces_out, segs, ch, wires, stats, warnings, timings)


def _silhouette_edge_params(sil_segs, model: Model, ecs, tol: float) -> dict[int, list[float]]:
    """Edge parameters where silhouettes end on their face's boundary edges.

    Each such end is also moved onto the exact root of n.d along the edge.
    Clipping the silhouette polyline at the face boundary lands up to
    chord_tol / sin(angle) away from it when the silhouette meets the edge at a
    shallow angle; the edge pass uses the exact root, so both now meet."""
    out: dict[int, list[float]] = {}
    for s in sil_segs:
        if np.linalg.norm(s.start - s.end) < tol:
            continue  # closed silhouette
        fi = s.source["face"] - 1
        moved = False
        for end in (0, -1):
            P = s.points[end]
            best = None
            for ei in model.face_edges[fi]:
                ec = ecs[ei]
                if ec.degenerate:
                    continue
                curve = edge_curve(ec.edge)
                if curve is None:
                    continue
                proj = GeomAPI_ProjectPointOnCurve(pnt(P), curve, ec.t0, ec.t1)
                if proj.NbPoints() and proj.LowerDistance() < tol and (best is None or proj.LowerDistance() < best[0]):
                    best = (proj.LowerDistance(), ei, proj.LowerDistanceParameter())
            if best is None:
                continue
            _, ei, t = best
            ec = ecs[ei]
            uses = [u for u in ec.uses if u.fc.index == fi]
            if s.uv is not None and len(uses) > 1:  # seam: the use whose pcurve is at this end
                uses.sort(key=lambda u: float(np.linalg.norm(np.array(u.uv(t)) - s.uv[end])))
            root = _root_near(uses[0], ec, t, max(50 * tol, 0.2)) if uses else None
            if root is not None:
                q = ec.c3.Value(root)
                s.points[end] = (q.X(), q.Y(), q.Z())
                if s.uv is not None:
                    s.uv[end] = uses[0].uv(root)
                t, moved = root, True
            out.setdefault(ei, []).append(t)
        if moved:
            _refit_silhouette(s)
    return out


def _root_near(use, ec, t: float, reach: float):
    """Root of n.d along the edge (this face's side) within `reach` mm of t."""
    p, v = gp_Pnt(), gp_Vec()
    ec.c3.D1(t, p, v)
    speed = v.Magnitude()
    if speed <= 0:
        return None
    w = reach / speed
    lo, hi = max(ec.t0, t - w), min(ec.t1, t + w)
    ts = np.linspace(lo, hi, 33)
    fs = [use.f0(float(x)) for x in ts]
    best = None
    for k in range(len(ts) - 1):
        fa, fb = fs[k], fs[k + 1]
        if not (fa == fa and fb == fb) or fa * fb > 0:
            continue
        a, b = float(ts[k]), float(ts[k + 1])
        for _ in range(80):
            m = 0.5 * (a + b)
            fm = use.f0(m)
            if fm != fm:
                break
            if (fm > 0) == (fa > 0):
                a, fa = m, fm
            else:
                b = m
            if b - a <= 1e-13 * abs(ec.t1 - ec.t0):
                break
        r = 0.5 * (a + b)
        if best is None or abs(r - t) < abs(best - t):
            best = r
    return best


def _refit_silhouette(s: sg.Segment) -> None:
    """Rebuild a silhouette's curve after its end points moved."""
    pts = np.array(s.points, dtype=float)
    keep = list(range(len(pts)))
    # drop points the moved ends now overshoot
    while len(keep) > 2 and np.dot(pts[keep[1]] - pts[keep[0]], pts[keep[2]] - pts[keep[1]]) < 0:
        keep.pop(1)
    while len(keep) > 2 and np.dot(pts[keep[-1]] - pts[keep[-2]], pts[keep[-2]] - pts[keep[-3]]) < 0:
        keep.pop(-2)
    s.points = pts[keep]
    if s.uv is not None:
        s.uv = np.array(s.uv, dtype=float)[keep]
    curve = sg.interpolate(s.points)
    if curve is not None:
        s.curve, s.t0, s.t1 = curve, curve.FirstParameter(), curve.LastParameter()
        s.length = sg.curve_length(curve, s.t0, s.t1)


def _bridges(ch: Chaining, max_len: float, defl: float) -> list[sg.Segment]:
    """Straight segments joining dangling ends that are close together,
    nearest pairs first."""
    ends = list(ch.dangling)
    pairs = sorted((float(np.linalg.norm(ch.node_xyz[a] - ch.node_xyz[b])), a, b)
                   for i, a in enumerate(ends) for b in ends[i + 1:])
    used: set[int] = set()
    out = []
    for dist, a, b in pairs:
        if dist > max_len:
            break
        if a in used or b in used:
            continue
        used.update((a, b))
        curve = GC_MakeSegment(pnt(ch.node_xyz[a]), pnt(ch.node_xyz[b])).Value()
        out.append(sg.from_curve(sg.BRIDGE, curve, curve.FirstParameter(), curve.LastParameter(), defl))
    return out


def _spurs(ch: Chaining, segs, max_len: float) -> set[int]:
    """Short dead ends: paths from a dangling node through degree-2 nodes that
    end at a junction (or at another dangling node) within max_len."""
    sn = ch.seg_nodes
    inc: dict[int, list[int]] = {}
    for si, (a, b) in enumerate(sn):
        inc.setdefault(int(a), []).append(si)
        inc.setdefault(int(b), []).append(si)
    out: set[int] = set()
    for start in ch.dangling:
        path, length, node, prev = [], 0.0, start, None
        while True:
            nxt = [si for si in inc[node] if si != prev]
            if len(nxt) != 1:
                break
            si = nxt[0]
            path.append(si)
            length += segs[si].length
            if length >= max_len:
                path = []
                break
            a, b = int(sn[si][0]), int(sn[si][1])
            node = b if a == node else a
            prev = si
            if len(inc[node]) != 2:
                break
        if path and length < max_len:
            out.update(path)
    return out


def _within(xyz: np.ndarray, polys: list[np.ndarray], tol: float) -> bool:
    """True if sample points of polyline xyz all lie within tol of the polylines."""
    polys = [p for p in polys if len(p) > 1]
    if not polys:
        return False
    s0 = np.vstack([p[:-1] for p in polys])
    s1 = np.vstack([p[1:] for p in polys])
    idx = np.linspace(0, len(xyz) - 1, min(len(xyz), 15)).astype(int)
    P = xyz[idx]
    seg = s1 - s0
    L2 = np.maximum((seg * seg).sum(axis=1), 1e-30)
    tt = np.clip(((P[:, None, :] - s0[None]) * seg[None]).sum(axis=2) / L2[None], 0.0, 1.0)
    proj = s0[None] + tt[..., None] * seg[None]
    dist = np.linalg.norm(P[:, None, :] - proj, axis=2).min(axis=1)
    return bool(dist.max() < tol)


def _band_sections(model, fcs, ecs, bands, find, d, opts) -> list[sg.Segment]:
    out = []
    for root in bands:
        members = [fc for fc in fcs if fc.cls == VERTICAL and find(fc.index) == root]
        heights = []
        for fc in members:
            if len(fc.samples_xyz):
                heights.append(fc.samples_xyz @ d)
            for ei in model.face_edges[fc.index]:
                if not ecs[ei].degenerate:
                    heights.append(ecs[ei].pts @ d)
        if not heights:
            continue
        h = np.concatenate(heights)
        mid = 0.5 * (h.min() + h.max())
        comp = make_compound([fc.face for fc in members])
        sec = BRepAlgoAPI_Section(comp, gp_Pln(pnt(d * mid), gp_Dir(*d)), False)
        sec.Approximation(True)
        sec.Build()
        if not sec.IsDone():
            continue
        for e in map_list(indexed_map(sec.Shape(), TopAbs_EDGE), "edge"):
            c = edge_curve(e)
            if c is None:
                continue
            t0, t1 = BRep_Tool.Range_s(e)
            out.append(sg.from_curve(sg.SECTION, c, t0, t1, opts.point_defl,
                                     faces=[fc.index + 1 for fc in members]))
    return out


def _undercuts(fcs, caster: RayCaster, t: float, lin: float) -> np.ndarray:
    eps = 3.0 * lin
    origins, owner, fvals = [], [], []
    for fc in fcs:
        if len(fc.samples_xyz) == 0:
            continue
        origins.append(fc.samples_xyz + eps * fc.samples_n)
        owner.append(np.full(len(fc.samples_xyz), fc.index))
        fvals.append(fc.samples_f)
    if not origins:
        return np.zeros(len(fcs))
    own = np.concatenate(owner)
    f = np.concatenate(fvals)
    up, down = caster.cast(np.vstack(origins), eps)
    # A sample may release toward +d if it does not face -d, and vice versa;
    # it is an undercut when every direction it could release in is blocked.
    blocked = ((f <= -t) | up) & ((f >= t) | down)
    cnt = np.bincount(own, minlength=len(fcs))
    hit = np.bincount(own, weights=blocked.astype(float), minlength=len(fcs))
    return np.where(cnt > 0, hit / np.maximum(cnt, 1), 0.0)
