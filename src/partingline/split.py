"""Phase 2: split faces along the parting line and sort them into cavity / core.

1. Cut edges are built on their faces: silhouettes from their exact UV points
   (pcurve on the face), band generators and sections as 3D edges.  The model
   is cut with BRepFeat_SplitShape.
2. Faces of the split shape are grouped by flood fill across edges that are
   *not* on the parting line.  Each group takes the side its faces vote for
   (area-weighted sign of n.d); groups made only of vertical faces ask the ray
   caster which way they can release.  A face whose own vote disagrees with its
   group is reported as a conflict - that means the parting line is wrong there.
3. Each side is written as shells (faces keep their shared edges).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeVertex
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.BRepFeat import BRepFeat_SplitShape
from OCP.BRepLib import BRepLib
from OCP.GCPnts import GCPnts_AbscissaPoint
from OCP.Geom2dAPI import Geom2dAPI_Interpolate
from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCP.TColgp import TColgp_HArray1OfPnt2d
from OCP.TopAbs import TopAbs_IN, TopAbs_ON
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Shell
from OCP.TopTools import TopTools_SequenceOfShape
from OCP.gp import gp_Pnt2d

from . import segments as sg
from .faces import FaceCtx
from .model import Model
from .occ import make_compound, pnt
from .undercut import RayCaster, mesh_arrays, mesh_deflection

CAVITY_SIDE, CORE_SIDE = 1, -1


@dataclass
class SplitResult:
    shape: object
    model: Model                     # indexed split shape
    side: list[int]                  # per split face: +1 cavity, -1 core
    cavity_faces: list[int]          # 0-based indices into model.faces
    core_faces: list[int]
    areas: dict
    components: int
    conflicts: list[int]             # 1-based split-face numbers
    boundary_length: dict            # free-boundary length of each side's faces
    warnings: list[str] = field(default_factory=list)
    unsplit: list[int] = field(default_factory=list)  # 1-based split faces assigned whole

    def shells(self, which: str):
        faces = self.cavity_faces if which == "cavity" else self.core_faces
        return _shells([self.model.faces[i] for i in faces], self.model, faces)


def _edge_from_uv(uv: np.ndarray, face, bounds=None, verts=None) -> object | None:
    """Edge on `face` through UV points.  End points within 1e-7 of the UV
    domain bounds are snapped onto them, so a curve that runs into a periodic
    seam really touches it.  `verts` (start, end) are shared vertices: cut
    edges that meet must share them or the splitter cuts the boundary twice."""
    uv = np.array(uv, dtype=float)
    if bounds is not None:
        u0, u1, v0, v1 = bounds
        for p in (uv[0], uv[-1]):
            for k, (lo, hi) in enumerate(((u0, u1), (v0, v1))):
                span = max(hi - lo, 1e-300)
                if abs(p[k] - lo) < 1e-7 * span:
                    p[k] = lo
                elif abs(p[k] - hi) < 1e-7 * span:
                    p[k] = hi
    pts = [uv[0]]
    for p in uv[1:]:
        if np.linalg.norm(p - pts[-1]) > 1e-12:
            pts.append(p)
    if len(pts) < 2:
        return None
    arr = TColgp_HArray1OfPnt2d(1, len(pts))
    for i, p in enumerate(pts, start=1):
        arr.SetValue(i, gp_Pnt2d(float(p[0]), float(p[1])))
    algo = Geom2dAPI_Interpolate(arr, False, 1e-12)
    algo.Perform()
    if not algo.IsDone():
        return None
    c2d = algo.Curve()
    loc = TopLoc_Location()
    surf = BRep_Tool.Surface_s(face, loc)
    me = None
    if verts is not None and loc.IsIdentity():
        _cover(verts[0], surf.Value(*c2d.Value(c2d.FirstParameter()).Coord()))
        _cover(verts[1], surf.Value(*c2d.Value(c2d.LastParameter()).Coord()))
        me = BRepBuilderAPI_MakeEdge(c2d, surf, verts[0], verts[1], c2d.FirstParameter(), c2d.LastParameter())
    if me is None or not me.IsDone():
        me = BRepBuilderAPI_MakeEdge(c2d, surf, c2d.FirstParameter(), c2d.LastParameter())
    if not me.IsDone():
        return None
    edge = me.Edge()
    BRepLib.BuildCurves3d_s(edge)
    if not loc.IsIdentity():
        edge = TopoDS.Edge_s(edge.Moved(loc))
    return edge


def _cover(vertex, point) -> None:
    """Grow a shared vertex's tolerance to reach an edge end point.  Ends of
    cut edges sit on fused/imported boundary edges only to within those edges'
    tolerance (~1e-5 mm); a vertex that does not cover them is not shared and
    the face is not split."""
    gap = BRep_Tool.Pnt_s(vertex).Distance(point)
    if gap > BRep_Tool.Tolerance_s(vertex):
        BRep_Builder().UpdateVertex(vertex, 1.2 * gap + 1e-7)


def _edge_from_segment(seg: sg.Segment, verts=None):
    me = None
    if verts is not None:
        _cover(verts[0], seg.curve.Value(seg.t0))
        _cover(verts[1], seg.curve.Value(seg.t1))
        me = BRepBuilderAPI_MakeEdge(seg.curve, verts[0], verts[1], seg.t0, seg.t1)
    if me is None or not me.IsDone():
        me = BRepBuilderAPI_MakeEdge(seg.curve, seg.t0, seg.t1)
    return me.Edge() if me.IsDone() else None


def _face_containing(point: np.ndarray, faces, candidates, tol: float):
    for fi in candidates:
        face = faces[fi]
        surf = BRep_Tool.Surface_s(face)
        proj = GeomAPI_ProjectPointOnSurf(pnt(point), surf)
        if proj.NbPoints() == 0 or proj.LowerDistance() > tol:
            continue
        clf = BRepClass_FaceClassifier(face, pnt(point), tol)
        if clf.State() in (TopAbs_IN, TopAbs_ON):
            return fi
    return None


def split_and_assign(model: Model, result) -> SplitResult:
    warnings: list[str] = []
    d = result.pull
    tol = max(1e-6, 1e-6 * model.diag)
    vertical = [f.number - 1 for f in result.faces if f.cls == "vertical"]

    # one shared vertex per chain node
    ch = result.chaining
    builder = BRep_Builder()
    node_vertex: dict[int, object] = {}

    def verts_of(si: int):
        out = []
        for k in ch.seg_nodes[si]:
            k = int(k)
            if k not in node_vertex:
                v = BRepBuilderAPI_MakeVertex(pnt(ch.node_xyz[k])).Vertex()
                builder.UpdateVertex(v, float(ch.node_tol[k]) + 1e-6)
                node_vertex[k] = v
            out.append(node_vertex[k])
        return out

    per_face: dict[int, list] = {}
    loose = TopTools_SequenceOfShape()
    for si, seg in enumerate(result.segments):
        if seg.kind == sg.SILHOUETTE and seg.uv is not None:
            fi = seg.source["face"] - 1
            fc = FaceCtx(fi, model.faces[fi])
            e = _edge_from_uv(seg.uv, model.faces[fi], (fc.u0, fc.u1, fc.v0, fc.v1), verts_of(si))
            if e is None:
                warnings.append(f"could not build silhouette edge on face {fi + 1}")
                continue
            per_face.setdefault(fi, []).append(e)
        elif seg.source.get("generator") or seg.kind == sg.SECTION:
            e = _edge_from_segment(seg, verts_of(si))
            if e is None:
                continue
            mid = seg.points[len(seg.points) // 2]
            fi = _face_containing(mid, model.faces, vertical, 1e3 * tol)
            ends_on = fi is not None and all(
                _face_containing(p, model.faces, [fi], 1e3 * tol) is not None for p in (seg.start, seg.end))
            if ends_on:
                per_face.setdefault(fi, []).append(e)
            else:
                loose.Append(e)

    ss = BRepFeat_SplitShape(model.shape)
    for fi, edges in per_face.items():
        for e in edges:
            ss.Add(e, model.faces[fi])
    if loose.Size():
        ss.Add(loose)
    ss.Build()
    if not ss.IsDone():
        raise RuntimeError("face splitting failed (BRepFeat_SplitShape)")
    shape = ss.Shape()
    smodel = Model(shape, model.import_info, model.heal_info)

    # Faces the splitter could not cut still have a silhouette running through
    # their interior.  They would join the two halves in the flood fill, so
    # they are kept out of it and assigned on their own.
    unsplit: set[int] = set()
    for seg in result.segments:
        if seg.kind != sg.SILHOUETTE:
            continue
        face = model.faces[seg.source["face"] - 1]
        pieces = list(ss.Modified(face)) or [face]
        probe = seg.curve.Value(0.5 * (seg.t0 + seg.t1))
        for piece in pieces:
            if BRepClass_FaceClassifier(TopoDS.Face_s(piece), probe, 1e-6).State() == TopAbs_IN:
                k = smodel.fmap.FindIndex(piece) - 1
                if k >= 0:
                    unsplit.add(k)
    if unsplit:
        warnings.append(f"{len(unsplit)} face(s) could not be split along the parting line and were "
                        f"assigned whole: {sorted(k + 1 for k in unsplit)[:10]}")

    # -- which edges of the split shape lie on the parting line -------------
    polys = [s.points for s in result.segments if len(s.points) > 1]
    s0 = np.vstack([p[:-1] for p in polys]) if polys else np.zeros((0, 3))
    s1 = np.vstack([p[1:] for p in polys]) if polys else np.zeros((0, 3))
    on_tol = max(5e-3, 0.5 * result.options.gap_tol)
    parting_edge = np.zeros(len(smodel.edges), dtype=bool)
    for ei, e in enumerate(smodel.edges):
        if BRep_Tool.Degenerated_s(e) or len(s0) == 0:
            continue
        c = BRepAdaptor_Curve(e)
        ts = np.linspace(c.FirstParameter(), c.LastParameter(), 5)[1:-1]
        P = np.array([[c.Value(float(t)).X(), c.Value(float(t)).Y(), c.Value(float(t)).Z()] for t in ts])
        parting_edge[ei] = _max_dist(P, s0, s1) < on_tol

    # -- flood fill across non-parting edges --------------------------------
    n = smodel.n_faces
    t = math.sin(math.radians(result.options.draft_tol_deg))
    vote = np.zeros(n)
    area = np.zeros(n)
    fcs = []
    for i, face in enumerate(smodel.faces):
        fc = FaceCtx(i, face)
        fc.set_pull(d)
        fc.sample_grid()
        fc.update_f(d)
        fcs.append(fc)
        area[i] = fc.area
        f = fc.samples_f
        strong = f[np.abs(f) > t]
        vote[i] = float(np.sign(strong).mean()) if len(strong) else 0.0

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # Never join faces that clearly face opposite ways: a crumb of geometry the
    # line does not resolve (slivers next to the line) must not merge the halves.
    n_off_line = 0
    for ei, uses in enumerate(smodel.edge_uses):
        if parting_edge[ei] or len(uses) != 2 or uses[0].face in unsplit or uses[1].face in unsplit:
            continue
        if vote[uses[0].face] * vote[uses[1].face] < -0.25:
            n_off_line += 1
            continue
        a, b = find(uses[0].face), find(uses[1].face)
        if a != b:
            parent[b] = a

    roots = sorted({find(i) for i in range(n)})
    side = [0] * n
    caster = None
    for r in roots:
        members = [i for i in range(n) if find(i) == r]
        total = sum(vote[i] * area[i] for i in members)
        if abs(total) > 1e-12:
            s = CAVITY_SIDE if total > 0 else CORE_SIDE
        else:
            # all vertical: ask which way the steel can leave
            if caster is None:
                lin = mesh_deflection(smodel.diag)
                V, T, owner = mesh_arrays(smodel.shape, smodel.faces, lin)
                caster = (RayCaster(V, T, d, owner), lin)
            s = _release_side(fcs[members[0]], caster, result.options.vertical, set(members))
        for i in members:
            side[i] = s

    if n_off_line:
        warnings.append(f"{n_off_line} edge(s) off the parting line separate faces that face opposite ways "
                        "(small regions the line does not resolve)")
    conflicts = [i + 1 for i in range(n) if abs(vote[i]) > 0.5 and np.sign(vote[i]) != side[i]]
    if conflicts:
        warnings.append(f"{len(conflicts)} split face(s) face the other way from their half: {conflicts[:10]}")

    cav = [i for i in range(n) if side[i] == CAVITY_SIDE]
    core = [i for i in range(n) if side[i] == CORE_SIDE]
    areas = {"cavity": float(area[cav].sum()), "core": float(area[core].sum()), "total": float(area.sum())}
    blen = {"cavity": _free_length(smodel, set(cav)), "core": _free_length(smodel, set(core))}
    return SplitResult(shape, smodel, side, cav, core, areas, len(roots), conflicts, blen, warnings,
                       sorted(k + 1 for k in unsplit))


def _max_dist(P: np.ndarray, s0: np.ndarray, s1: np.ndarray) -> float:
    seg = s1 - s0
    L2 = np.maximum((seg * seg).sum(axis=1), 1e-30)
    out = 0.0
    for p in P:
        tt = np.clip(((p - s0) * seg).sum(axis=1) / L2, 0.0, 1.0)
        out = max(out, float(np.linalg.norm(s0 + tt[:, None] * seg - p, axis=1).min()))
    return out


def _release_side(fc: FaceCtx, caster, rule: str, own: set[int]) -> int:
    rc, lin = caster
    if len(fc.samples_xyz) == 0:
        return CORE_SIDE if rule != "bottom" else CAVITY_SIDE
    k = len(fc.samples_xyz) // 2
    up, down = rc.cast(fc.samples_xyz[k] + 3 * lin * fc.samples_n[k], 0.5 * lin, own)
    if not up[0] and down[0]:
        return CAVITY_SIDE
    if not down[0] and up[0]:
        return CORE_SIDE
    return CORE_SIDE if rule != "bottom" else CAVITY_SIDE


def _free_length(smodel: Model, faces: set[int]) -> float:
    """Length of edges used by exactly one face of the set (its open boundary)."""
    total = 0.0
    for ei, uses in enumerate(smodel.edge_uses):
        k = sum(1 for u in uses if u.face in faces)
        if k == 1 and not BRep_Tool.Degenerated_s(smodel.edges[ei]):
            total += GCPnts_AbscissaPoint.Length_s(BRepAdaptor_Curve(smodel.edges[ei]))
    return total


def _shells(faces, smodel: Model, idx: list[int]):
    """Group faces into shells by shared edges."""
    if not faces:
        return make_compound([])
    parent = {i: i for i in idx}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    members = set(idx)
    for uses in smodel.edge_uses:
        fs = [u.face for u in uses if u.face in members]
        for a, b in zip(fs, fs[1:]):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
    groups: dict[int, list[int]] = {}
    for i in idx:
        groups.setdefault(find(i), []).append(i)
    builder = BRep_Builder()
    shells = []
    for g in groups.values():
        sh = TopoDS_Shell()
        builder.MakeShell(sh)
        for i in g:
            builder.Add(sh, smodel.faces[i])
        shells.append(sh)
    return make_compound(shells)
