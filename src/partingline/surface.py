"""Phase 4: a simple parting surface from the parting line.

sweep  (default) every parting segment is ruled outward, perpendicular to the
       pull, by `distance`.  Convex corners of the line get a flat sector so
       the surface has no gaps; concave corners and tight curves make the
       strips overlap, which is reported as self-intersection (not hidden).
ruled  every outer loop is ruled to a rectangle in a plane normal to the pull
       (the loops' bounding box grown by `distance`, at the loops' mean
       height).  Points map along rays from the loop's centroid, so loops that
       are not star-shaped around it fold over - again reported.

Loops nested inside another loop (seen along the pull, e.g. a bore) are
holes: they are capped with a flat face when planar, a filling otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_Sewing,
)
from OCP.BRepFill import BRepFill
from OCP.BRepGProp import BRepGProp
from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeFilling
from OCP.GC import GC_MakeArcOfCircle, GC_MakeSegment
from OCP.GeomAbs import GeomAbs_C0
from OCP.GeomAPI import GeomAPI_Interpolate
from OCP.GProp import GProp_GProps
from OCP.TColgp import TColgp_HArray1OfPnt
from OCP.TColStd import TColStd_HArray1OfReal
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCP.gp import gp_Ax2, gp_Circ, gp_Dir, gp_Pnt, gp_Vec

from .occ import indexed_map, make_compound, map_list, orthonormal_basis, pnt
from .undercut import mesh_arrays

MODES = ("sweep", "ruled")


@dataclass
class SurfaceResult:
    shape: object
    mode: str
    distance: float
    faces: int
    area: float
    outer_loops: list[int]
    filled_loops: list[int]
    self_intersections: int
    touching: int                # boundary contacts that are not crossings (not counted: 0)
    height: float | None = None
    warnings: list[str] = field(default_factory=list)

    def report(self) -> dict:
        return {
            "mode": self.mode,
            "distance": self.distance,
            "plane_height": self.height,
            "faces": self.faces,
            "area": self.area,
            "outer_loops": self.outer_loops,
            "filled_loops": self.filled_loops,
            "self_intersections": self.self_intersections,
            "touching_contacts": self.touching,
            "warnings": self.warnings,
        }

    def summary_line(self) -> str:
        si = (f"{self.self_intersections} self-intersecting face pair(s)  ** check **" if self.self_intersections
              else "no self-intersections")
        return (f"Surface : {self.mode} {self.distance:g} mm, {self.faces} face(s), {self.area:.1f} mm2, "
                f"{len(self.filled_loops)} hole(s) capped, {si}")


# -- helpers ------------------------------------------------------------------

def _poly2(points: np.ndarray, e1, e2) -> np.ndarray:
    return np.stack([points @ e1, points @ e2], axis=1)


def _signed_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _inside(pt: np.ndarray, poly: np.ndarray) -> bool:
    x, y = pt
    xs, ys = poly[:, 0], poly[:, 1]
    xs2, ys2 = np.roll(xs, -1), np.roll(ys, -1)
    cross = (ys > y) != (ys2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = xs + (y - ys) * (xs2 - xs) / (ys2 - ys)
    return bool(np.count_nonzero(cross & (x < xint)) % 2)


def _interp(points: list, params: list):
    arr = TColgp_HArray1OfPnt(1, len(points))
    par = TColStd_HArray1OfReal(1, len(points))
    for i, (p, t) in enumerate(zip(points, params), start=1):
        arr.SetValue(i, pnt(p))
        par.SetValue(i, float(t))
    algo = GeomAPI_Interpolate(arr, par, False, 1e-9)
    algo.Perform()
    return algo.Curve() if algo.IsDone() else None


def _ruled(curve, t0: float, t1: float, ts: np.ndarray, targets: list[np.ndarray]):
    """Ruled face between curve[t0,t1] and a curve through `targets` at `ts`."""
    keep = [0]
    for i in range(1, len(ts)):
        if np.linalg.norm(targets[i] - targets[keep[-1]]) > 1e-7 and ts[i] - ts[keep[-1]] > 1e-12:
            keep.append(i)
    if len(keep) < 2:
        return None
    if keep[-1] != len(ts) - 1:
        keep[-1] = len(ts) - 1
    other = _interp([targets[i] for i in keep], [ts[i] for i in keep])
    if other is None:
        return None
    e1 = BRepBuilderAPI_MakeEdge(curve, t0, t1)
    e2 = BRepBuilderAPI_MakeEdge(other, other.FirstParameter(), other.LastParameter())
    if not (e1.IsDone() and e2.IsDone()):
        return None
    try:
        return BRepFill.Face_s(e1.Edge(), e2.Edge())
    except Exception:
        return None


def _samples(seg, n_min: int = 9, step: float | None = None) -> np.ndarray:
    n = n_min if step is None else max(n_min, int(math.ceil(seg.length / step)) + 1)
    return np.linspace(seg.t0, seg.t1, min(n, 400))


def _point_tangent(curve, t: float):
    p, v = gp_Pnt(), gp_Vec()
    curve.D1(float(t), p, v)
    return np.array([p.X(), p.Y(), p.Z()]), np.array([v.X(), v.Y(), v.Z()])


# -- main -----------------------------------------------------------------------

def parting_surface(model, result, mode: str = "sweep", distance: float = 20.0,
                    height: float | None = None, check: bool = True) -> SurfaceResult:
    if mode not in MODES:
        raise ValueError(f"surface mode must be one of {MODES}")
    d = result.pull
    e1, e2 = orthonormal_basis(d)
    loops = result.loops
    segs = result.segments
    warnings: list[str] = []

    polys = [_poly2(lp.points, e1, e2) for lp in loops]
    areas = [abs(_signed_area(p)) for p in polys]
    depth = []
    for i, lp in enumerate(loops):
        if not lp.closed:
            depth.append(-1)
            continue
        probe = polys[i][0]
        depth.append(sum(1 for j, q in enumerate(polys)
                         if j != i and loops[j].closed and areas[j] > areas[i] and _inside(probe, q)))
    outer = [i for i, k in enumerate(depth) if k >= 0 and k % 2 == 0]
    holes = [i for i, k in enumerate(depth) if k >= 0 and k % 2 == 1]
    for i, k in enumerate(depth):
        if k < 0:
            warnings.append(f"loop {i + 1} is open; no surface built from it")

    faces = []
    plane_h = None
    if mode == "sweep":
        for i in outer:
            faces += _sweep_loop(loops[i], segs, polys[i], d, e1, e2, distance, warnings)
    else:
        allpts = np.vstack([loops[i].points for i in outer]) if outer else np.zeros((0, 3))
        if len(allpts):
            plane_h = float(np.mean(allpts @ d)) if height is None else float(height)
            p2 = _poly2(allpts, e1, e2)
            lo, hi = p2.min(axis=0) - distance, p2.max(axis=0) + distance
            for i in outer:
                faces += _ruled_loop(loops[i], segs, polys[i], d, e1, e2, lo, hi, plane_h, warnings)

    for i in holes:
        f = _cap(result.wires[i] if i < len(result.wires) else None, loops[i], d, model.diag)
        if f is None:
            warnings.append(f"could not cap hole loop {i + 1}")
        else:
            faces.append(f)

    if not faces:
        return SurfaceResult(make_compound([]), mode, distance, 0, 0.0, [i + 1 for i in outer],
                             [i + 1 for i in holes], 0, 0, plane_h, warnings)

    sew = BRepBuilderAPI_Sewing(max(2 * result.options.gap_tol, 1e-4))
    for f in faces:
        sew.Add(f)
    sew.Perform()
    shape = sew.SewedShape()
    if shape.IsNull():
        shape = make_compound(faces)

    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, props, 1e-4)
    n_si, n_touch = 0, 0
    if check:
        n_si, n_touch = _self_intersections(shape, model.diag)
        if n_si:
            warnings.append(f"parting surface intersects itself ({n_si} crossing face pair(s)); "
                            "reduce the distance or use the other mode")
    return SurfaceResult(shape, mode, distance, indexed_map(shape, TopAbs_FACE).Extent(), abs(props.Mass()),
                         [i + 1 for i in outer], [i + 1 for i in holes], n_si, n_touch, plane_h, warnings)


def _self_intersections(shape, diag: float) -> tuple[int, int]:
    """(face pairs that cross each other, 0).

    Checked on a triangulation: OpenCascade's exact BRepAlgoAPI_Check takes
    minutes on swept B-spline strips at oblique pulls.  Triangle pairs that
    share a vertex (neighbours along a sewn edge) only touch and are skipped;
    a crossing is an edge of one triangle passing through the interior of the
    other."""
    faces = map_list(indexed_map(shape, TopAbs_FACE), "face")
    V, T, owner = mesh_arrays(shape, faces, max(1e-3 * diag, 1e-3), 0.3)
    if len(T) < 2:
        return 0, 0
    q = max(1e-6 * diag, 1e-7)
    # merge coincident mesh nodes (each face is meshed with its own nodes)
    _, first, vid = np.unique(np.round(V / q).astype(np.int64), axis=0,
                              return_index=True, return_inverse=True)
    T = vid.reshape(-1)[T]
    P = V[first]
    A, B, C = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    lo = np.minimum(np.minimum(A, B), C)
    hi = np.maximum(np.maximum(A, B), C)
    pairs = _bbox_pairs(lo, hi)
    if not len(pairs):
        return 0, 0
    i, j = pairs[:, 0], pairs[:, 1]
    shared = (T[i][:, :, None] == T[j][:, None, :]).any(axis=(1, 2))
    i, j = i[~shared], j[~shared]
    eps = 1e-6
    hit = np.zeros(len(i), dtype=bool)
    for a, b in ((i, j), (j, i)):
        for k in range(3):
            p0, p1 = P[T[a, k]], P[T[a, (k + 1) % 3]]
            hit |= _segment_crosses(p0, p1, A[b], B[b], C[b], eps)
    fp = {tuple(sorted((int(owner[x]), int(owner[y])))) for x, y in zip(i[hit], j[hit])}
    return len(fp), 0


def _bbox_pairs(lo: np.ndarray, hi: np.ndarray, chunk: int = 2_000_000) -> np.ndarray:
    """Index pairs (i < j) of overlapping boxes: sweep along x, then test y and z."""
    order = np.argsort(lo[:, 0])
    lo, hi = lo[order], hi[order]
    # boxes j > i overlap i in x while lo_x[j] <= hi_x[i]
    stop = np.searchsorted(lo[:, 0], hi[:, 0], side="right")
    counts = np.maximum(stop - np.arange(len(lo)) - 1, 0)
    out = []
    i0 = 0
    while i0 < len(lo):
        csum = np.cumsum(counts[i0:])
        i1 = i0 + max(1, int(np.searchsorted(csum, chunk)))
        c = counts[i0:i1]
        i = np.repeat(np.arange(i0, i1), c)
        start = np.repeat(np.cumsum(c) - c, c)
        j = i + 1 + (np.arange(len(i)) - start)
        ok = ((lo[i, 1:] <= hi[j, 1:]) & (lo[j, 1:] <= hi[i, 1:])).all(axis=1)
        out.append(np.stack([order[i[ok]], order[j[ok]]], axis=1))
        i0 = i1
    return np.vstack(out) if out else np.zeros((0, 2), dtype=np.int64)


def _segment_crosses(p0, p1, a, b, c, eps: float) -> np.ndarray:
    """Does segment p0-p1 pass through the interior of triangle abc (vectorised)?"""
    d = p1 - p0
    e1, e2 = b - a, c - a
    h = np.cross(d, e2)
    det = (e1 * h).sum(axis=1)
    ok = np.abs(det) > 1e-14 * (np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) *
                                np.linalg.norm(d, axis=1) + 1e-300)
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = p0 - a
    u = (s * h).sum(axis=1) * inv
    qv = np.cross(s, e1)
    v = (d * qv).sum(axis=1) * inv
    t = (e2 * qv).sum(axis=1) * inv
    return ok & (u > eps) & (v > eps) & (u + v < 1 - eps) & (t > eps) & (t < 1 - eps)


def _outward(tangent: np.ndarray, orient: float, e1, e2) -> np.ndarray | None:
    t2 = np.array([tangent @ e1, tangent @ e2])
    n = np.linalg.norm(t2)
    if n < 1e-9 * (np.linalg.norm(tangent) + 1e-300):
        return None  # segment runs along the pull (a band generator)
    t2 /= n
    o2 = orient * np.array([t2[1], -t2[0]])
    return o2[0] * e1 + o2[1] * e2


def _sweep_loop(loop, segs, poly2, d, e1, e2, D: float, warnings) -> list:
    orient = 1.0 if _signed_area(poly2) > 0 else -1.0
    faces = []
    ends = []  # per item: (start point, outward at start, end point, outward at end)
    pending = []
    step = max(D / 4.0, 1e-3)
    for si, rev in loop.items:
        s = segs[si]
        ts = _samples(s, 9, step)
        pts, outs = [], []
        for t in ts:
            p, v = _point_tangent(s.curve, t)
            o = _outward(-v if rev else v, orient, e1, e2)
            pts.append(p)
            outs.append(o)
        pending.append((s, ts, pts, outs, rev))
    # points where the line runs along the pull (end of a vertical arc) take
    # the direction of the nearest defined point on the same segment ...
    for k, (s, ts, pts, outs, rev) in enumerate(pending):
        known = [i for i, o in enumerate(outs) if o is not None]
        if known and len(known) < len(outs):
            outs = [o if o is not None else outs[min(known, key=lambda j: abs(j - i))]
                    for i, o in enumerate(outs)]
            pending[k] = (s, ts, pts, outs, rev)
    # ... and whole band generators borrow it from a neighbouring segment
    n = len(pending)
    for k in range(n):
        s, ts, pts, outs, rev = pending[k]
        if any(o is None for o in outs):
            borrow = None
            for j in list(range(k - 1, -1, -1)) + list(range(n - 1, k, -1)):
                po = pending[j][3]
                cand = [o for o in po if o is not None]
                if cand:
                    prev_rev = pending[j][4]
                    borrow = cand[0] if prev_rev else cand[-1]
                    break
            if borrow is None:
                continue
            outs = [borrow if o is None else o for o in outs]
            pending[k] = (s, ts, pts, outs, rev)
    for s, ts, pts, outs, rev in pending:
        if any(o is None for o in outs):
            continue
        f = _ruled(s.curve, s.t0, s.t1, ts, [p + D * o for p, o in zip(pts, outs)])
        if f is None:
            warnings.append(f"could not sweep segment from {s.source}")
        else:
            faces.append(f)
        # traversal order of this item's end points
        if rev:
            ends.append((pts[-1], outs[-1], pts[0], outs[0]))
        else:
            ends.append((pts[0], outs[0], pts[-1], outs[-1]))
    # corner sectors
    for k in range(len(ends)):
        _, _, p_end, o_a = ends[k]
        p_next, o_b, _, _ = ends[(k + 1) % len(ends)]
        if k == len(ends) - 1 and not loop.closed:
            break
        cross = float(np.dot(np.cross(o_a, o_b), d))
        ang = math.atan2(cross, float(np.dot(o_a, o_b)))
        if abs(ang) < math.radians(0.5) or ang * orient < 0:
            continue  # aligned, or a concave corner (strips overlap: reported by the check)
        P = 0.5 * (p_end + p_next)
        f = _sector(P, o_a, o_b, D, d, ang)
        if f is not None:
            faces.append(f)
    return faces


def _sector(P, o_a, o_b, D, d, ang):
    qa, qb = P + D * o_a, P + D * o_b
    axis = gp_Dir(*(d if ang > 0 else -d))
    circ = gp_Circ(gp_Ax2(pnt(P), axis, gp_Dir(*o_a)), D)
    try:
        arc = GC_MakeArcOfCircle(circ, pnt(qa), pnt(qb), True).Value()
    except Exception:
        return None
    w = BRepBuilderAPI_MakeWire()
    w.Add(BRepBuilderAPI_MakeEdge(GC_MakeSegment(pnt(P), pnt(qa)).Value()).Edge())
    w.Add(BRepBuilderAPI_MakeEdge(arc).Edge())
    w.Add(BRepBuilderAPI_MakeEdge(GC_MakeSegment(pnt(qb), pnt(P)).Value()).Edge())
    if not w.IsDone():
        return None
    mf = BRepBuilderAPI_MakeFace(w.Wire(), True)
    return mf.Face() if mf.IsDone() else None


def _ray_rect(c: np.ndarray, theta: float, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    v = np.array([math.cos(theta), math.sin(theta)])
    best = math.inf
    for k in range(2):
        if abs(v[k]) > 1e-15:
            for bound in (lo[k], hi[k]):
                s = (bound - c[k]) / v[k]
                if s > 0:
                    q = c + s * v
                    j = 1 - k
                    if lo[j] - 1e-9 <= q[j] <= hi[j] + 1e-9:
                        best = min(best, s)
    return c + best * v


def _ruled_loop(loop, segs, poly2, d, e1, e2, lo, hi, h, warnings) -> list:
    """Rule each segment to the rectangle [lo, hi] (in e1/e2) at height h."""
    pts = poly2[:-1] if len(poly2) > 2 and np.allclose(poly2[0], poly2[-1]) else poly2
    a = _signed_area(pts)
    cx = float(np.sum((pts[:, 0] + np.roll(pts[:, 0], -1)) * (pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]))) / (6 * a)
    cy = float(np.sum((pts[:, 1] + np.roll(pts[:, 1], -1)) * (pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]))) / (6 * a)
    c = np.array([cx, cy])
    corners = [np.array(x) for x in ((lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1]))]
    corner_ang = [math.atan2(q[1] - c[1], q[0] - c[0]) for q in corners]

    def theta_of(P):
        p2 = np.array([P @ e1, P @ e2])
        return math.atan2(p2[1] - c[1], p2[0] - c[0])

    def lift(q2):
        return q2[0] * e1 + q2[1] * e2 + h * d

    faces = []
    for si, _rev in loop.items:
        s = segs[si]
        ts = list(np.linspace(s.t0, s.t1, max(17, min(400, int(s.length / 0.5) + 1))))
        # split where the ray from the centroid passes a rectangle corner
        cuts = []
        th = [theta_of(_point_tangent(s.curve, t)[0]) for t in ts]
        for k in range(len(ts) - 1):
            for ca in corner_ang:
                da = math.remainder(th[k] - ca, 2 * math.pi)
                db = math.remainder(th[k + 1] - ca, 2 * math.pi)
                if da * db < 0 and abs(da - db) < math.pi:
                    lo_t, hi_t, f_lo = ts[k], ts[k + 1], da
                    for _ in range(50):
                        mid = 0.5 * (lo_t + hi_t)
                        fm = math.remainder(theta_of(_point_tangent(s.curve, mid)[0]) - ca, 2 * math.pi)
                        if (fm < 0) == (f_lo < 0):
                            lo_t, f_lo = mid, fm
                        else:
                            hi_t = mid
                    cuts.append(0.5 * (lo_t + hi_t))
        bounds = [s.t0, *sorted(cuts), s.t1]
        for ta, tb in zip(bounds[:-1], bounds[1:]):
            if tb - ta < 1e-9 * (s.t1 - s.t0):
                continue
            sub = np.linspace(ta, tb, max(9, int(len(ts) * (tb - ta) / (s.t1 - s.t0)) + 2))
            targets = []
            for t in sub:
                P = _point_tangent(s.curve, t)[0]
                targets.append(lift(_ray_rect(c, theta_of(P), lo, hi)))
            if max(np.linalg.norm(q - targets[0]) for q in targets) < 1e-7:
                f = _triangle(_point_tangent(s.curve, ta)[0], _point_tangent(s.curve, tb)[0], targets[0])
            else:
                f = _ruled(s.curve, ta, tb, sub, targets)
            if f is None:
                warnings.append(f"could not rule segment from {s.source}")
            else:
                faces.append(f)
    return faces


def _triangle(a, b, c):
    """Flat face for a segment along the pull ruled to a single point."""
    w = BRepBuilderAPI_MakeWire()
    for p, q in ((a, b), (b, c), (c, a)):
        w.Add(BRepBuilderAPI_MakeEdge(pnt(p), pnt(q)).Edge())
    if not w.IsDone():
        return None
    mf = BRepBuilderAPI_MakeFace(w.Wire(), True)
    return mf.Face() if mf.IsDone() else None


def _cap(wire, loop, d, diag):
    if wire is None:
        return None
    h = loop.points @ d
    if float(np.ptp(h)) < 1e-6 * max(diag, 1.0):
        mf = BRepBuilderAPI_MakeFace(wire, True)
        if mf.IsDone():
            return mf.Face()
    fill = BRepOffsetAPI_MakeFilling()
    for e in map_list(indexed_map(wire, TopAbs_EDGE), "edge"):
        fill.Add(e, GeomAbs_C0)
    try:
        fill.Build()
    except Exception:
        return None
    if not fill.IsDone():
        return None
    faces = map_list(indexed_map(fill.Shape(), TopAbs_FACE), "face")
    return faces[0] if faces else None
