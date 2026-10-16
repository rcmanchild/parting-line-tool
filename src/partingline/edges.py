"""Parting status along model edges.

For every edge we look at each of its face-uses (a seam edge has two uses on
the same face) and compute the *side* sigma of that use at points along the
edge:

    +1  the face material next to the edge goes with the cavity (n.d > 0)
    -1  it goes with the core (n.d < 0)
     0  the face is a vertical (zero-draft) face

Whole faces classified cavity/core/vertical have a constant sigma.  Mixed faces
use the local sign of n.d; where n.d is ~0 right at the edge (a smooth
silhouette lying exactly on the edge, e.g. a cylinder seam) we step a little
into the face and use the sign found there.

An edge stretch is a parting edge when the two sides have sigma +1 and -1.
That covers sharp edges (a prop trailing edge) and smooth silhouettes that run
along an edge.  Stretches between a vertical face (0) and a +/-1 face are kept
as VTOP / VBOT candidates for the vertical-band rule.
"""

from __future__ import annotations

import math

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Curve2d
from OCP.GeomAbs import GeomAbs_BezierCurve, GeomAbs_BSplineCurve, GeomAbs_Circle, GeomAbs_Line
from OCP.TopAbs import TopAbs_REVERSED
from OCP.TopoDS import TopoDS, TopoDS_Edge
from OCP.gp import gp_Pnt2d, gp_Vec2d

from .faces import CAVITY, CORE, MIXED, VERTICAL, FaceCtx

NONE, EDGE, VTOP, VBOT = 0, 1, 2, 3
T_SMALL = 1e-4  # |n.d| below this at an edge -> look inside the face for the side


class UseEval:
    """One face-use of an edge."""

    def __init__(self, fc: FaceCtx, edge: TopoDS_Edge, orientation):
        self.fc = fc
        self.rev = orientation == TopAbs_REVERSED
        self.pc = BRepAdaptor_Curve2d(TopoDS.Edge_s(edge.Oriented(orientation)), fc.fwd)
        self._p = gp_Pnt2d()
        self._v = gp_Vec2d()

    def uv(self, t: float) -> tuple[float, float]:
        p = self.pc.Value(t)
        return p.X(), p.Y()

    def f0(self, t: float) -> float:
        u, v = self.uv(t)
        return self.fc.f_safe(u, v)

    def inward(self, t: float):
        """(point, unit 3D direction into the face across the edge, face normal)."""
        self.pc.D1(t, self._p, self._v)
        u, v = self._p.X(), self._p.Y()
        tu, tv = self._v.X(), self._v.Y()
        if self.rev:
            tu, tv = -tu, -tv
        p, n = self.fc.point_normal(u, v)
        du, dv = self.fc._Du, self.fc._Dv
        w = np.array([du.X() * -tv + dv.X() * tu, du.Y() * -tv + dv.Y() * tu, du.Z() * -tv + dv.Z() * tu])
        if n is not None:
            w -= (w @ n) * n  # keep it tangent to the face
        norm = np.linalg.norm(w)
        return p, (w / norm if norm > 0 else w), n

    def sigma(self, t: float, delta: float) -> int:
        cls = self.fc.cls
        if cls == VERTICAL:
            return 0
        if cls == CAVITY:
            return 1
        if cls == CORE:
            return -1
        self.pc.D1(t, self._p, self._v)
        u, v = self._p.X(), self._p.Y()
        f0 = self.fc.f(u, v)
        if f0 == f0 and abs(f0) >= T_SMALL:
            return 1 if f0 > 0 else -1
        # Material lies to the left of the pcurve traversed in its orientation.
        tu, tv = self._v.X(), self._v.Y()
        if self.rev:
            tu, tv = -tu, -tv
        wu, wv = -tv, tu
        length = self.fc.uv_dir_length(u, v, wu, wv)
        if not length > 0.0:
            return 0
        for dist in (delta, 10.0 * delta, 100.0 * delta):
            lam = dist / length
            fi = self.fc.f(u + lam * wu, v + lam * wv)
            if fi == fi and abs(fi) > 1e-10:
                return 1 if fi > 0 else -1
        return 0


def _sample_count(c3: BRepAdaptor_Curve) -> int:
    t = c3.GetType()
    if t == GeomAbs_Line:
        return 9
    if t == GeomAbs_Circle:
        span = abs(c3.LastParameter() - c3.FirstParameter())
        return max(9, int(math.ceil(48 * span / (2 * math.pi))) + 1)
    if t in (GeomAbs_BSplineCurve, GeomAbs_BezierCurve):
        try:
            n = 3 * (c3.NbKnots() + c3.Degree())
        except Exception:
            n = 33
        return int(min(129, max(17, n)))
    return 33


class EdgeCtx:
    def __init__(self, index: int, edge: TopoDS_Edge, uses, fcs: list[FaceCtx]):
        self.index = index
        self.edge = edge
        self.degenerate = BRep_Tool.Degenerated_s(edge)
        self.uses: list[UseEval] = []
        self.intervals: list[tuple[float, float, int, int]] = []  # (ta, tb, status, vertical face)
        if self.degenerate:
            return
        self.c3 = BRepAdaptor_Curve(edge)
        self.t0, self.t1 = self.c3.FirstParameter(), self.c3.LastParameter()
        self.ts = np.linspace(self.t0, self.t1, _sample_count(self.c3))
        pts = []
        for t in self.ts:
            p = self.c3.Value(float(t))
            pts.append((p.X(), p.Y(), p.Z()))
        self.pts = np.array(pts)
        for use in uses:
            try:
                self.uses.append(UseEval(fcs[use.face], edge, use.orientation))
            except Exception:
                pass  # no pcurve on that face; treat the edge as unshared there
        self.f0 = [np.array([ue.f0(float(t)) for t in self.ts]) for ue in self.uses]
        for ue, f0 in zip(self.uses, self.f0):
            ue.fc.edge_f.append(f0)

    # -- status -------------------------------------------------------------
    def status_at(self, t: float, delta: float) -> tuple[int, int]:
        a, b = self.uses
        sa, sb = a.sigma(t, delta), b.sigma(t, delta)
        return _combine(sa, sb, a.fc, b.fc)

    def compute_intervals(self, delta: float, extra=()) -> None:
        """Status runs along the edge.  `extra` are parameters where a silhouette
        ends on this edge; we sample there and halfway to the neighbouring
        samples so that short parting stretches between them are not skipped."""
        self.intervals = []
        if self.degenerate or len(self.uses) != 2:
            return
        a, b = self.uses
        if a.fc.cls == VERTICAL and b.fc.cls == VERTICAL:
            return
        if len(extra):
            ts = sorted({*map(float, self.ts), *(min(max(float(e), self.t0), self.t1) for e in extra)})
            mids = []
            for e in extra:
                k = int(np.searchsorted(ts, e))
                for j in (k - 1, k + 1):
                    if 0 <= j < len(ts):
                        mids.append(0.5 * (ts[j] + float(e)))
            self.ts = np.array(sorted({*ts, *mids}))
        sts = [self.status_at(float(t), delta) for t in self.ts]
        if all(s[0] == NONE for s in sts):
            return
        # Runs of equal status, with bisected transitions.
        bounds = [self.t0]
        run_status = [sts[0]]
        for k in range(len(self.ts) - 1):
            if sts[k + 1] != sts[k]:
                lo, hi = float(self.ts[k]), float(self.ts[k + 1])
                tb = self._bisect(lo, hi, sts[k], delta)
                bounds.append(self._exact_crossing(lo, hi, tb))
                run_status.append(sts[k + 1])
        bounds.append(self.t1)
        for i, st in enumerate(run_status):
            if st[0] != NONE and bounds[i + 1] - bounds[i] > 1e-12 * (self.t1 - self.t0):
                self.intervals.append((bounds[i], bounds[i + 1], st[0], st[1]))

    def _exact_crossing(self, lo: float, hi: float, tb: float) -> float:
        """Snap a bisected transition to where n.d of a mixed side is exactly 0.

        Near n.d = 0 the side test looks a little into the face, which shifts
        the transition along the edge when a silhouette meets the edge at a
        shallow angle.  The true point - where the silhouette ends - is the
        root of n.d along the edge."""
        best = tb
        for use in self.uses:
            if use.fc.cls != MIXED:
                continue
            fa, fb = use.f0(lo), use.f0(hi)
            if not (fa == fa and fb == fb):
                continue
            # a sample can sit right on the root (silhouette ends are sampled)
            if abs(fa) < 1e-10 or abs(fb) < 1e-10:
                root = lo if abs(fa) < abs(fb) else hi
                if abs(root - tb) < abs(best - tb) or best == tb:
                    best = root
                continue
            if fa * fb > 0:
                continue
            a, b = lo, hi
            for _ in range(80):
                m = 0.5 * (a + b)
                fm = use.f0(m)
                if fm != fm:
                    break
                if (fm > 0) == (fa > 0):
                    a, fa = m, fm
                else:
                    b = m
                if b - a <= 1e-12 * abs(self.t1 - self.t0):
                    break
            root = 0.5 * (a + b)
            if abs(root - tb) < abs(best - tb) or best == tb:
                best = root
        return best

    def _bisect(self, lo: float, hi: float, st_lo, delta: float) -> float:
        span = abs(self.t1 - self.t0)
        for _ in range(60):
            if hi - lo <= 1e-11 * span:
                break
            mid = 0.5 * (lo + hi)
            if self.status_at(mid, delta) == st_lo:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


def _combine(sa: int, sb: int, fa: FaceCtx, fb: FaceCtx) -> tuple[int, int]:
    if sa * sb == -1:
        return EDGE, -1
    if sa == 0 and sb != 0 and fa.cls == VERTICAL:
        return (VTOP if sb > 0 else VBOT), fa.index
    if sb == 0 and sa != 0 and fb.cls == VERTICAL:
        return (VTOP if sa > 0 else VBOT), fb.index
    return NONE, -1


def edge_curve(edge: TopoDS_Edge):
    """3D curve of an edge with its location applied."""
    return BRep_Tool.Curve_s(edge, 0.0, 0.0)
