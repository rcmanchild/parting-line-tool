"""Parting line across vertical (zero-draft) bands.

A vertical face is swept along d, so steel touching it can only leave along
+d or -d.  For a point X on a band we cast rays along the face (just outside
it) in both directions:

    only +d free  -> cavity      only -d free -> core
    both free     -> the rule decides (top: core, bottom: cavity)
    neither free  -> undercut (reported elsewhere; treated like 'both free')

With nothing in the way this reproduces the plain rule: the line sits on the
band's top (or bottom) edge.  When something attached to the band blocks one
direction - a blade on a hub, a boss on a wall - part of the band must go with
the other half.  The line then:

* follows band edges only where the band's side differs from the neighbour's
  side, and
* drops along a generator (a straight line parallel to d) from every point
  where the parting line arrives at the band (a blade's TE or LE silhouette at
  the root) up/down to the band's far edge.
"""

from __future__ import annotations

import numpy as np
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.GC import GC_MakeSegment
from OCP.GeomAPI import GeomAPI_ProjectPointOnCurve, GeomAPI_ProjectPointOnSurf
from OCP.BRep import BRep_Tool

from . import segments as sg
from .edges import edge_curve
from .occ import pnt

CAVITY_SIDE, CORE_SIDE = 1, -1


class BandSolver:
    def __init__(self, model, fcs, ecs, caster, d, rule: str, lin: float, opts):
        self.model, self.fcs, self.ecs = model, fcs, ecs
        self.caster, self.d, self.rule, self.opts = caster, d, rule, opts
        self.eps_n = 3.0 * lin       # ray offset off the band surface
        self.eps_g = 6.0 * lin       # step into the band from an edge
        self.eps_hit = 0.5 * lin
        self.tol = max(1e-6, 1e-7 * model.diag)
        self.warnings: list[str] = []
        self.region: set[int] = set()

    # -- assignment of a band point ---------------------------------------
    def assign(self, x: np.ndarray, n: np.ndarray) -> int:
        # a band cannot block its own release; ignoring it also stops rays that
        # run alongside a face tilted within the draft tolerance from drifting into it
        up, down = self.caster.cast(x + self.eps_n * n, self.eps_hit, self.region)
        up_free, down_free = not up[0], not down[0]
        if up_free and not down_free:
            return CAVITY_SIDE
        if down_free and not up_free:
            return CORE_SIDE
        return CORE_SIDE if self.rule == "top" else CAVITY_SIDE

    def assign_from_edge(self, use, t: float) -> int:
        """Assignment of the band just inside `use`'s face next to edge param t."""
        p, w, n = use.inward(t)
        if n is None:
            return 0
        a = self.assign(p + self.eps_g * w, n)
        # near steep neighbours a tiny step can land under them; step further once
        b = self.assign(p + 4.0 * self.eps_g * w, n)
        return a if a == b else b

    # -- main ----------------------------------------------------------------
    def solve(self, region: set[int], arrivals: list[np.ndarray]) -> list[sg.Segment]:
        ecs = self.ecs
        self.region = region
        touching = []  # (edge ctx, [use in region...])
        for ec in ecs:
            if ec.degenerate or len(ec.uses) != 2:
                continue
            ins = [u.fc.index in region for u in ec.uses]
            if any(ins):
                touching.append((ec, ins))
        boundary = [ec for ec, ins in touching if ins.count(True) == 1]

        gens: list[sg.Segment] = []
        breaks: dict[int, list[float]] = {}
        g = self.d if self.rule == "top" else -self.d

        def shoot(points) -> int:
            n = 0
            for T in points:
                seg, hit = self._generator(T, g, region, boundary)
                if seg is None:
                    # blocked from the other side (e.g. a through slot at an oblique
                    # pull): the halves can divide along the generator behind T
                    seg, hit = self._generator(T, -g, region, boundary, quiet=True)
                if seg is not None:
                    gens.append(seg)
                    breaks.setdefault(hit[0], []).append(hit[1])
                    n += 1
            return n

        # generators from points where the parting line arrives at the band
        shoot(arrivals)
        out = self._edge_pieces(touching, breaks)
        # second pass: the band's own edge line can also turn into the band
        # (e.g. at the corner of a small vertical face); drop generators from
        # its dangling ends too
        loose = self._dangling(out + gens, arrivals)
        if loose and shoot(loose):
            out = self._edge_pieces(touching, breaks)
        return out + gens

    def _edge_pieces(self, touching, breaks) -> list[sg.Segment]:
        """Band edges are parting line where the two sides end up in different halves."""
        out: list[sg.Segment] = []
        for ec, ins in touching:
            bounds = sorted({ec.t0, ec.t1, *(b for iv in ec.intervals for b in iv[:2]),
                             *breaks.get(ec.index, [])})
            pieces = []
            for a, b in zip(bounds[:-1], bounds[1:]):
                if b - a <= 1e-9 * (ec.t1 - ec.t0):
                    continue
                m = 0.5 * (a + b)
                sides = []
                for use, inside in zip(ec.uses, ins):
                    sides.append(self.assign_from_edge(use, m) if inside else use.sigma(m, self.eps_hit))
                pieces.append((a, b, sides[0] * sides[1] == -1))
            # merge consecutive line pieces
            run = None
            curve = None
            for a, b, is_line in pieces + [(None, None, False)]:
                if is_line and run is not None and abs(run[1] - a) < 1e-12:
                    run = (run[0], b)
                    continue
                if run is not None:
                    curve = curve or edge_curve(ec.edge)
                    out.append(sg.from_curve(sg.VERTICAL, curve, run[0], run[1], self.opts.point_defl,
                                             edge=ec.index + 1))
                    run = None
                if is_line:
                    run = (a, b)
        return out

    def _dangling(self, segs: list[sg.Segment], arrivals: list[np.ndarray]) -> list[np.ndarray]:
        """Ends of band segments that connect to nothing (arrivals count as connected)."""
        tol = self.opts.gap_tol
        pts = [p for s in segs for p in (s.start, s.end)]
        out = []
        for i, p in enumerate(pts):
            if any(np.linalg.norm(p - q) < tol for q in arrivals):
                continue
            if sum(1 for j, q in enumerate(pts) if j != i and np.linalg.norm(p - q) < tol) == 0:
                if not any(np.linalg.norm(p - q) < tol for q in out):
                    out.append(p)
        return out

    # -- generators -----------------------------------------------------------
    def _on_region(self, x: np.ndarray, region: set[int], strict: bool = True):
        """Normal of the region face containing x, or None.  With strict=False
        a point on a face's boundary (where the inside test is unreliable) counts."""
        for fi in region:
            fc = self.fcs[fi]
            proj = GeomAPI_ProjectPointOnSurf(pnt(x), BRep_Tool.Surface_s(fc.fwd))
            if proj.NbPoints() == 0 or proj.LowerDistance() > 10 * self.tol:
                continue
            u, v = proj.LowerDistanceParameters()
            if not strict or fc.inside(u, v):
                return fc.point_normal(u, v)[1]
        return None

    def _generator(self, T: np.ndarray, g: np.ndarray, region: set[int], boundary, quiet: bool = False):
        step = max(4.0 * self.eps_g, 1e-4 * self.model.diag)
        # a face counted vertical within the draft tolerance is not exactly
        # parallel to d: run the generator along d projected onto the face
        n0 = self._on_region(T, region, strict=False)
        if n0 is not None:
            gt = g - float(g @ n0) * n0
            if np.linalg.norm(gt) > 0.5:
                g = gt / np.linalg.norm(gt)
        if self._on_region(T + step * g, region) is None:
            return None, None  # the band does not continue along g from T
        L = 2.0 * self.model.diag + 1.0
        start = T + 1e-3 * step * g
        line = BRepBuilderAPI_MakeEdge(pnt(start), pnt(T + L * g)).Edge()
        best = None
        for ec in boundary:
            dist = BRepExtrema_DistShapeShape(line, ec.edge)
            if not dist.IsDone() or dist.Value() > 10 * self.tol:
                continue
            for k in range(1, dist.NbSolution() + 1):
                q = dist.PointOnShape1(k)
                q = np.array([q.X(), q.Y(), q.Z()])
                lam = float((q - T) @ g)
                if lam > step * 0.5 and (best is None or lam < best[0]):
                    best = (lam, q, ec)
        if best is None:
            if quiet:
                return None, None
            self.warnings.append(f"generator from {np.round(T, 3).tolist()} found no band edge to end on")
            return None, None
        lam, q, ec = best
        # keep it only if the band really changes halves across this line
        mid = T + 0.5 * lam * g
        n = self._on_region(mid, region)
        if n is None:
            return None, None
        side = np.cross(n, g)
        side /= np.linalg.norm(side) or 1.0
        s = 2.0 * self.eps_g
        if self.assign(mid + s * side, n) == self.assign(mid - s * side, n):
            return None, None
        proj = GeomAPI_ProjectPointOnCurve(pnt(q), edge_curve(ec.edge))
        t_hit = proj.LowerDistanceParameter() if proj.NbPoints() else None
        if t_hit is None:
            return None, None
        on_edge = ec.c3.Value(t_hit)
        on_edge = np.array([on_edge.X(), on_edge.Y(), on_edge.Z()])
        # keep the generator exactly along g; its end is within the
        # extrema tolerance of the band edge, well inside the chaining gap
        end = T + float((on_edge - T) @ g) * g
        curve = GC_MakeSegment(pnt(T), pnt(end)).Value()
        seg = sg.from_curve(sg.VERTICAL, curve, curve.FirstParameter(), curve.LastParameter(),
                            self.opts.point_defl, generator=True)
        return seg, (ec.index, t_hit)


def arrival_points(segs: list[sg.Segment], region: set[int], model, ecs, tol: float, cache) -> list[np.ndarray]:
    """Endpoints of edge/silhouette parting segments that sit on the band's boundary."""
    polys = []
    for ec in ecs:
        if ec.degenerate or len(ec.uses) != 2:
            continue
        ins = [u.fc.index in region for u in ec.uses]
        if ins.count(True) != 1:
            continue
        if ec.index not in cache:
            c = edge_curve(ec.edge)
            cache[ec.index] = sg.sample_curve(c, ec.t0, ec.t1, tol / 4) if c is not None else ec.pts
        polys.append(cache[ec.index])
    if not polys:
        return []
    s0 = np.vstack([p[:-1] for p in polys])
    s1 = np.vstack([p[1:] for p in polys])
    seg = s1 - s0
    L2 = np.maximum((seg * seg).sum(axis=1), 1e-30)
    out: list[np.ndarray] = []
    for s in segs:
        if s.kind not in (sg.EDGE, sg.SILHOUETTE):
            continue
        for P in (s.start, s.end):
            tt = np.clip(((P - s0) * seg).sum(axis=1) / L2, 0.0, 1.0)
            dist = np.linalg.norm(s0 + tt[:, None] * seg - P, axis=1).min()
            if dist < tol and not any(np.linalg.norm(P - q) < tol for q in out):
                out.append(P)
    return out


__all__ = ["BandSolver", "arrival_points"]
