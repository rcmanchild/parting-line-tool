"""Per-face evaluation: normals, n.d sampling and classification."""

from __future__ import annotations

import math

import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepGProp import BRepGProp
from OCP.BRepTools import BRepTools
from OCP.BRepTopAdaptor import BRepTopAdaptor_FClass2d
from OCP.GeomAbs import (
    GeomAbs_BezierSurface,
    GeomAbs_BSplineSurface,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_OffsetSurface,
    GeomAbs_OtherSurface,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_SurfaceOfExtrusion,
    GeomAbs_SurfaceOfRevolution,
    GeomAbs_Torus,
)
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_FORWARD, TopAbs_IN, TopAbs_REVERSED
from OCP.TopoDS import TopoDS, TopoDS_Face
from OCP.gp import gp_Pnt, gp_Pnt2d, gp_Vec

CAVITY, CORE, VERTICAL, MIXED = "cavity", "core", "vertical", "mixed"

SURFACE_NAMES = {
    GeomAbs_Plane: "plane",
    GeomAbs_Cylinder: "cylinder",
    GeomAbs_Cone: "cone",
    GeomAbs_Sphere: "sphere",
    GeomAbs_Torus: "torus",
    GeomAbs_BezierSurface: "bezier",
    GeomAbs_BSplineSurface: "bspline",
    GeomAbs_SurfaceOfRevolution: "revolution",
    GeomAbs_SurfaceOfExtrusion: "extrusion",
    GeomAbs_OffsetSurface: "offset",
    GeomAbs_OtherSurface: "other",
}

TWO_PI = 2.0 * math.pi


class FaceCtx:
    """Evaluator for one face.  All UV work happens on the FORWARD-oriented face;
    `sign` flips the surface normal for faces that are REVERSED in the shape so
    that `normal()` is always the outward material normal."""

    def __init__(self, index: int, face: TopoDS_Face):
        self.index = index
        self.face = face
        self.fwd = TopoDS.Face_s(face.Oriented(TopAbs_FORWARD))
        self.sign = -1.0 if face.Orientation() == TopAbs_REVERSED else 1.0
        self.ad = BRepAdaptor_Surface(self.fwd, True)
        self.stype = self.ad.GetType()
        self.type_name = SURFACE_NAMES.get(self.stype, "other")
        self.u0, self.u1, self.v0, self.v1 = BRepTools.UVBounds_s(self.fwd)
        self.du = self.u1 - self.u0
        self.dv = self.v1 - self.v0
        self.tol2d = 1e-9 * max(self.du, self.dv, 1e-12)
        self.cls2d = BRepTopAdaptor_FClass2d(self.fwd, self.tol2d)
        self._P = gp_Pnt()
        self._Du = gp_Vec()
        self._Dv = gp_Vec()
        self._d = gp_Vec(0.0, 0.0, 1.0)
        self._area: float | None = None
        # Filled by the analysis.
        self.samples_uv = np.zeros((0, 2))
        self.samples_xyz = np.zeros((0, 3))
        self.samples_n = np.zeros((0, 3))
        self.samples_f = np.zeros(0)
        self.edge_f: list[np.ndarray] = []
        self.cls: str | None = None
        self.fmin = math.nan
        self.fmax = math.nan

    # -- evaluation ---------------------------------------------------------
    def set_pull(self, d: np.ndarray) -> None:
        self._d = gp_Vec(float(d[0]), float(d[1]), float(d[2]))

    def f(self, u: float, v: float) -> float:
        """n . d at (u, v); NaN where the normal is undefined (poles)."""
        self.ad.D1(u, v, self._P, self._Du, self._Dv)
        n = self._Du.Crossed(self._Dv)
        m = n.Magnitude()
        if m < 1e-300 or m < 1e-12 * self._Du.Magnitude() * self._Dv.Magnitude():
            return math.nan
        return self.sign * n.Dot(self._d) / m

    def f_safe(self, u: float, v: float) -> float:
        """Like f(), but nudges toward the UV centre at singular points."""
        val = self.f(u, v)
        if val == val:
            return val
        uc, vc = 0.5 * (self.u0 + self.u1), 0.5 * (self.v0 + self.v1)
        for k in (1e-7, 1e-5, 1e-3):
            val = self.f(u + k * (uc - u), v + k * (vc - v))
            if val == val:
                return val
        return math.nan

    def point(self, u: float, v: float) -> np.ndarray:
        p = self.ad.Value(u, v)
        return np.array([p.X(), p.Y(), p.Z()])

    def point_normal(self, u: float, v: float):
        self.ad.D1(u, v, self._P, self._Du, self._Dv)
        n = self._Du.Crossed(self._Dv)
        m = n.Magnitude()
        p = np.array([self._P.X(), self._P.Y(), self._P.Z()])
        if m < 1e-300 or m < 1e-12 * self._Du.Magnitude() * self._Dv.Magnitude():
            return p, None
        s = self.sign / m
        return p, np.array([n.X() * s, n.Y() * s, n.Z() * s])

    def uv_dir_length(self, u: float, v: float, wu: float, wv: float) -> float:
        """3D length of the UV direction (wu, wv) at (u, v)."""
        self.ad.D1(u, v, self._P, self._Du, self._Dv)
        x = self._Du.X() * wu + self._Dv.X() * wv
        y = self._Du.Y() * wu + self._Dv.Y() * wv
        z = self._Du.Z() * wu + self._Dv.Z() * wv
        return math.sqrt(x * x + y * y + z * z)

    def inside(self, u: float, v: float) -> bool:
        return self.cls2d.Perform(gp_Pnt2d(u, v), True) == TopAbs_IN

    @property
    def area(self) -> float:
        if self._area is None:
            # adaptive integration: the default Gauss rule can be several % off
            # on trimmed faces with holes
            props = GProp_GProps()
            BRepGProp.SurfaceProperties_s(self.face, props, 1e-4)
            self._area = abs(props.Mass())
        return self._area

    # -- sampling -----------------------------------------------------------
    def sample_count(self) -> int:
        if self.stype == GeomAbs_Plane:
            return 8
        if self.stype in (GeomAbs_BSplineSurface, GeomAbs_BezierSurface):
            spans = max(self._spans("u"), self._spans("v"))
            return int(min(32, max(12, 2 * spans)))
        return 12

    def sample_grid(self, n: int | None = None) -> None:
        """Cell-centred UV grid, keeping only points inside the trimmed face."""
        n = n or self.sample_count()
        us = self.u0 + (np.arange(n) + 0.5) / n * self.du
        vs = self.v0 + (np.arange(n) + 0.5) / n * self.dv
        uv, pts, nrm = [], [], []
        plane_n = None
        if self.stype == GeomAbs_Plane:
            _, plane_n = self.point_normal(0.5 * (self.u0 + self.u1), 0.5 * (self.v0 + self.v1))
        for u in us:
            for v in vs:
                if not self.inside(u, v):
                    continue
                if plane_n is not None:
                    p, nn = self.point(u, v), plane_n
                else:
                    p, nn = self.point_normal(u, v)
                    if nn is None:
                        continue
                uv.append((u, v))
                pts.append(p)
                nrm.append(nn)
        self.samples_uv = np.array(uv, dtype=float).reshape(-1, 2)
        self.samples_xyz = np.array(pts, dtype=float).reshape(-1, 3)
        self.samples_n = np.array(nrm, dtype=float).reshape(-1, 3)

    def update_f(self, d: np.ndarray) -> None:
        self.samples_f = self.samples_n @ d if len(self.samples_n) else np.zeros(0)

    def classify(self, t: float) -> str:
        vals = [self.samples_f] + [a[np.isfinite(a)] for a in self.edge_f]
        allf = np.concatenate(vals) if vals else np.zeros(0)
        if allf.size == 0:
            self.fmin = self.fmax = math.nan
            self.cls = VERTICAL
            return self.cls
        self.fmin = float(allf.min())
        self.fmax = float(allf.max())
        if self.fmax <= t and self.fmin >= -t:
            self.cls = VERTICAL
        elif self.fmin >= -t:
            self.cls = CAVITY
        elif self.fmax <= t:
            self.cls = CORE
        else:
            self.cls = MIXED
        return self.cls

    # -- silhouette grid sizing --------------------------------------------
    def _spans(self, which: str) -> int:
        try:
            bs = self.ad.BSpline() if self.stype == GeomAbs_BSplineSurface else None
        except Exception:
            bs = None
        if bs is None:
            return 1
        if which == "u":
            knots = [bs.UKnot(i) for i in range(1, bs.NbUKnots() + 1)]
            lo, hi, deg = self.u0, self.u1, bs.UDegree()
        else:
            knots = [bs.VKnot(i) for i in range(1, bs.NbVKnots() + 1)]
            lo, hi, deg = self.v0, self.v1, bs.VDegree()
        inner = sum(1 for k in knots if lo < k < hi)
        return (inner + 1) * max(1, (deg + 1) // 3)

    def silhouette_grid(self) -> tuple[int, int]:
        """Marching-squares grid size (nodes per direction) for this face."""

        def ang(span: float, per_turn: int) -> int:
            return max(9, int(math.ceil(per_turn * span / TWO_PI)) + 1)

        t = self.stype
        if t in (GeomAbs_Cylinder, GeomAbs_Cone):
            return ang(self.du, 96), 3
        if t == GeomAbs_Sphere:
            return ang(self.du, 96), max(9, int(math.ceil(96 * self.dv / TWO_PI)) + 1)
        if t == GeomAbs_Torus:
            return ang(self.du, 96), ang(self.dv, 96)
        if t == GeomAbs_SurfaceOfRevolution:
            return ang(self.du, 96), 41
        if t == GeomAbs_SurfaceOfExtrusion:
            return 65, 3
        if t in (GeomAbs_BSplineSurface, GeomAbs_BezierSurface):
            nu = int(min(257, max(25, 4 * self._spans("u") + 9)))
            nv = int(min(257, max(25, 4 * self._spans("v") + 9)))
            return nu, nv
        return 49, 49


def axis_of(fc: FaceCtx):
    """Axis direction of a cylindrical/conical/revolution face, else None."""
    if fc.stype == GeomAbs_Cylinder:
        a = fc.ad.Cylinder().Axis()
    elif fc.stype == GeomAbs_Cone:
        a = fc.ad.Cone().Axis()
    elif fc.stype == GeomAbs_SurfaceOfRevolution:
        a = fc.ad.AxeOfRevolution()
    else:
        return None
    d, p = a.Direction(), a.Location()
    return np.array([d.X(), d.Y(), d.Z()]), np.array([p.X(), p.Y(), p.Z()])


__all__ = ["FaceCtx", "CAVITY", "CORE", "VERTICAL", "MIXED", "SURFACE_NAMES", "axis_of"]
