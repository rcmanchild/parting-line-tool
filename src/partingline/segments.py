"""Parting-line segments: a trimmed 3D curve plus a dense polyline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from OCP.GCPnts import GCPnts_AbscissaPoint, GCPnts_QuasiUniformDeflection
from OCP.Geom import Geom_Curve
from OCP.GeomAdaptor import GeomAdaptor_Curve
from OCP.GeomAPI import GeomAPI_Interpolate
from OCP.TColgp import TColgp_HArray1OfPnt

from .occ import pnt

# Segment kinds
EDGE = "edge"              # along a model edge whose two sides fall on opposite halves
VERTICAL = "vertical"      # top/bottom boundary of a vertical (zero-draft) band
SILHOUETTE = "silhouette"  # smooth silhouette inside a face (n.d = 0)
SECTION = "section"        # mid-height section through a vertical band
BRIDGE = "bridge"          # straight closure of a small gap the analysis left (reported)


@dataclass
class Segment:
    kind: str
    curve: Geom_Curve
    t0: float
    t1: float
    points: np.ndarray            # (N, 3), ordered from t0 to t1
    length: float
    source: dict = field(default_factory=dict)  # e.g. {"edge": 12} or {"face": 3}
    uv: np.ndarray | None = None  # silhouettes: (N, 2) UV on source face

    @property
    def start(self) -> np.ndarray:
        return self.points[0]

    @property
    def end(self) -> np.ndarray:
        return self.points[-1]


def sample_curve(curve: Geom_Curve, t0: float, t1: float, defl: float) -> np.ndarray:
    ad = GeomAdaptor_Curve(curve, t0, t1)
    algo = GCPnts_QuasiUniformDeflection(ad, defl, t0, t1)
    if algo.IsDone() and algo.NbPoints() >= 2:
        pts = [algo.Value(i) for i in range(1, algo.NbPoints() + 1)]
    else:
        pts = [curve.Value(t) for t in np.linspace(t0, t1, 33)]
    return np.array([[p.X(), p.Y(), p.Z()] for p in pts])


def curve_length(curve: Geom_Curve, t0: float, t1: float) -> float:
    ad = GeomAdaptor_Curve(curve, t0, t1)
    try:
        return float(GCPnts_AbscissaPoint.Length_s(ad, t0, t1))
    except Exception:
        pts = sample_curve(curve, t0, t1, 1e-3)
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def from_curve(kind: str, curve: Geom_Curve, t0: float, t1: float, defl: float, **source) -> Segment:
    pts = sample_curve(curve, t0, t1, defl)
    return Segment(kind, curve, t0, t1, pts, curve_length(curve, t0, t1), dict(source))


def interpolate(points: np.ndarray, closed: bool = False) -> Geom_Curve | None:
    """Cubic interpolation through 3D points (duplicates removed)."""
    pts = [points[0]]
    for p in points[1:]:
        if np.linalg.norm(p - pts[-1]) > 1e-7:
            pts.append(p)
    if closed and len(pts) > 2 and np.linalg.norm(pts[-1] - pts[0]) < 1e-7:
        pts.pop()
    if len(pts) < 2:
        return None
    arr = TColgp_HArray1OfPnt(1, len(pts))
    for i, p in enumerate(pts, start=1):
        arr.SetValue(i, pnt(p))
    algo = GeomAPI_Interpolate(arr, bool(closed and len(pts) > 2), 1e-9)
    algo.Perform()
    if not algo.IsDone():
        return None
    return algo.Curve()
