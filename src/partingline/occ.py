"""Thin helpers around OCP so the rest of the code reads cleanly."""

from __future__ import annotations

import numpy as np
from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Builder
from OCP.BRepBndLib import BRepBndLib
from OCP.TopExp import TopExp
from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Shape
from OCP.TopTools import TopTools_IndexedMapOfShape
from OCP.gp import gp_Pnt

_CASTS = {
    "face": TopoDS.Face_s,
    "edge": TopoDS.Edge_s,
    "vertex": TopoDS.Vertex_s,
    "solid": TopoDS.Solid_s,
    "shell": TopoDS.Shell_s,
    "wire": TopoDS.Wire_s,
}


def indexed_map(shape: TopoDS_Shape, kind) -> TopTools_IndexedMapOfShape:
    m = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, kind, m)
    return m


def map_list(m: TopTools_IndexedMapOfShape, cast: str) -> list:
    fn = _CASTS[cast]
    return [fn(m.FindKey(i)) for i in range(1, m.Extent() + 1)]


def make_compound(shapes) -> TopoDS_Compound:
    comp = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(comp)
    for s in shapes:
        builder.Add(comp, s)
    return comp


def bbox(shape: TopoDS_Shape) -> tuple[np.ndarray, np.ndarray]:
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box, False)
    if box.IsVoid():
        return np.zeros(3), np.zeros(3)
    x0, y0, z0, x1, y1, z1 = box.Get()
    return np.array([x0, y0, z0]), np.array([x1, y1, z1])


def pnt(p) -> gp_Pnt:
    return gp_Pnt(float(p[0]), float(p[1]), float(p[2]))


def xyz(p: gp_Pnt) -> np.ndarray:
    return np.array([p.X(), p.Y(), p.Z()])


def orthonormal_basis(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (e1, e2) spanning the plane perpendicular to unit vector d.

    e1 is the projection of global X (or Y if d is close to X), so DXF output
    keeps a predictable orientation.
    """
    ref = np.array([1.0, 0.0, 0.0])
    if abs(d @ ref) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = ref - (ref @ d) * d
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    return e1, e2
