"""Loading, healing and indexing of the input B-rep.

Units: OpenCascade's STEP reader converts lengths to the session unit, which we
pin to millimetres.  We never rescale on our own; when the file declares a unit
other than mm we say so loudly in the report and the printed summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing
from OCP.BRepGProp import BRepGProp
from OCP.BRepLib import BRepLib
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.ShapeAnalysis import ShapeAnalysis_ShapeTolerance
from OCP.ShapeFix import ShapeFix_Shape
from OCP.ShapeUpgrade import ShapeUpgrade_ShapeDivideContinuity
from OCP.GeomAbs import GeomAbs_C1
from OCP.STEPControl import STEPControl_Reader
from OCP.TColStd import TColStd_SequenceOfAsciiString
from OCP.TopAbs import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_FORWARD,
    TopAbs_INTERNAL,
    TopAbs_EXTERNAL,
    TopAbs_SHAPE,
    TopAbs_SHELL,
    TopAbs_SOLID,
)
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Shape

from .occ import bbox, indexed_map, make_compound, map_list

# Length units the STEP reader may report, in mm.
UNIT_TO_MM = {
    "MM": 1.0,
    "MILLIMETRE": 1.0,
    "MILLIMETER": 1.0,
    "CM": 10.0,
    "CENTIMETRE": 10.0,
    "M": 1000.0,
    "METRE": 1000.0,
    "METER": 1000.0,
    "UM": 1e-3,
    "MICROMETRE": 1e-3,
    "INCH": 25.4,
    "IN": 25.4,
    "FT": 304.8,
    "FOOT": 304.8,
}


@dataclass
class ImportInfo:
    path: str = ""
    file_units: list[str] = field(default_factory=list)
    converted_from: str | None = None  # set when the reader scaled the file to mm
    scale_to_mm: float = 1.0


@dataclass
class HealInfo:
    shapefix: bool = False
    sewn: bool = False
    faces_split_at_creases: int = 0
    free_edges_before: int = 0
    free_edges_after: int = 0
    solids: int = 0
    shells: int = 0
    reversed_solids_fixed: int = 0
    max_tolerance: float = 0.0
    edges_tol_over_gap: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class EdgeUse:
    """One appearance of an edge on a face (a seam edge has two on one face)."""

    face: int  # 0-based face index
    orientation: object  # TopAbs_FORWARD / TopAbs_REVERSED, relative to the FORWARD face


class Model:
    """A healed shape plus stable 0-based face/edge indexing.

    Face numbers shown to users are 1-based (index + 1).
    """

    def __init__(self, shape: TopoDS_Shape, import_info: ImportInfo | None = None,
                 heal_info: HealInfo | None = None):
        self.shape = shape
        self.import_info = import_info or ImportInfo()
        self.heal_info = heal_info or HealInfo()
        self.fmap = indexed_map(shape, TopAbs_FACE)
        self.emap = indexed_map(shape, TopAbs_EDGE)
        self.faces = map_list(self.fmap, "face")
        self.edges = map_list(self.emap, "edge")
        self.bmin, self.bmax = bbox(shape)
        self.diag = float(np.linalg.norm(self.bmax - self.bmin))
        self.edge_uses: list[list[EdgeUse]] = [[] for _ in self.edges]
        self.face_edges: list[list[int]] = [[] for _ in self.faces]
        for fi, face in enumerate(self.faces):
            fwd = face.Oriented(TopAbs_FORWARD)
            ex = TopExp_Explorer(fwd, TopAbs_EDGE)
            while ex.More():
                e = TopoDS.Edge_s(ex.Current())
                ex.Next()
                ori = e.Orientation()
                if ori in (TopAbs_INTERNAL, TopAbs_EXTERNAL):
                    continue
                ei = self.emap.FindIndex(e) - 1
                if ei < 0:
                    continue
                self.edge_uses[ei].append(EdgeUse(fi, ori))
                if ei not in self.face_edges[fi]:
                    self.face_edges[fi].append(ei)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    def free_edge_count(self) -> int:
        return _free_edge_count(self.shape)


def _free_edge_count(shape: TopoDS_Shape) -> int:
    fmap = indexed_map(shape, TopAbs_FACE)
    emap = indexed_map(shape, TopAbs_EDGE)
    counts = np.zeros(emap.Extent(), dtype=int)
    for face in map_list(fmap, "face"):
        ex = TopExp_Explorer(face, TopAbs_EDGE)
        while ex.More():
            e = TopoDS.Edge_s(ex.Current())
            ex.Next()
            if e.Orientation() in (TopAbs_INTERNAL, TopAbs_EXTERNAL):
                continue
            counts[emap.FindIndex(e) - 1] += 1
    free = 0
    for i, e in enumerate(map_list(emap, "edge")):
        if counts[i] == 1 and not BRep_Tool.Degenerated_s(e):
            free += 1
    return free


def read_step(path: str | Path) -> tuple[TopoDS_Shape, ImportInfo]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    Interface_Static.SetCVal_s("xstep.cascade.unit", "MM")
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP read failed for {path} (status {status})")
    lengths, angles, solid_angles = (TColStd_SequenceOfAsciiString() for _ in range(3))
    reader.FileUnits(lengths, angles, solid_angles)
    units = sorted({lengths.Value(i).ToCString().upper() for i in range(1, lengths.Length() + 1)})
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape is None or shape.IsNull():
        raise RuntimeError(f"STEP file {path} contains no transferable shape")
    info = ImportInfo(path=str(path), file_units=units)
    non_mm = [u for u in units if UNIT_TO_MM.get(u, None) != 1.0]
    if non_mm:
        info.converted_from = ",".join(non_mm)
        info.scale_to_mm = UNIT_TO_MM.get(non_mm[0], float("nan"))
    return shape, info


def heal(shape: TopoDS_Shape, sew_tol: float = 0.01, gap_tol: float = 0.01) -> tuple[TopoDS_Shape, HealInfo]:
    info = HealInfo()
    info.free_edges_before = _free_edge_count(shape)

    sf = ShapeFix_Shape(shape)
    sf.SetPrecision(1e-7)
    sf.SetMaxTolerance(max(sew_tol, 1e-3))
    sf.Perform()
    shape = sf.Shape()
    info.shapefix = True

    free = _free_edge_count(shape)
    if free:
        sew = BRepBuilderAPI_Sewing(sew_tol)
        sew.Add(shape)
        sew.Perform()
        sewn = sew.SewedShape()
        if not sewn.IsNull():
            free_sewn = _free_edge_count(sewn)
            if free_sewn < free:
                shape, free, info.sewn = sewn, free_sewn, True
                # Closed shells that are not yet solids become solids.
                if indexed_map(shape, TopAbs_SOLID).Extent() == 0 and free == 0:
                    solids = []
                    for sh in map_list(indexed_map(shape, TopAbs_SHELL), "shell"):
                        ms = BRepBuilderAPI_MakeSolid(sh)
                        if ms.IsDone():
                            solids.append(ms.Solid())
                    if solids:
                        shape = solids[0] if len(solids) == 1 else make_compound(solids)
                sf = ShapeFix_Shape(shape)
                sf.Perform()
                shape = sf.Shape()
    info.free_edges_after = _free_edge_count(shape)

    # Outward normals matter: a solid with negative volume is inside-out.
    solids = map_list(indexed_map(shape, TopAbs_SOLID), "solid")
    fixed = []
    for s in solids:
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(s, props)
        if props.Mass() < 0:
            info.reversed_solids_fixed += 1
            fixed.append(s.Reversed())
        else:
            fixed.append(s)
    if info.reversed_solids_fixed:
        shape = fixed[0] if len(fixed) == 1 else make_compound(fixed)

    # Creases (C0 knots) inside B-spline faces become real edges, so the
    # sharp-edge logic sees them; a crease inside a face would otherwise hide
    # a parting edge.
    n_before = indexed_map(shape, TopAbs_FACE).Extent()
    div = ShapeUpgrade_ShapeDivideContinuity(shape)
    div.SetBoundaryCriterion(GeomAbs_C1)
    div.SetPCurveCriterion(GeomAbs_C1)
    div.SetSurfaceCriterion(GeomAbs_C1)
    if div.Perform() and not div.Result().IsNull():
        divided = div.Result()
        n_after = indexed_map(divided, TopAbs_FACE).Extent()
        if n_after > n_before:
            shape = divided
            info.faces_split_at_creases = n_after - n_before

    BRepLib.BuildCurves3d_s(shape)

    info.solids = len(solids)
    info.shells = indexed_map(shape, TopAbs_SHELL).Extent()
    st = ShapeAnalysis_ShapeTolerance()
    info.max_tolerance = float(st.Tolerance(shape, 1, TopAbs_SHAPE))
    n_big = 0
    for e in map_list(indexed_map(shape, TopAbs_EDGE), "edge"):
        if BRep_Tool.Tolerance_s(e) > gap_tol:
            n_big += 1
    info.edges_tol_over_gap = n_big

    if info.solids == 0:
        info.warnings.append("no solid found; analysing open faces/shells (normals may be inconsistent)")
    if info.solids > 1:
        info.warnings.append(f"{info.solids} solids; overlapping solids give internal faces and a messy parting line")
    if info.free_edges_after:
        info.warnings.append(f"{info.free_edges_after} free (unshared) edges remain after healing; the model is not watertight")
    if info.reversed_solids_fixed:
        info.warnings.append(f"{info.reversed_solids_fixed} inside-out solid(s) were reversed")
    if n_big:
        info.warnings.append(f"{n_big} edges have tolerance above the {gap_tol} mm gap tolerance (max {info.max_tolerance:.3g} mm)")
    return shape, info


def load_model(path: str | Path, do_heal: bool = True, gap_tol: float = 0.01) -> Model:
    shape, iinfo = read_step(path)
    hinfo = HealInfo()
    if do_heal:
        shape, hinfo = heal(shape, gap_tol=gap_tol)
    model = Model(shape, iinfo, hinfo)
    if iinfo.converted_from:
        model.heal_info.warnings.insert(
            0,
            f"file length unit is {iinfo.converted_from}, not mm: the STEP reader converted it to mm "
            f"(x{iinfo.scale_to_mm:g}). All output is in mm.",
        )
    if model.diag < 1.0 or model.diag > 1e4:
        model.heal_info.warnings.append(
            f"bounding-box diagonal is {model.diag:.4g} mm; check that the file units are right"
        )
    return model


def shape_model(shape: TopoDS_Shape, do_heal: bool = True) -> Model:
    """Build a Model from an in-memory shape (tests, scripting)."""
    hinfo = HealInfo()
    if do_heal:
        shape, hinfo = heal(shape)
    return Model(shape, ImportInfo(path="<memory>"), hinfo)


__all__ = ["Model", "EdgeUse", "ImportInfo", "HealInfo", "read_step", "heal", "load_model", "shape_model"]
