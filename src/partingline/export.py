"""Writers: STEP / IGES wires, projected DXF, CSV points, JSON report."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from OCP.IFSelect import IFSelect_RetDone
from OCP.IGESControl import IGESControl_Writer
from OCP.Interface import Interface_Static
from OCP.Message import Message, Message_Gravity
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

from .occ import make_compound, orthonormal_basis
from .pull import describe

LINE_FORMATS = ("step", "iges", "dxf", "csv")
OUTPUT_FORMATS = LINE_FORMATS + ("split", "surface")  # split: cavity/core shells; surface: parting surface


def _quiet_occ() -> None:
    """Stop the STEP/IGES writers from printing transfer statistics."""
    for printer in Message.DefaultMessenger_s().Printers():
        printer.SetTraceLevel(Message_Gravity.Message_Fail)


def write_step(shapes, path: Path) -> None:
    _quiet_occ()
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    w = STEPControl_Writer()
    w.Transfer(make_compound(shapes), STEPControl_AsIs)
    if w.Write(str(path)) != IFSelect_RetDone:
        raise RuntimeError(f"STEP write failed: {path}")


def write_iges(shapes, path: Path) -> None:
    _quiet_occ()
    w = IGESControl_Writer("MM", 0)
    w.AddShape(make_compound(shapes))
    w.ComputeModel()
    if not w.Write(str(path)):
        raise RuntimeError(f"IGES write failed: {path}")


def write_dxf(loops, d: np.ndarray, path: Path) -> None:
    """Parting line projected onto the plane normal to the pull direction.

    2D axes: e1 = global X projected onto that plane (Y if pulling along X),
    e2 = d x e1.  R12 ASCII, one POLYLINE per loop, units mm.
    """
    e1, e2 = orthonormal_basis(d)
    out = ["0", "SECTION", "2", "HEADER",
           "9", "$ACADVER", "1", "AC1009",
           "9", "$INSUNITS", "70", "4",
           "9", "$MEASUREMENT", "70", "1",
           "0", "ENDSEC",
           "0", "SECTION", "2", "ENTITIES"]
    for lp in loops:
        pts = lp.points
        if lp.closed and len(pts) > 2 and np.linalg.norm(pts[0] - pts[-1]) < 1e-9:
            pts = pts[:-1]
        out += ["0", "POLYLINE", "8", "PARTING_LINE", "66", "1",
                "10", "0.0", "20", "0.0", "30", "0.0", "70", "1" if lp.closed else "0"]
        for p in pts:
            out += ["0", "VERTEX", "8", "PARTING_LINE",
                    "10", f"{p @ e1:.6f}", "20", f"{p @ e2:.6f}", "30", "0.0"]
        out += ["0", "SEQEND", "8", "PARTING_LINE"]
    out += ["0", "ENDSEC", "0", "EOF"]
    path.write_text("\n".join(out) + "\n", encoding="ascii")


def write_csv(loops, path: Path) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["loop", "closed", "index", "x", "y", "z"])
        for li, lp in enumerate(loops, start=1):
            for k, p in enumerate(lp.points):
                w.writerow([li, int(lp.closed), k, f"{p[0]:.6f}", f"{p[1]:.6f}", f"{p[2]:.6f}"])


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not math.isfinite(f) else round(f, 6)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


def report_dict(model, result, outputs: dict | None = None, auto_info: dict | None = None) -> dict:
    ii, hi = model.import_info, model.heal_info
    opts = result.options
    e1, e2 = orthonormal_basis(result.pull)
    rep = {
        "input": {
            "file": ii.path,
            "file_length_units": ii.file_units,
            "converted_to_mm_from": ii.converted_from,
            "scale_to_mm": ii.scale_to_mm,
            "units": "mm",
            "faces": model.n_faces,
            "edges": len(model.edges),
            "bbox_min": model.bmin,
            "bbox_max": model.bmax,
            "bbox_diagonal": model.diag,
        },
        "healing": {
            "shapefix": hi.shapefix,
            "sewn": hi.sewn,
            "faces_split_at_creases": hi.faces_split_at_creases,
            "solids": hi.solids,
            "shells": hi.shells,
            "free_edges_before": hi.free_edges_before,
            "free_edges_after": hi.free_edges_after,
            "reversed_solids_fixed": hi.reversed_solids_fixed,
            "max_tolerance": hi.max_tolerance,
            "edges_tol_over_gap": hi.edges_tol_over_gap,
        },
        "pull": {"vector": result.pull, "label": describe(result.pull),
                 "dxf_axes": {"e1": e1, "e2": e2}},
        "options": {
            "draft_tol_deg": opts.draft_tol_deg,
            "vertical": opts.vertical,
            "gap_tol": opts.gap_tol,
            "chord_tol": opts.chord_tol,
            "undercut_min_frac": opts.undercut_min_frac,
            "low_draft_deg": opts.low_draft_deg,
        },
        "summary": result.stats,
        "loops": [
            {"id": i, "closed": lp.closed, "length": lp.length, "segments": len(lp.items),
             "length_by_kind": lp.kinds, "start": lp.points[0] if len(lp.points) else None}
            for i, lp in enumerate(result.loops, start=1)
        ],
        "faces": [vars(f) for f in result.faces],
        "auto": auto_info,
        "split": _split_dict(result.split),
        "surface": _surface_dict(result.surface),
        "warnings": result.warnings,
        "timings_s": result.timings,
        "outputs": outputs or {},
    }
    return _clean(rep)


def _split_dict(sp):
    if sp is None:
        return None
    return {
        "faces": sp.model.n_faces,
        "cavity_faces": len(sp.cavity_faces),
        "core_faces": len(sp.core_faces),
        "cavity_area": sp.areas["cavity"],
        "core_area": sp.areas["core"],
        "total_area": sp.areas["total"],
        "groups": sp.components,
        "conflicts": sp.conflicts,
        "unsplit_faces": sp.unsplit,
        "open_boundary_length": sp.boundary_length,
        "warnings": sp.warnings,
    }


def _surface_dict(sf):
    if sf is None:
        return None
    return sf.report()


def write_report(model, result, path: Path, outputs: dict | None = None, auto_info: dict | None = None) -> dict:
    rep = report_dict(model, result, outputs, auto_info)
    path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return rep


def write_split(split, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    out = {}
    for which in ("cavity", "core"):
        p = outdir / f"{which}_faces.step"
        write_step([split.shells(which)], p)
        out[which] = str(p)
    return out


def write_line_outputs(result, outdir: Path, formats=LINE_FORMATS, stem: str = "parting_line") -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    written = {}
    if not result.loops:
        return written
    for fmt in formats:
        if fmt not in LINE_FORMATS:
            continue
        p = outdir / f"{stem}.{fmt}"
        if fmt == "step":
            write_step(result.wires, p)
        elif fmt == "iges":
            write_iges(result.wires, p)
        elif fmt == "dxf":
            write_dxf(result.loops, result.pull, p)
        elif fmt == "csv":
            write_csv(result.loops, p)
        else:
            raise ValueError(f"unknown format {fmt!r}")
        written[fmt] = str(p)
    return written


def summary_text(model, result) -> str:
    s = result.stats
    ii = model.import_info
    lines = []
    lines.append(f"Input   : {ii.path}  ({model.n_faces} faces, bbox diag {model.diag:.2f} mm)")
    units = ",".join(ii.file_units) or "?"
    if ii.converted_from:
        lines.append(f"Units   : file {units} -> converted to mm (x{ii.scale_to_mm:g})  ** check this **")
    else:
        lines.append(f"Units   : file {units}, analysis in mm")
    lines.append(f"Pull    : {describe(result.pull)}   draft tol {result.options.draft_tol_deg} deg, "
                 f"vertical rule '{result.options.vertical}'")
    c, a = s["face_counts"], s["face_area"]
    lines.append("Faces   : " + "  ".join(f"{k} {c[k]} ({a[k]:.1f} mm2)" for k in c))
    md = s["min_draft_deg"]

    def fmt(v):
        return "-" if v is None else f"{v:.2f} deg"

    lines.append(f"Draft   : min cavity {fmt(md['cavity'])}, min core {fmt(md['core'])}, "
                 f"area under {result.options.low_draft_deg} deg draft {s['low_draft_area']:.1f} mm2")
    vf = s["vertical_faces"]
    lines.append(f"Vertical: {len(vf)} face(s), {s['vertical_area']:.1f} mm2 need draft"
                 + (f"  faces {_short(vf)}" if vf else ""))
    uf = s["undercut_faces"]
    lines.append(f"Undercut: {len(uf)} face(s), {s['undercut_area']:.1f} mm2"
                 + (f"  faces {_short(uf)}" if uf else ""))
    lp = s["loops"]
    lines.append(f"Line    : {lp['count']} loop(s), {lp['closed']} closed, {lp['open']} open, "
                 f"total {lp['total_length']:.3f} mm")
    for i, loop in enumerate(result.loops, start=1):
        kinds = ", ".join(f"{k} {v:.1f}" for k, v in sorted(loop.kinds.items()))
        lines.append(f"          #{i}: {'closed' if loop.closed else 'OPEN  '} {loop.length:10.3f} mm"
                     f"  [{kinds}]")
    if lp["junctions"]:
        lines.append(f"          {lp['junctions']} junction(s) with >2 segments")
    sp = result.split
    if sp is not None:
        lines.append(f"Split   : {sp.model.n_faces} faces -> cavity {len(sp.cavity_faces)} ({sp.areas['cavity']:.1f} mm2), "
                     f"core {len(sp.core_faces)} ({sp.areas['core']:.1f} mm2), {len(sp.conflicts)} conflict(s)")
    sf = result.surface
    if sf is not None:
        lines.append(sf.summary_line())
    for w in result.warnings + (sp.warnings if sp is not None else []) + (sf.warnings if sf is not None else []):
        lines.append(f"WARNING : {w}")
    lines.append(f"Time    : {result.timings.get('total', 0):.2f} s")
    return "\n".join(lines)


def _short(nums, n=12) -> str:
    txt = ",".join(str(x) for x in nums[:n])
    return txt + (",..." if len(nums) > n else "")
