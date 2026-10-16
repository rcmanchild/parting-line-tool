"""One analysis run with its optional extras and outputs; shared by the CLI and the viewer."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .analysis import Options, Result, analyze
from .export import write_line_outputs, write_report, write_split, write_step
from .split import split_and_assign
from .surface import parting_surface


@dataclass
class SurfaceSpec:
    mode: str = "sweep"
    distance: float = 20.0
    height: float | None = None


def run(model, pull: np.ndarray, opts: Options, formats=(), outdir: Path | None = None,
        surface: SurfaceSpec | None = None, auto_info: dict | None = None,
        extra_time: float = 0.0) -> tuple[Result, dict]:
    """Analyse, then split / build the surface when `formats` asks for them,
    and write everything (plus report.json) to `outdir` if given."""
    surface = surface or SurfaceSpec()
    result = analyze(model, pull, opts)
    t = result.timings
    if extra_time:
        t["total"] = round(t["total"] + extra_time, 3)

    def timed(name, fn):
        t0 = time.perf_counter()
        out = fn()
        t[name] = round(time.perf_counter() - t0, 3)
        t["total"] = round(t["total"] + t[name], 3)
        return out

    if "split" in formats and result.loops:
        result.split = timed("split", lambda: split_and_assign(model, result))
    if "surface" in formats and result.loops:
        result.surface = timed("surface", lambda: parting_surface(
            model, result, surface.mode, surface.distance, surface.height))

    written: dict = {}
    if outdir is not None:
        written = write_line_outputs(result, outdir, formats)
        if result.split is not None:
            written.update({f"{k}_faces": v for k, v in write_split(result.split, outdir).items()})
        if result.surface is not None:
            p = outdir / "parting_surface.step"
            write_step([result.surface.shape], p)
            written["surface"] = str(p)
        report_path = outdir / "report.json"
        written["report"] = str(report_path)
        write_report(model, result, report_path, written, auto_info)
    return result, written
