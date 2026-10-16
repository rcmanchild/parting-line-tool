"""Command-line interface.

    partingline analyze part.step --pull +Z --formats step,dxf,csv --out out/
    partingline analyze part.step --auto --formats all
    partingline auto part.step             # rank pull directions only
    partingline faces part.step            # list faces, to pick --pull-face
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .analysis import VERTICAL_RULES, Options
from . import job
from .auto import rank_directions, table
from .export import OUTPUT_FORMATS, summary_text
from .job import SurfaceSpec
from .faces import FaceCtx, axis_of
from .model import load_model
from .pull import describe, parse_pull, pull_from_face
from .surface import MODES as SURFACE_MODES


def _formats(text: str) -> list[str]:
    if text.strip().lower() == "all":
        return list(OUTPUT_FORMATS)
    fmts = [f.strip().lower().lstrip(".") for f in text.split(",") if f.strip()]
    fmts = ["iges" if f == "igs" else "step" if f == "stp" else f for f in fmts]
    bad = [f for f in fmts if f not in OUTPUT_FORMATS]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown format(s) {bad}; choose from {OUTPUT_FORMATS} or 'all'")
    return fmts


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("step", type=Path, help="input STEP file (units: mm; non-mm files are converted and reported)")
    p.add_argument("--no-heal", action="store_true", help="skip ShapeFix/sewing")


def _options(args) -> Options:
    return Options(
        draft_tol_deg=args.draft_tol,
        vertical=args.vertical,
        gap_tol=args.gap_tol,
        undercut=not args.no_undercut,
        undercut_min_frac=args.undercut_frac,
        undercut_min_area=args.undercut_min_area,
        low_draft_deg=args.low_draft,
        bridge_tol=args.bridge_tol,
    )


def _rank(model, args):
    cands, info = rank_directions(model, samples=args.auto_samples, top=args.auto_top,
                                  draft_tol_deg=args.draft_tol, low_draft_deg=args.low_draft,
                                  undercut_weight=args.auto_undercut_weight)
    print(table(cands, info))
    print(f"          {info['samples']} directions in {info['time_s']:.1f} s")
    return cands, info


def cmd_analyze(args) -> int:
    t0 = time.perf_counter()
    model = load_model(args.step, do_heal=not args.no_heal, gap_tol=args.gap_tol)
    t_load = time.perf_counter() - t0

    auto_info = None
    if args.auto:
        cands, auto_info = _rank(model, args)
        if not 1 <= args.auto_pick <= len(cands):
            raise ValueError(f"--auto-pick must be between 1 and {len(cands)}")
        d, how = cands[args.auto_pick - 1].d, f"--auto rank {args.auto_pick}"
        t_load += auto_info["time_s"]
    elif args.pull_face is not None:
        d, how = pull_from_face(model, args.pull_face)
    else:
        d, how = parse_pull(args.pull), "given"
    if args.flip:
        d = -d

    outdir = args.out or Path(f"{args.step.stem}_parting")
    surface = SurfaceSpec(args.surface, args.surface_distance, args.surface_height)
    result, written = job.run(model, d, _options(args), args.formats, outdir, surface, auto_info, t_load)
    result.timings["load_heal"] = round(time.perf_counter() - t0 - sum(
        v for k, v in result.timings.items() if k not in ("total", "load_heal")), 3)

    print(summary_text(model, result))
    if how != "given":
        print(f"Pull from {how}: {describe(d)}{' (flipped)' if args.flip else ''}")
    print("Wrote   : " + ", ".join(written.values()))
    return 0


def cmd_auto(args) -> int:
    model = load_model(args.step, do_heal=not args.no_heal)
    _rank(model, args)
    return 0


def cmd_faces(args) -> int:
    model = load_model(args.step, do_heal=not args.no_heal)
    print(f"{model.n_faces} faces  (bbox diag {model.diag:.2f} mm)")
    print(f"{'#':>5}  {'type':<10} {'area':>12}  pull candidate")
    for i, face in enumerate(model.faces):
        fc = FaceCtx(i, face)
        info = ""
        if fc.type_name == "plane":
            _, n = fc.point_normal(0.5 * (fc.u0 + fc.u1), 0.5 * (fc.v0 + fc.v1))
            info = f"normal {describe(n)}"
        else:
            ax = axis_of(fc)
            if ax is not None:
                extra = ""
                if fc.type_name == "cylinder":
                    extra = f"  r={fc.ad.Cylinder().Radius():.3f}"
                info = f"axis {describe(ax[0])} through {np.round(ax[1], 3).tolist()}{extra}"
        print(f"{i + 1:>5}  {fc.type_name:<10} {fc.area:>12.3f}  {info}")
    return 0


def _add_auto(p: argparse.ArgumentParser) -> None:
    p.add_argument("--auto-samples", type=int, default=500, metavar="N", help="directions to try (default 500)")
    p.add_argument("--auto-top", type=int, default=5, metavar="N", help="how many to show (default 5)")
    p.add_argument("--auto-undercut-weight", type=float, default=1.0, metavar="W",
                   help="score = W x undercut area + near-zero-draft area (default 1)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="partingline", description="Parting-line tool for mold design")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="compute the parting line for a pull direction")
    _add_common(a)
    g = a.add_mutually_exclusive_group()
    g.add_argument("--pull", default="+Z", help="+X,-X,+Y,-Y,+Z,-Z or a vector 'x,y,z' (default +Z)")
    g.add_argument("--pull-face", type=int, metavar="N",
                   help="1-based face number: planar -> its normal, cylinder/cone -> its axis")
    g.add_argument("--auto", action="store_true",
                   help="rank pull directions (score = undercut area + near-zero-draft area) and use the best")
    _add_auto(a)
    a.add_argument("--auto-pick", type=int, default=1, metavar="N", help="with --auto, use rank N (default 1)")
    a.add_argument("--flip", action="store_true", help="reverse the pull direction")
    a.add_argument("--draft-tol", type=float, default=0.5, metavar="DEG",
                   help="faces within this draft angle count as vertical (default 0.5)")
    a.add_argument("--vertical", choices=VERTICAL_RULES, default="top",
                   help="where the line goes on vertical bands (default top, toward +pull)")
    a.add_argument("--gap-tol", type=float, default=0.01, metavar="MM", help="chaining gap tolerance (default 0.01)")
    a.add_argument("--bridge-tol", type=float, default=0.5, metavar="MM",
                   help="close remaining gaps up to this length with reported straight bridges (default 0.5; 0 = off)")
    a.add_argument("--formats", type=_formats, default=["step"],
                   help="comma list of step,iges,dxf,csv (line), split (cavity/core face shells), "
                        "surface (parting surface), or 'all' (default step)")
    a.add_argument("--out", type=Path, help="output directory (default <name>_parting)")
    a.add_argument("--surface", choices=SURFACE_MODES, default="sweep",
                   help="parting surface: sweep outward perpendicular to the pull, or ruled to a "
                        "rectangle in a plane normal to the pull (default sweep)")
    a.add_argument("--surface-distance", type=float, default=20.0, metavar="MM",
                   help="sweep distance / rectangle margin (default 20)")
    a.add_argument("--surface-height", type=float, metavar="MM",
                   help="ruled mode: height of the rectangle plane along the pull (default: mean line height)")
    a.add_argument("--no-undercut", action="store_true", help="skip the ray-cast undercut check")
    a.add_argument("--undercut-frac", type=float, default=0.05,
                   help="flag a face as undercut when this fraction of it is blocked (default 0.05)")
    a.add_argument("--undercut-min-area", type=float, default=0.5, metavar="MM2",
                   help="ignore undercuts smaller than this blocked area (default 0.5)")
    a.add_argument("--low-draft", type=float, default=1.0, metavar="DEG",
                   help="report area with draft below this angle (default 1.0)")
    a.set_defaults(func=cmd_analyze)

    r = sub.add_parser("auto", help="rank pull directions over a hemisphere and show the top ones")
    _add_common(r)
    _add_auto(r)
    r.add_argument("--draft-tol", type=float, default=0.5, metavar="DEG")
    r.add_argument("--low-draft", type=float, default=1.0, metavar="DEG")
    r.set_defaults(func=cmd_auto)

    f = sub.add_parser("faces", help="list faces with numbers, types and normals/axes")
    _add_common(f)
    f.set_defaults(func=cmd_faces)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
