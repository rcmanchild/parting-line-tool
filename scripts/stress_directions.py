"""Robustness check: analyse + split every fixture at random pull directions.

    python scripts/stress_directions.py [N] [SEED]

A direction is reported as bad when the parting line has open chains or
junctions, the split has conflicts, or the cavity half's open boundary length
differs from the parting-line length by more than 0.1 %.  Uses the generated
fixtures (run `python tests/make_fixtures.py` first) plus any real parts in
fixtures/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from partingline.analysis import Options, analyze
from partingline.model import load_model
from partingline.split import split_and_assign

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    files = sorted((ROOT / "fixtures" / "generated").glob("*.step"))
    files += sorted(p for p in (ROOT / "fixtures").glob("*") if p.suffix.lower() in (".step", ".stp"))
    total_bad = 0
    for f in files:
        model = load_model(f)
        bad, t0, tmax = [], time.perf_counter(), 0.0
        for d in dirs:
            t = time.perf_counter()
            try:
                r = analyze(model, d, Options())
                lp = r.stats["loops"]
                sp = split_and_assign(model, r) if lp["count"] else None
                conf = len(sp.conflicts) if sp else 0
                gap = abs(sp.boundary_length["cavity"] - lp["total_length"]) if sp else 0.0
                if lp["open"] or lp["junctions"] or conf or gap > 1e-3 * max(lp["total_length"], 1.0):
                    bad.append((np.round(d, 3).tolist(), f"open {lp['open']}", f"junctions {lp['junctions']}",
                                f"conflicts {conf}", f"boundary-gap {gap:.3f}",
                                f"bridges {len(r.stats.get('bridges', []))}"))
            except Exception as exc:  # report and keep going
                bad.append((np.round(d, 3).tolist(), "EXCEPTION", repr(exc)[:100]))
            tmax = max(tmax, time.perf_counter() - t)
        total_bad += len(bad)
        print(f"{f.name:28s} {len(dirs)} dirs, {len(bad)} bad, max {tmax:.1f} s, total {time.perf_counter() - t0:.1f} s")
        for b in bad[:8]:
            print("     ", *b)
    return 1 if total_bad else 0


if __name__ == "__main__":
    sys.exit(main())
