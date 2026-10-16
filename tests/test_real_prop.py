"""Real propeller STEP files dropped into fixtures/ (skipped when none present).

Pull direction, per file, from an optional sidecar fixtures/<name>.json:
    {"pull": "0,0,1"}  or  {"pull_face": 17}
Without one we use the axis of the cylindrical face with the largest area
(normally the hub bore or hub OD).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from partingline.analysis import analyze
from partingline.export import summary_text
from partingline.faces import FaceCtx, axis_of
from partingline.model import load_model
from partingline.pull import parse_pull, pull_from_face
from partingline.split import split_and_assign

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
PROPS = sorted(p for p in FIXTURES.glob("*") if p.suffix.lower() in (".step", ".stp"))


def _pull_for(path: Path, model):
    side = path.with_suffix(".json")
    if side.exists():
        cfg = json.loads(side.read_text())
        if "pull_face" in cfg:
            return pull_from_face(model, int(cfg["pull_face"]))[0]
        return parse_pull(cfg["pull"])
    best = None
    for i, face in enumerate(model.faces):
        fc = FaceCtx(i, face)
        if fc.type_name == "cylinder" and (best is None or fc.area > best[0]):
            best = (fc.area, axis_of(fc)[0])
    if best is None:
        pytest.skip(f"{path.name}: no cylindrical face; add {side.name} with a pull direction")
    return best[1]


@pytest.mark.skipif(not PROPS, reason="no real prop STEP in fixtures/")
@pytest.mark.parametrize("path", PROPS, ids=[p.name for p in PROPS])
def test_real_prop(path):
    t0 = time.perf_counter()
    model = load_model(path)
    result = analyze(model, _pull_for(path, model))
    elapsed = time.perf_counter() - t0
    print("\n" + summary_text(model, result))
    loops = result.stats["loops"]
    assert loops["count"] >= 1
    assert loops["open"] == 0, "parting line is not closed"
    assert loops["junctions"] == 0
    assert elapsed < 30.0, f"{path.name} took {elapsed:.1f} s"

    sp = split_and_assign(model, result)
    assert sp.conflicts == []
    assert sp.areas["cavity"] + sp.areas["core"] == pytest.approx(sum(f.area for f in result.faces), rel=1e-5)
    assert sp.boundary_length["cavity"] == pytest.approx(loops["total_length"], rel=1e-5)
