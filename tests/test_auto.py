"""--auto direction ranking, hemisphere sampling and gap bridges."""

from __future__ import annotations

import json

import numpy as np
from OCP.GC import GC_MakeSegment

from conftest import model_for
from make_fixtures import GENERATED
from partingline import segments as sg
from partingline.analysis import _bridges
from partingline.auto import hemisphere, rank_directions
from partingline.chain import chain
from partingline.cli import main
from partingline.occ import pnt


def test_hemisphere_unit_and_upper():
    dirs = hemisphere(200)
    assert dirs.shape == (203, 3)
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0)
    assert (dirs[:, 2] >= 0.0).all()
    assert np.allclose(dirs[:3], np.eye(3))  # +X, +Y, +Z always sampled


def test_auto_box_drafted_prefers_z():
    cands, info = rank_directions(model_for("box_drafted"), samples=200)
    assert cands[0].label == "+Z"
    assert cands[0].score == 0.0
    assert info["presets"]["+Z"]["rank"] == 1


def test_auto_tslot_prefers_slot_axis():
    # the T-slot runs along Y: only (near) +/-Y pulls release it without undercut
    cands, _ = rank_directions(model_for("tslot_block"), samples=200)
    assert abs(cands[0].d[1]) > 0.95
    assert cands[0].undercut_area == 0.0
    z = rank_directions(model_for("tslot_block"), samples=200)[1]["presets"]["+Z"]
    assert z["undercut_area"] > 100.0


def test_cli_auto_uses_best(tmp_path, capsys):
    out = tmp_path / "o"
    rc = main(["analyze", str(GENERATED / "box_drafted.step"), "--auto", "--auto-samples", "100",
               "--out", str(out)])
    assert rc == 0
    text = capsys.readouterr().out
    assert "--auto rank 1" in text
    rep = json.loads((out / "report.json").read_text())
    assert np.allclose(rep["pull"]["vector"], [0, 0, 1])
    assert rep["auto"]["top"][0]["label"] == "+Z"


def _line(a, b):
    c = GC_MakeSegment(pnt(np.array(a, float)), pnt(np.array(b, float))).Value()
    return sg.from_curve(sg.EDGE, c, c.FirstParameter(), c.LastParameter(), 1e-3)


def test_small_gap_is_bridged_and_reported():
    # a 10 mm square with a 0.2 mm gap in one side
    segs = [_line((0, 0, 0), (10, 0, 0)), _line((10, 0, 0), (10, 10, 0)),
            _line((10, 10, 0), (0, 10, 0)), _line((0, 10, 0), (0, 0.2, 0))]
    ch = chain(segs, 0.01)
    assert len(ch.dangling) == 2 and not ch.loops[0].closed
    assert _bridges(ch, 0.1, 1e-3) == []          # too long for the tolerance
    bridges = _bridges(ch, 0.5, 1e-3)
    assert len(bridges) == 1 and bridges[0].kind == sg.BRIDGE
    assert abs(bridges[0].length - 0.2) < 1e-9
    closed = chain(segs + bridges, 0.01)
    assert len(closed.loops) == 1 and closed.loops[0].closed
