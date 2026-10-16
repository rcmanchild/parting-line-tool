"""Viewer: headless screenshot, picking and key handling (skipped without offscreen GL)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import model_for
from make_fixtures import GENERATED
from partingline.analysis import Options
from partingline.cli import main

viewer = pytest.importorskip("partingline.viewer")
if not viewer.offscreen_available():
    pytest.skip("VTK offscreen rendering is not available here", allow_module_level=True)


def _png_size(path):
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def test_cli_screenshot(tmp_path, capsys):
    png = tmp_path / "shot.png"
    rc = main(["view", str(GENERATED / "hub_blades.step"), "--split", "--screenshot", str(png),
               "--size", "640", "480"])
    assert rc == 0
    assert _png_size(png) == (640, 480)
    assert "Split   :" in capsys.readouterr().out


@pytest.fixture
def box_viewer():
    state = viewer.ViewState(pull=np.array([0.0, 0.0, 1.0]), opts=Options())
    v = viewer.Viewer(model_for("box_drafted"), state, size=(400, 300), offscreen=True)
    v.recompute()
    v.reset_camera()
    v.window.Render()
    return v


def test_pick_and_keys(box_viewer, tmp_path):
    v = box_viewer
    face = v.pick(200, 150)  # the centre of the view is on the part
    assert face is not None
    assert "picked face" in v.status.GetInput()

    v.key("x")
    assert np.allclose(v.state.pull, [1, 0, 0])
    v.key("y", shift=True)
    assert np.allclose(v.state.pull, [0, -1, 0])
    v.key("f")
    assert np.allclose(v.state.pull, [0, 1, 0])
    v.key("v")
    assert v.state.opts.vertical == "bottom"
    v.key("z")

    v.key("s")
    assert v.result.split is not None
    assert len(v._labels) == v.result.split.model.n_faces
    v.key("s")
    assert v.result.split is None

    # pull from a picked planar face gives its normal
    top = max((f for f in v.result.faces if f.type == "plane"), key=lambda f: f.min_draft_deg).number - 1
    v.state.picked = top
    v.key("p")
    assert np.allclose(v.state.pull, [0, 0, 1])

    v.state.outdir = tmp_path / "exp"
    v.state.formats = ("step",)
    v.key("e")
    assert (tmp_path / "exp" / "parting_line.step").exists()
    assert v.key("q") is False
