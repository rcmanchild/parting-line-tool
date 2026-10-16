"""CLI, output files, pull-direction input and unit handling."""

from __future__ import annotations

import json

import numpy as np
import pytest
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
from OCP.TopAbs import TopAbs_EDGE

from conftest import model_for
from make_fixtures import GENERATED, box_vertical
from partingline.cli import main
from partingline.faces import FaceCtx
from partingline.model import load_model
from partingline.occ import indexed_map
from partingline.pull import parse_pull, pull_from_face


def test_cli_writes_all_outputs(tmp_path, capsys):
    out = tmp_path / "out"
    rc = main(["analyze", str(GENERATED / "sphere.step"), "--pull", "+Z", "--formats", "all", "--out", str(out)])
    assert rc == 0
    text = capsys.readouterr().out
    assert "1 loop(s), 1 closed" in text
    for name in ("parting_line.step", "parting_line.iges", "parting_line.dxf", "parting_line.csv", "report.json",
                 "cavity_faces.step", "core_faces.step"):
        assert (out / name).stat().st_size > 0, name

    # STEP round trip: OCC writes wireframes as a GEOMETRIC_CURVE_SET, so the
    # file holds one trimmed curve per parting segment (loop grouping is in the report)
    rd = STEPControl_Reader()
    assert rd.ReadFile(str(out / "parting_line.step")) == IFSelect_RetDone
    rd.TransferRoots()
    shape = rd.OneShape()
    assert indexed_map(shape, TopAbs_EDGE).Extent() == rep_segments(out)

    # DXF: one closed polyline, projected (2D)
    dxf = (out / "parting_line.dxf").read_text().splitlines()
    assert dxf.count("POLYLINE") == 1 and dxf[-1] == "EOF"

    # CSV points lie on the equator
    pts = np.loadtxt(out / "parting_line.csv", delimiter=",", skiprows=1)[:, 3:6]
    assert np.abs(np.linalg.norm(pts[:, :2], axis=1) - 10).max() < 1e-3

    rep = json.loads((out / "report.json").read_text())
    assert rep["summary"]["loops"]["count"] == 1
    assert rep["summary"]["loops"]["closed"] == 1
    assert rep["input"]["units"] == "mm"
    assert rep["pull"]["label"] == "+Z"
    assert rep["split"]["conflicts"] == [] and rep["split"]["groups"] == 2


def rep_segments(out) -> int:
    rep = json.loads((out / "report.json").read_text())
    return sum(rep["summary"]["segments"].values())


def test_cli_faces_listing(capsys):
    assert main(["faces", str(GENERATED / "horizontal_cylinder.step")]) == 0
    text = capsys.readouterr().out
    assert "cylinder" in text and "axis" in text


def test_cli_pull_face_and_flip(tmp_path, capsys):
    model = model_for("box_vertical")
    top = next(i for i, f in enumerate(model.faces, start=1)
               if np.allclose(pull_from_face(model, i)[0], [0, 0, 1]))
    rc = main(["analyze", str(GENERATED / "box_vertical.step"), "--pull-face", str(top), "--flip",
               "--out", str(tmp_path)])
    assert rc == 0
    rep = json.loads((tmp_path / "report.json").read_text())
    assert rep["pull"]["label"] == "-Z"
    assert rep["loops"][0]["start"][2] == pytest.approx(0.0, abs=1e-9)  # 'top' toward -Z is z = 0


def test_cli_bad_pull(capsys):
    assert main(["analyze", str(GENERATED / "sphere.step"), "--pull", "up"]) == 2
    assert "cannot parse" in capsys.readouterr().err


@pytest.mark.parametrize("text, vec", [
    ("+Z", (0, 0, 1)), ("-x", (-1, 0, 0)), ("y", (0, 1, 0)), ("0,0,2", (0, 0, 1)), ("1, 1, 0", (0.7071068, 0.7071068, 0)),
])
def test_parse_pull(text, vec):
    assert np.allclose(parse_pull(text), vec, atol=1e-6)


def test_pull_from_cylinder_axis():
    model = model_for("horizontal_cylinder")
    cyl = next(i for i, face in enumerate(model.faces, start=1)
               if FaceCtx(i - 1, face).type_name == "cylinder")
    d, how = pull_from_face(model, cyl)
    assert abs(abs(d[0]) - 1) < 1e-12 and "axis" in how


def test_inch_file_is_reported_not_silently_scaled(tmp_path):
    path = tmp_path / "box_inch.step"
    Interface_Static.SetCVal_s("write.step.unit", "INCH")
    try:
        w = STEPControl_Writer()
        w.Transfer(box_vertical().val().wrapped, STEPControl_AsIs)
        assert w.Write(str(path)) == IFSelect_RetDone
    finally:
        Interface_Static.SetCVal_s("write.step.unit", "MM")
    assert "INCH" in path.read_text().upper()
    model = load_model(path)
    assert model.import_info.converted_from == "INCH"
    assert any("not mm" in w for w in model.heal_info.warnings)
    # geometry is in mm after the reader's conversion: 40 x 30 x 20
    assert np.allclose(model.bmax - model.bmin, [40, 30, 20], atol=1e-3)
