"""Phase 4: parting surface (sweep / ruled) and its self-intersection report."""

from __future__ import annotations

import functools
import math

import pytest
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_FACE

from conftest import GENERATED, run
from partingline.cli import main
from partingline.occ import indexed_map
from partingline.surface import parting_surface


@functools.lru_cache(maxsize=None)
def surf(name: str, mode: str, distance: float = 20.0):
    model, result = run(name)
    return parting_surface(model, result, mode, distance)


def test_box_sweep_strips_and_corner_sectors():
    s = surf("box_drafted", "sweep")
    assert s.faces == 8  # 4 strips + 4 corner sectors
    assert s.area == pytest.approx(2 * (40 + 30) * 20 + math.pi * 20**2, rel=1e-6)
    assert s.self_intersections == 0


def test_box_ruled_to_rectangle():
    s = surf("box_drafted", "ruled")
    assert s.area == pytest.approx((40 + 40) * (30 + 40) - 40 * 30, rel=1e-6)
    assert s.self_intersections == 0
    assert s.height == pytest.approx(0.0, abs=1e-9)  # the bottom loop's height


def test_sphere_sweep_is_annulus():
    s = surf("sphere", "sweep")
    assert s.area == pytest.approx(math.pi * (30**2 - 10**2), rel=1e-3)
    assert s.self_intersections == 0


def test_sphere_ruled():
    s = surf("sphere", "ruled")
    assert s.area == pytest.approx(60 * 60 - math.pi * 100, rel=1e-3)


def test_cylinder_sweep_has_no_self_intersection():
    s = surf("horizontal_cylinder", "sweep")
    # side strips, half-tunnels over the cap arcs, and four corner sectors
    expected = 2 * 50 * 20 + 2 * math.pi * 8 * 20 + math.pi * 20**2
    assert s.area == pytest.approx(expected, rel=1e-3)
    assert s.self_intersections == 0


def test_hub_bore_is_capped_and_ruled_surface_is_clean():
    s = surf("hub_blades", "ruled")
    assert len(s.filled_loops) == 1
    assert s.self_intersections == 0


def test_self_intersections_are_reported_not_hidden():
    # the stepped hub/blade line overlaps itself when swept straight outward
    s = surf("hub_blades", "sweep")
    assert s.self_intersections > 0
    assert any("intersects itself" in w for w in s.warnings)


def test_cli_writes_parting_surface(tmp_path):
    rc = main(["analyze", str(GENERATED / "box_drafted.step"), "--formats", "surface",
               "--surface", "sweep", "--surface-distance", "10", "--out", str(tmp_path)])
    assert rc == 0
    rd = STEPControl_Reader()
    assert rd.ReadFile(str(tmp_path / "parting_surface.step")) == IFSelect_RetDone
    rd.TransferRoots()
    assert indexed_map(rd.OneShape(), TopAbs_FACE).Extent() == 8
