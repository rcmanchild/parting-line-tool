"""Phase 2: faces split along the parting line and sorted into cavity / core."""

from __future__ import annotations

import functools
import math

import pytest

from conftest import run
from partingline.split import split_and_assign

FIXTURES = ["sphere", "horizontal_cylinder", "box_drafted", "box_vertical", "tslot_block",
            "twisted_blade", "hub_blades"]


@functools.lru_cache(maxsize=None)
def split(name: str, vertical: str = "top"):
    model, result = run(name, vertical=vertical)
    return result, split_and_assign(model, result)


@pytest.mark.parametrize("name", FIXTURES)
def test_split_is_consistent(name):
    result, sp = split(name)
    assert sp.conflicts == []
    assert sp.cavity_faces and sp.core_faces
    original = sum(f.area for f in result.faces)
    assert sp.areas["cavity"] + sp.areas["core"] == pytest.approx(original, rel=1e-6)
    # each half's open boundary is exactly the parting line
    line = result.stats["loops"]["total_length"]
    assert sp.boundary_length["cavity"] == pytest.approx(line, rel=1e-6)
    assert sp.boundary_length["core"] == pytest.approx(line, rel=1e-6)


def test_sphere_halves():
    _, sp = split("sphere")
    assert sp.areas["cavity"] == pytest.approx(2 * math.pi * 100, rel=1e-6)
    assert sp.areas["core"] == pytest.approx(2 * math.pi * 100, rel=1e-6)


def test_cylinder_upper_half_is_cavity():
    _, sp = split("horizontal_cylinder")
    assert sp.areas["cavity"] == pytest.approx(math.pi * 8 * 50, rel=1e-6)


def test_box_drafted_only_bottom_is_core():
    _, sp = split("box_drafted")
    assert sp.areas["core"] == pytest.approx(40 * 30, rel=1e-9)


@pytest.mark.parametrize("rule, cavity_area", [("top", 40 * 30), ("bottom", 40 * 30 + 2 * (40 + 30) * 20)])
def test_box_vertical_walls_follow_rule(rule, cavity_area):
    _, sp = split("box_vertical", rule)
    assert sp.areas["cavity"] == pytest.approx(cavity_area, rel=1e-9)


def test_hub_blades_two_halves():
    _, sp = split("hub_blades")
    shells = {w: sp.shells(w) for w in ("cavity", "core")}
    from OCP.TopAbs import TopAbs_SHELL
    from partingline.occ import indexed_map
    assert indexed_map(shells["cavity"], TopAbs_SHELL).Extent() == 1


# oblique pulls that used to fail (from scripts/stress_directions.py)
OBLIQUE = [
    # T-slot walls 0.17 deg off vertical: generators must follow the wall
    ("tslot_block", (0.003, 0.737, -0.676)),
    # blade silhouettes running almost along the blade edge leave slivers
    ("hub_blades", (-0.889, -0.434, -0.147)),
    ("hub_blades", (-0.178, -0.962, 0.206)),
    ("hub_blades", (-0.265, 0.96, -0.093)),
]


@pytest.mark.parametrize("name,pull", OBLIQUE)
def test_oblique_pull_closes_and_splits(name, pull):
    n = math.sqrt(sum(c * c for c in pull))
    model, result = run(name, pull=tuple(c / n for c in pull))
    lp = result.stats["loops"]
    assert lp["open"] == 0 and lp["junctions"] == 0
    sp = split_and_assign(model, result)
    assert sp.conflicts == []
    assert sp.boundary_length["cavity"] == pytest.approx(lp["total_length"], rel=1e-3)
