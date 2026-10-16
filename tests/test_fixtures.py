"""Parting-line results on the synthetic cadquery fixtures."""

from __future__ import annotations

import math

import numpy as np
import pytest
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAPI import GeomAPI_ProjectPointOnCurve, GeomAPI_ProjectPointOnSurf
from OCP.gp import gp_Pnt, gp_Vec

from conftest import run
from make_fixtures import BLADE_CHORD, BLADE_ROOT_Y, BLADE_SECTIONS, BLADE_TIP_Y, blade_section
from partingline import segments as sg


def _single_closed_loop(result):
    assert result.stats["loops"]["count"] == 1, result.stats["loops"]
    loop = result.loops[0]
    assert loop.closed
    assert result.stats["loops"]["junctions"] == 0
    return loop


def _wire_points(result, n=400):
    """Points sampled on the exported wire curves (not the polylines)."""
    out = []
    for s in result.segments:
        for t in np.linspace(s.t0, s.t1, n):
            p = s.curve.Value(float(t))
            out.append((p.X(), p.Y(), p.Z()))
    return np.array(out)


# --- sphere ------------------------------------------------------------------

def test_sphere_equator_circle():
    _, r = run("sphere")
    loop = _single_closed_loop(r)
    assert set(loop.kinds) == {sg.SILHOUETTE}
    for pts in (loop.points, _wire_points(r)):
        radius = np.linalg.norm(pts[:, :2], axis=1)
        assert np.abs(radius - 10.0).max() < 1e-3
        assert np.abs(pts[:, 2]).max() < 1e-3
    assert loop.length == pytest.approx(2 * math.pi * 10.0, abs=1e-3)


# --- horizontal cylinder -----------------------------------------------------

def _side_lines(result):
    sides = []
    for s in result.segments:
        p = s.points
        if (np.abs(p[:, 2]).max() < 1e-6 and np.abs(np.abs(p[:, 1]) - 8.0).max() < 1e-6
                and np.ptp(p[:, 0]) > 49.999):
            sides.append(s)
    return sides


def test_horizontal_cylinder_two_side_lines():
    _, r = run("horizontal_cylinder")
    sides = _side_lines(r)
    assert len(sides) == 2
    ys = sorted(float(np.sign(s.points[0, 1])) for s in sides)
    assert ys == [-1.0, 1.0]
    for s in sides:
        assert s.length == pytest.approx(50.0, abs=1e-6)
    # Flat end caps are vertical faces: flagged, and with the default 'top'
    # rule the line crosses each cap along its upper half circle.
    assert len(r.stats["vertical_faces"]) == 2
    loop = _single_closed_loop(r)
    assert loop.length == pytest.approx(100.0 + 2 * math.pi * 8.0, abs=1e-3)


def test_horizontal_cylinder_middle_rule_is_rectangle():
    _, r = run("horizontal_cylinder", vertical="middle")
    assert len(_side_lines(r)) == 2
    loop = _single_closed_loop(r)
    assert np.abs(loop.points[:, 2]).max() < 1e-6
    assert loop.length == pytest.approx(2 * 50.0 + 2 * 16.0, abs=1e-3)


# --- boxes -------------------------------------------------------------------

def test_box_drafted_bottom_loop():
    _, r = run("box_drafted")
    loop = _single_closed_loop(r)
    assert np.abs(loop.points[:, 2]).max() < 1e-6  # bottom edge loop at z = 0
    assert loop.length == pytest.approx(140.0, abs=1e-6)
    assert r.stats["vertical_faces"] == []
    assert not any(f.needs_draft for f in r.faces)
    assert r.stats["min_draft_deg"]["cavity"] == pytest.approx(2.0, abs=1e-6)
    assert r.stats["undercut_area"] == 0.0


def test_box_vertical_top_loop_and_flags():
    _, r = run("box_vertical")
    loop = _single_closed_loop(r)
    assert np.abs(loop.points[:, 2] - 20.0).max() < 1e-6  # top edge by default
    assert loop.length == pytest.approx(140.0, abs=1e-6)
    assert len(r.stats["vertical_faces"]) == 4
    assert r.stats["vertical_area"] == pytest.approx(2 * (40 + 30) * 20.0, rel=1e-6)
    assert r.stats["undercut_area"] == 0.0


@pytest.mark.parametrize("rule, z", [("bottom", 0.0), ("middle", 10.0)])
def test_box_vertical_rules(rule, z):
    _, r = run("box_vertical", vertical=rule)
    loop = _single_closed_loop(r)
    assert np.abs(loop.points[:, 2] - z).max() < 1e-6
    assert loop.length == pytest.approx(140.0, abs=1e-6)


# --- T-slot ------------------------------------------------------------------

def test_tslot_undercut_detected():
    model, r = run("tslot_block")
    assert r.stats["undercut_area"] > 0
    flagged = [f for f in r.faces if f.undercut]
    assert flagged
    # the downward-facing lip ceilings (z = 20, |x| between 5 and 12) are undercuts
    ceilings = []
    for f in r.faces:
        face = model.faces[f.number - 1]
        ad = BRepAdaptor_Surface(face)
        c = ad.Value(0.5 * (ad.FirstUParameter() + ad.LastUParameter()),
                     0.5 * (ad.FirstVParameter() + ad.LastVParameter()))
        if f.type == "plane" and f.cls == "core" and abs(c.Z() - 20.0) < 1e-6:
            ceilings.append(f)
    assert len(ceilings) == 2
    assert all(f.undercut and f.undercut_frac > 0.99 for f in ceilings)
    # the top face and outer walls are not undercuts
    assert sum(f.area * f.undercut_frac for f in r.faces if not f.undercut) == 0.0


# --- twisted blade -----------------------------------------------------------

def _plane_crossings(points: np.ndarray, y: float) -> np.ndarray:
    a, b = points[:-1], points[1:]
    s = (a[:, 1] - y) * (b[:, 1] - y)
    idx = np.nonzero(s < 0)[0]
    t = (y - a[idx, 1]) / (b[idx, 1] - a[idx, 1])
    return a[idx] + t[:, None] * (b[idx] - a[idx])


def test_twisted_blade_le_silhouette_and_te_edge():
    model, r = run("twisted_blade")
    loop = _single_closed_loop(r)
    assert loop.kinds.get(sg.EDGE, 0) > 0.95 * (BLADE_TIP_Y - BLADE_ROOT_Y)        # TE edge
    assert loop.kinds.get(sg.SILHOUETTE, 0) > 0.95 * (BLADE_TIP_Y - BLADE_ROOT_Y)  # LE silhouette

    te_seg = next(s for s in r.segments if s.kind == sg.EDGE)
    blade_face = next(model.faces[f.number - 1] for f in r.faces if f.cls == "mixed")
    surf = BRep_Tool.Surface_s(blade_face)
    # interior section planes, where the loft passes exactly through the profiles
    for y in np.linspace(BLADE_ROOT_Y, BLADE_TIP_Y, BLADE_SECTIONS)[1:-1]:
        sec, _ = blade_section(float(y))
        te, le = sec[0], sec[len(sec) // 2]
        hits = _plane_crossings(loop.points, float(y))
        assert len(hits) == 2, f"y={y}: {len(hits)} crossings"
        # the TE parting curve passes exactly through the profile's TE point
        assert GeomAPI_ProjectPointOnCurve(gp_Pnt(*te), te_seg.curve).LowerDistance() < 1e-6
        d_te = np.linalg.norm(hits - te, axis=1)
        assert d_te.min() < 2e-3  # one polyline crossing is on the TE (polyline deflection 1e-3)
        other = hits[int(np.argmax(d_te))]
        assert np.linalg.norm(other - le) < 0.15 * BLADE_CHORD  # the other is near the LE
        # ... and it is a true silhouette point: surface normal perpendicular to Z
        proj = GeomAPI_ProjectPointOnSurf(gp_Pnt(*other), surf)
        u, v = proj.LowerDistanceParameters()
        ad = BRepAdaptor_Surface(blade_face)
        p, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
        ad.D1(u, v, p, du, dv)
        n = du.Crossed(dv)
        # polyline interpolation between exact points: within the 1e-3 mm chord tol
        assert abs(n.Z() / n.Magnitude()) < 5e-3


# --- hub with blades (vertical band interrupted by blades) ------------------

def test_hub_blades_single_loop_with_generators():
    from make_fixtures import BORE_R, HUB_H, HUB_R
    _, r = run("hub_blades")
    assert r.stats["loops"]["count"] == 2
    assert r.stats["loops"]["closed"] == 2 and r.stats["loops"]["junctions"] == 0
    loops = sorted(r.loops, key=lambda lp: lp.length)
    bore, main = loops
    assert bore.length == pytest.approx(2 * math.pi * BORE_R, abs=1e-3)
    # main loop: 3 TE edges, 3 LE silhouettes, and the hub band between them
    items = [r.segments[si] for si, _ in main.items]
    assert sum(1 for s in items if s.kind == sg.EDGE) == 3
    assert sum(1 for s in items if s.kind == sg.SILHOUETTE) == 3
    gens = [s for s in items if s.source.get("generator")]
    assert len(gens) == 6  # TE and LE root of each blade up to the hub top
    for s in gens:
        direction = s.points[-1] - s.points[0]
        assert np.linalg.norm(np.cross(direction / np.linalg.norm(direction), [0, 0, 1])) < 1e-9
        assert max(s.points[0, 2], s.points[-1, 2]) == pytest.approx(HUB_H / 2, abs=1e-6)
        # starts at the blade root vertex, which the boolean fuse placed within ~1e-5 of the hub
        assert np.linalg.norm(s.points[0, :2]) == pytest.approx(HUB_R, abs=1e-4)
    # only the stretches of the hub-top rim between blades are parting line
    rim = sum(s.length for s in items if s.kind == sg.VERTICAL and not s.source.get("generator")
              and abs(s.points[:, 2] - HUB_H / 2).max() < 1e-6)
    assert 0 < rim < 2 * math.pi * HUB_R
