"""Generate the synthetic STEP fixtures with cadquery.

    python tests/make_fixtures.py            # writes fixtures/generated/*.step
"""

from __future__ import annotations

import math
from pathlib import Path

import cadquery as cq
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "fixtures" / "generated"

# Twisted-blade parameters (shared with the tests)
BLADE_CHORD = 30.0
BLADE_ROOT_Y = 10.0
BLADE_TIP_Y = 110.0
BLADE_PITCH_ROOT = 45.0  # deg
BLADE_PITCH_TIP = 15.0   # deg -> 30 deg twist
BLADE_SECTIONS = 6


def sphere():
    return cq.Workplane("XY").sphere(10.0)


def horizontal_cylinder():
    # axis along X, x from 0 to 50, radius 8
    return cq.Workplane("YZ").circle(8.0).extrude(50.0)


def box_drafted():
    # 40 x 30 base at z = 0, walls drafted 2 deg inward toward +Z
    return cq.Workplane("XY").rect(40.0, 30.0).extrude(20.0, taper=2.0)


def box_vertical():
    return cq.Workplane("XY").rect(40.0, 30.0).extrude(20.0)


def tslot_block():
    block = cq.Workplane("XY").box(60.0, 40.0, 30.0, centered=(True, True, False))
    profile = [(-5, 31), (5, 31), (5, 20), (12, 20), (12, 10), (-12, 10), (-12, 20), (-5, 20)]
    slot = cq.Workplane("XZ").polyline(profile).close().extrude(25.0, both=True)
    return block.cut(slot)


def naca4_closed(m=0.04, p=0.4, t=0.12, n=61) -> np.ndarray:
    """NACA 4-digit with a closed (sharp) trailing edge.

    Returns (2n-1, 2) points: TE -> upper -> LE -> lower -> TE, unit chord.
    """
    beta = np.linspace(0.0, math.pi, n)
    x = 0.5 * (1 - np.cos(beta))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    yc = np.where(x < p, m / p**2 * (2 * p * x - x**2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2))
    dyc = np.where(x < p, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    th = np.arctan(dyc)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    upper = np.stack([xu, yu], axis=1)[::-1]   # TE -> LE
    lower = np.stack([xl, yl], axis=1)[1:]     # LE -> TE (skip duplicate LE)
    pts = np.vstack([upper, lower])
    pts[0] = pts[-1] = (1.0, 0.0)              # exactly sharp TE
    return pts


def blade_section(y: float) -> tuple[np.ndarray, float]:
    """3D points of the airfoil section at span y, and its pitch angle (deg)."""
    s = (y - BLADE_ROOT_Y) / (BLADE_TIP_Y - BLADE_ROOT_Y)
    pitch = BLADE_PITCH_ROOT + s * (BLADE_PITCH_TIP - BLADE_PITCH_ROOT)
    a = math.radians(pitch)
    af = naca4_closed()
    xa = (af[:, 0] - 0.25) * BLADE_CHORD
    za = af[:, 1] * BLADE_CHORD
    # chord inclined by the pitch angle in the XZ plane; Z is the hub axis
    X = xa * math.cos(a) - za * math.sin(a)
    Z = xa * math.sin(a) + za * math.cos(a)
    return np.stack([X, np.full_like(X, y), Z], axis=1), pitch


def twisted_blade():
    wires = []
    for y in np.linspace(BLADE_ROOT_Y, BLADE_TIP_Y, BLADE_SECTIONS):
        pts, _ = blade_section(float(y))
        edge = cq.Edge.makeSpline([cq.Vector(*p) for p in pts])
        wires.append(cq.Wire.assembleEdges([edge]))
    solid = cq.Solid.makeLoft(wires, ruled=False)
    return cq.Workplane("XY").add(solid)


HUB_R, HUB_H, BORE_R = 20.0, 40.0, 6.0


def hub_blades():
    """3 twisted blades fused to a vertical hub with a bore (hub axis = Z)."""
    hub = (cq.Workplane("XY").circle(HUB_R).circle(BORE_R).extrude(HUB_H)
           .translate((0, 0, -HUB_H / 2)).val())
    wires = []
    for y in np.linspace(8.0, 80.0, 4):
        s = (y - 8.0) / 72.0
        a = math.radians(BLADE_PITCH_ROOT + s * (BLADE_PITCH_TIP - BLADE_PITCH_ROOT))
        af = naca4_closed(n=41)
        xa = (af[:, 0] - 0.25) * BLADE_CHORD
        za = af[:, 1] * BLADE_CHORD
        X = xa * math.cos(a) - za * math.sin(a)
        Z = xa * math.sin(a) + za * math.cos(a)
        edge = cq.Edge.makeSpline([cq.Vector(x, y, z) for x, z in zip(X, Z)])
        wires.append(cq.Wire.assembleEdges([edge]))
    blade = cq.Solid.makeLoft(wires, ruled=False)
    prop = hub
    for k in range(3):
        prop = prop.fuse(blade.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), 120.0 * k))
    return cq.Workplane("XY").add(prop.clean())


FIXTURES = {
    "sphere": sphere,
    "horizontal_cylinder": horizontal_cylinder,
    "box_drafted": box_drafted,
    "box_vertical": box_vertical,
    "tslot_block": tslot_block,
    "twisted_blade": twisted_blade,
    "hub_blades": hub_blades,
}


def build_all(outdir: Path = GENERATED) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, fn in FIXTURES.items():
        path = outdir / f"{name}.step"
        cq.exporters.export(fn(), str(path))
        paths[name] = path
    return paths


if __name__ == "__main__":
    for name, path in build_all().items():
        print(f"{name:22s} -> {path}")
