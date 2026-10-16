"""Pull-direction input: presets, explicit vectors, and directions from faces."""

from __future__ import annotations

import numpy as np

from .faces import FaceCtx, axis_of

PRESETS = {
    "+X": (1, 0, 0), "-X": (-1, 0, 0),
    "+Y": (0, 1, 0), "-Y": (0, -1, 0),
    "+Z": (0, 0, 1), "-Z": (0, 0, -1),
}


def parse_pull(text: str) -> np.ndarray:
    """'+Z', 'z', '-x' or '0,0,1' -> unit vector."""
    t = text.strip().upper().replace(" ", "")
    if t in ("X", "Y", "Z"):
        t = "+" + t
    if t in PRESETS:
        return np.array(PRESETS[t], dtype=float)
    try:
        parts = [float(p) for p in t.split(",")]
    except ValueError:
        raise ValueError(f"cannot parse pull direction {text!r}; use +X/-X/+Y/-Y/+Z/-Z or x,y,z") from None
    if len(parts) != 3:
        raise ValueError(f"pull vector needs 3 components, got {text!r}")
    v = np.array(parts)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("pull vector must be non-zero")
    return v / n


def pull_from_face(model, face_number: int) -> tuple[np.ndarray, str]:
    """Direction from a 1-based face number: planar -> outward normal,
    cylinder/cone/revolution -> axis.  Returns (unit vector, description)."""
    if not 1 <= face_number <= model.n_faces:
        raise ValueError(f"face {face_number} out of range 1..{model.n_faces}")
    fc = FaceCtx(face_number - 1, model.faces[face_number - 1])
    if fc.type_name == "plane":
        _, n = fc.point_normal(0.5 * (fc.u0 + fc.u1), 0.5 * (fc.v0 + fc.v1))
        return n / np.linalg.norm(n), f"normal of planar face {face_number}"
    ax = axis_of(fc)
    if ax is not None:
        d = ax[0] / np.linalg.norm(ax[0])
        return d, f"axis of {fc.type_name} face {face_number}"
    raise ValueError(
        f"face {face_number} is a {fc.type_name} surface; pick a planar face (normal) "
        "or a cylindrical/conical face (axis)"
    )


def describe(d: np.ndarray) -> str:
    for name, v in PRESETS.items():
        if np.allclose(d, v, atol=1e-12):
            return name
    return f"{d[0]:.6g},{d[1]:.6g},{d[2]:.6g}"
