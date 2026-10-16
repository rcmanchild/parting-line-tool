"""--auto: rank pull directions by  score = W * undercut area + near-zero-draft area
(W = 1 by default, as specified; raise it to favour undercut-free directions).
Directions whose scores tie (within 0.01 % of the part area) are ordered by
draw depth (the part's extent along the pull), shallowest first.

The part is meshed once; every triangle gets the exact outward surface normal
at its centroid.  Directions are spread over a hemisphere (d and -d give the
same parting, only cavity and core swap).  For each direction

* near-zero-draft area  = area of triangles with |n.d| < sin(low_draft)
* undercut area         = area-weighted samples whose every allowed release
                          direction is blocked (parallel ray casting, as in
                          the main analysis)

A coarse pass scores all directions; the best distinct ones are re-scored on a
finer mesh with more rays, and the top N are returned.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepTools import BRepTools
from OCP.TopLoc import TopLoc_Location

from .faces import FaceCtx
from .pull import PRESETS, describe
from .undercut import RayCaster, mesh_deflection


@dataclass
class Candidate:
    d: np.ndarray
    undercut_area: float
    low_draft_area: float
    depth: float = 0.0
    weight: float = 1.0

    @property
    def score(self) -> float:
        return self.weight * self.undercut_area + self.low_draft_area

    @property
    def label(self) -> str:
        return describe(self.d)

    def as_dict(self) -> dict:
        return {"vector": [float(x) for x in self.d], "label": self.label, "score": self.score,
                "undercut_area": self.undercut_area, "low_draft_area": self.low_draft_area,
                "depth": self.depth}


class _Mesh:
    """Triangles with exact outward normals at their centroids."""

    def __init__(self, model, lin: float):
        self.lin = lin
        BRepTools.Clean_s(model.shape)  # force this deflection even if a mesh exists
        BRepMesh_IncrementalMesh(model.shape, lin, False, 0.5, True)
        verts, tris, cents, norms = [], [], [], []
        base = 0
        for fi, face in enumerate(model.faces):
            loc = TopLoc_Location()
            tri = BRep_Tool.Triangulation_s(face, loc)
            if tri is None:
                continue
            fc = FaceCtx(fi, face)
            trsf = None if loc.IsIdentity() else loc.Transformation()
            nn = tri.NbNodes()
            uv = None
            if tri.HasUVNodes():
                uv = np.array([(tri.UVNode(i).X(), tri.UVNode(i).Y()) for i in range(1, nn + 1)])
            for i in range(1, nn + 1):
                p = tri.Node(i)
                if trsf is not None:
                    p = p.Transformed(trsf)
                verts.append((p.X(), p.Y(), p.Z()))
            for i in range(1, tri.NbTriangles() + 1):
                t = tri.Triangle(i)
                a, b, c = t.Value(1) - 1, t.Value(2) - 1, t.Value(3) - 1
                tris.append((base + a, base + b, base + c))
                n = None
                if uv is not None:
                    u, v = (uv[a] + uv[b] + uv[c]) / 3.0
                    p3, n = fc.point_normal(u, v)
                cents.append(None if n is None else p3)
                norms.append(n)
            base += nn
        self.V = np.array(verts, dtype=float).reshape(-1, 3)
        self.T = np.array(tris, dtype=np.int64).reshape(-1, 3)
        P = self.V[self.T]
        cross = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        self.A = 0.5 * np.linalg.norm(cross, axis=1)
        geo_n = cross / np.maximum(2 * self.A, 1e-300)[:, None]
        cent = P.mean(axis=1)
        N = np.empty_like(cent)
        for i, n in enumerate(norms):
            if n is None:
                N[i] = geo_n[i]  # no UV nodes / singular point: fall back to the facet normal
            else:
                N[i] = n
                if cents[i] is not None:
                    cent[i] = cents[i]
        self.C = cent
        self.N = N


def hemisphere(n: int) -> np.ndarray:
    """n roughly uniform directions with z >= 0 (Fibonacci), plus +X, +Y, +Z."""
    i = np.arange(n)
    z = (i + 0.5) / n
    phi = i * math.pi * (3.0 - math.sqrt(5.0))
    r = np.sqrt(1.0 - z * z)
    dirs = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)
    presets = np.array([PRESETS["+X"], PRESETS["+Y"], PRESETS["+Z"]], dtype=float)
    return np.vstack([presets, dirs])


def _score(mesh: _Mesh, dirs: np.ndarray, t: float, low: float, n_rays: int, seed: int = 0):
    """(undercut_area, low_draft_area, draw depth) for each direction."""
    total = float(mesh.A.sum())
    # near-zero draft and projected area, all directions at once, in chunks
    lowa = np.zeros(len(dirs))
    depth = np.zeros(len(dirs))
    for k0 in range(0, len(dirs), 64):
        f = np.abs(mesh.N @ dirs[k0:k0 + 64].T)
        lowa[k0:k0 + 64] = (mesh.A[:, None] * (f < low)).sum(axis=0)
        h = mesh.V @ dirs[k0:k0 + 64].T
        depth[k0:k0 + 64] = h.max(axis=0) - h.min(axis=0)
    # area-weighted ray origins
    rng = np.random.default_rng(seed)
    p = mesh.A / total
    idx = rng.choice(len(mesh.A), size=min(n_rays, len(mesh.A)), replace=len(mesh.A) < n_rays, p=p)
    w = total / len(idx)
    eps = 3.0 * mesh.lin
    O = mesh.C[idx] + eps * mesh.N[idx]
    Ns = mesh.N[idx]
    under = np.zeros(len(dirs))
    for k, d in enumerate(dirs):
        caster = RayCaster(mesh.V, mesh.T, d)
        up, down = caster.cast(O, eps)
        f = Ns @ d
        blocked = ((f <= -t) | up) & ((f >= t) | down)
        under[k] = w * float(blocked.sum())
    return under, lowa, depth


def _order(score: np.ndarray, depth: np.ndarray, tie: float) -> list[int]:
    """Ascending score; within `tie` of each other, shallowest draw first."""
    idx = sorted(range(len(score)), key=lambda k: score[k])
    out: list[int] = []
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and score[idx[j + 1]] - score[idx[i]] <= tie:
            j += 1
        out.extend(sorted(idx[i:j + 1], key=lambda k: depth[k]))
        i = j + 1
    return out


def _distinct(order, dirs, min_deg: float, limit: int) -> list[int]:
    out: list[int] = []
    cos_min = math.cos(math.radians(min_deg))
    for k in order:
        if all(abs(float(dirs[k] @ dirs[j])) < cos_min for j in out):
            out.append(int(k))
        if len(out) >= limit:
            break
    return out


def rank_directions(model, samples: int = 500, top: int = 5, draft_tol_deg: float = 0.5,
                    low_draft_deg: float = 1.0, rays: int = 3000, refine: int = 15,
                    min_separation_deg: float = 5.0, undercut_weight: float = 1.0
                    ) -> tuple[list[Candidate], dict]:
    t0 = time.perf_counter()
    t = math.sin(math.radians(draft_tol_deg))
    low = math.sin(math.radians(low_draft_deg))
    dirs = hemisphere(samples)

    coarse = _Mesh(model, max(2e-3 * model.diag, 2.0 * mesh_deflection(model.diag)))
    under, lowa, depth = _score(coarse, dirs, t, low, rays)
    t_coarse = time.perf_counter() - t0
    tie = 1e-4 * float(coarse.A.sum())

    order = _order(undercut_weight * under + lowa, depth, tie)
    best = _distinct(order, dirs, min_separation_deg, refine)
    best += [k for k in range(3) if k not in best]  # always re-score +X, +Y, +Z

    fine = _Mesh(model, mesh_deflection(model.diag))
    u2, l2, p2 = _score(fine, dirs[best], t, low, 2 * rays, seed=1)
    pool = [Candidate(dirs[k], float(u), float(lo), float(pa), undercut_weight)
            for k, u, lo, pa in zip(best, u2, l2, p2)]
    cands = [pool[i] for i in _order(np.array([c.score for c in pool]),
                                     np.array([c.depth for c in pool]), tie)]
    presets = {c.label: (i + 1, c) for i, c in enumerate(cands) if c.label in ("+X", "+Y", "+Z")}
    info = {
        "samples": len(dirs),
        "rays": rays,
        "coarse_triangles": int(len(coarse.T)),
        "fine_triangles": int(len(fine.T)),
        "time_s": round(time.perf_counter() - t0, 3),
        "coarse_time_s": round(t_coarse, 3),
        "undercut_weight": undercut_weight,
        "top": [c.as_dict() for c in cands[:top]],
        "presets": {k: {"rank": r, **c.as_dict()} for k, (r, c) in presets.items()},
    }
    return cands[:top], info


def table(cands: list[Candidate], info: dict | None = None) -> str:
    w = cands[0].weight if cands else 1.0
    head = "score = undercut + low-draft" if w == 1 else f"score = {w:g} x undercut + low-draft"
    lines = [f"Auto    : {head}  (areas in mm2)",
             "   #  pull                             score   undercut  low-draft    depth"]

    def row(rank, c):
        v = c.d
        lab = c.label if len(c.label) <= 3 else ""
        return (f"  {rank:>2}  {v[0]:+.4f},{v[1]:+.4f},{v[2]:+.4f} {lab:>3}  {c.score:9.1f}  "
                f"{c.undercut_area:9.1f}  {c.low_draft_area:9.1f}  {c.depth:7.2f}")

    for i, c in enumerate(cands, start=1):
        lines.append(row(i, c))
    if info:
        shown = {c.label for c in cands}
        extra = sorted((v["rank"], k) for k, v in info.get("presets", {}).items() if k not in shown)
        if extra:
            lines.append("   for reference:")
            for rank, k in extra:
                p = info["presets"][k]
                c = Candidate(np.array(p["vector"]), p["undercut_area"], p["low_draft_area"],
                              p["depth"], w)
                lines.append(row(rank, c))
    return "\n".join(lines)
