"""Ray casting along +/-d against a triangulation of the part.

All rays are parallel to d, so we project the mesh onto the plane normal to d,
bin triangles into a 2D grid once, and test each ray only against triangles in
its cell.  Everything is vectorised with numpy; OpenCascade is used only to mesh.
Used for the undercut check and for deciding which half can release each part
of a vertical band.
"""

from __future__ import annotations

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopAbs import TopAbs_REVERSED
from OCP.TopLoc import TopLoc_Location

from .occ import orthonormal_basis


def mesh_arrays(shape, faces, lin_defl: float, ang_defl: float = 0.35):
    """Triangulate and return (V (n,3), T (m,3), tri_face (m,)).

    Triangles are wound so their normals point out of the material (BRepMesh
    follows the surface's own orientation, so REVERSED faces are flipped)."""
    BRepMesh_IncrementalMesh(shape, lin_defl, False, ang_defl, True)
    verts, tris, owner = [], [], []
    base = 0
    for fi, face in enumerate(faces):
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is None:
            continue
        trsf = None if loc.IsIdentity() else loc.Transformation()
        flip = face.Orientation() == TopAbs_REVERSED
        nn = tri.NbNodes()
        for i in range(1, nn + 1):
            p = tri.Node(i)
            if trsf is not None:
                p = p.Transformed(trsf)
            verts.append((p.X(), p.Y(), p.Z()))
        for i in range(1, tri.NbTriangles() + 1):
            t = tri.Triangle(i)
            a, b, c = base + t.Value(1) - 1, base + t.Value(2) - 1, base + t.Value(3) - 1
            tris.append((a, c, b) if flip else (a, b, c))
            owner.append(fi)
        base += nn
    return (np.array(verts, dtype=float).reshape(-1, 3),
            np.array(tris, dtype=np.int64).reshape(-1, 3),
            np.array(owner, dtype=np.int64))


class RayCaster:
    """Answers: does a ray from point p along +d (or -d) hit the mesh?"""

    def __init__(self, V: np.ndarray, T: np.ndarray, d: np.ndarray, owner: np.ndarray | None = None,
                 max_pairs: int = 30_000_000):
        self.d = d
        self.owner = owner
        e1, e2 = orthonormal_basis(d)
        self.M = np.stack([e1, e2], axis=1)
        self.T = T
        self.H = V @ d
        P2 = V @ self.M
        A, B, C = P2[T[:, 0]], P2[T[:, 1]], P2[T[:, 2]]
        self.A, self.B, self.C = A, B, C
        # which way each triangle faces along d (outward normal . d)
        P3 = V[T]
        self.nd = np.cross(P3[:, 1] - P3[:, 0], P3[:, 2] - P3[:, 0]) @ d
        self.area2 = (B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0])
        scale = float(np.ptp(P2, axis=0).max()) if len(P2) else 1.0
        scale = scale or 1.0
        keep = np.abs(self.area2) > 1e-16 * scale * scale  # triangles parallel to d are never hit
        tid = np.nonzero(keep)[0]
        self.empty = len(tid) == 0
        if self.empty:
            return
        mn = np.minimum(np.minimum(A[tid], B[tid]), C[tid])
        mx = np.maximum(np.maximum(A[tid], B[tid]), C[tid])
        lo = mn.min(axis=0) - 1e-9 * scale
        hi = mx.max(axis=0) + 1e-9 * scale
        ext = np.maximum(hi - lo, 1e-9 * scale)
        cs = max(float(np.sqrt(ext[0] * ext[1] / len(tid))) * 1.5, float(ext.max()) / 2048)
        while True:
            nx = max(1, int(np.ceil(ext[0] / cs)))
            ny = max(1, int(np.ceil(ext[1] / cs)))
            ix0 = np.clip(((mn[:, 0] - lo[0]) / cs).astype(np.int64), 0, nx - 1)
            ix1 = np.clip(((mx[:, 0] - lo[0]) / cs).astype(np.int64), 0, nx - 1)
            iy0 = np.clip(((mn[:, 1] - lo[1]) / cs).astype(np.int64), 0, ny - 1)
            iy1 = np.clip(((mx[:, 1] - lo[1]) / cs).astype(np.int64), 0, ny - 1)
            wx = ix1 - ix0 + 1
            counts = wx * (iy1 - iy0 + 1)
            total = int(counts.sum())
            if total <= max_pairs:
                break
            cs *= 2.0
        rep = np.repeat(np.arange(len(tid)), counts)
        first = np.repeat(np.cumsum(counts) - counts, counts)
        k = np.arange(total) - first
        w = wx[rep]
        cells = (iy0[rep] + k // w) * nx + (ix0[rep] + k % w)
        order = np.argsort(cells, kind="stable")
        self.cell_tris = tid[rep[order]]
        self.cell_count = np.bincount(cells, minlength=nx * ny)
        self.cell_start = np.cumsum(self.cell_count) - self.cell_count
        self.lo, self.cs, self.nx, self.ny = lo, cs, nx, ny

    def cast(self, origins: np.ndarray, eps: float, ignore=None) -> tuple[np.ndarray, np.ndarray]:
        """(blocked_up, blocked_down) for each origin.

        Only hits where the ray *enters* material count (the triangle faces
        against the ray): a ray that is already inside - because it started a
        hair inside at a concave corner, or drifted in alongside a face tilted
        within the draft tolerance - is not blocked by the wall it leaves
        through.  Hits closer than eps, and hits on faces in `ignore`, do not
        count either."""
        origins = np.atleast_2d(origins)
        R = len(origins)
        up = np.zeros(R, dtype=bool)
        down = np.zeros(R, dtype=bool)
        if R == 0 or self.empty:
            return up, down
        O2 = origins @ self.M
        Oh = origins @ self.d
        cx = np.floor((O2[:, 0] - self.lo[0]) / self.cs).astype(np.int64)
        cy = np.floor((O2[:, 1] - self.lo[1]) / self.cs).astype(np.int64)
        valid = (cx >= 0) & (cx < self.nx) & (cy >= 0) & (cy < self.ny)
        rcell = np.where(valid, cy * self.nx + cx, 0)
        A, B, C, H, T = self.A, self.B, self.C, self.H, self.T
        batch = 4096
        for b0 in range(0, R, batch):
            rays = np.arange(b0, min(R, b0 + batch))
            rays = rays[valid[rays]]
            if len(rays) == 0:
                continue
            cnt = self.cell_count[rcell[rays]]
            if cnt.sum() == 0:
                continue
            rr = np.repeat(rays, cnt)
            off = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
            tt = self.cell_tris[np.repeat(self.cell_start[rcell[rays]], cnt) + off]
            p = O2[rr]
            a, b, c = A[tt], B[tt], C[tt]
            w0 = (b[:, 0] - p[:, 0]) * (c[:, 1] - p[:, 1]) - (b[:, 1] - p[:, 1]) * (c[:, 0] - p[:, 0])
            w1 = (c[:, 0] - p[:, 0]) * (a[:, 1] - p[:, 1]) - (c[:, 1] - p[:, 1]) * (a[:, 0] - p[:, 0])
            w2 = (a[:, 0] - p[:, 0]) * (b[:, 1] - p[:, 1]) - (a[:, 1] - p[:, 1]) * (b[:, 0] - p[:, 0])
            ar = self.area2[tt]
            s = np.sign(ar)
            tol = -1e-12 * np.abs(ar)
            inside = (w0 * s >= tol) & (w1 * s >= tol) & (w2 * s >= tol)
            if not inside.any():
                continue
            if ignore is not None and self.owner is not None:
                inside &= ~np.isin(self.owner[tt], np.fromiter(ignore, dtype=np.int64))
                if not inside.any():
                    continue
            rr, tt = rr[inside], tt[inside]
            h = (w0[inside] * H[T[tt, 0]] + w1[inside] * H[T[tt, 1]] + w2[inside] * H[T[tt, 2]]) / ar[inside]
            dh = h - Oh[rr]
            nd = self.nd[tt]
            up[rr[(dh > eps) & (nd < 0)]] = True
            down[rr[(dh < -eps) & (nd > 0)]] = True
        return up, down


def mesh_deflection(diag: float) -> float:
    return float(np.clip(2e-4 * diag, 0.005, 0.05))
