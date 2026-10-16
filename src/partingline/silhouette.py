"""Smooth silhouettes inside mixed faces.

The silhouette of a face for pull direction d is the zero set of
f(u, v) = n(u, v) . d.  We find it numerically but to full precision:

1. marching squares on a UV grid sized to the surface type / knot spans,
2. every crossing on a grid edge is solved to |f| < 1e-12 (Illinois method),
3. polylines are densified by Newton-projecting chord midpoints onto f = 0
   until the 3D sagitta is below `chord_tol`,
4. pieces are clipped to the trimmed face (points ON the boundary count as
   outside: silhouettes that run exactly along an edge are the edge pass's job).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .faces import FaceCtx


@dataclass
class SilPiece:
    face: int
    uv: np.ndarray   # (N, 2)
    xyz: np.ndarray  # (N, 3)
    closed: bool


def _root(fc: FaceCtx, a: np.ndarray, b: np.ndarray, fa: float, fb: float) -> np.ndarray:
    """Point on segment a-b where f changes sign (f(a) > 0 xor f(b) > 0)."""
    lo, hi, flo, fhi = 0.0, 1.0, fa, fb
    side = 0
    x = 0.5
    for _ in range(80):
        if fhi != flo:
            x = lo - flo * (hi - lo) / (fhi - flo)
        if not (lo < x < hi):
            x = 0.5 * (lo + hi)
        p = a + x * (b - a)
        fx = fc.f_safe(p[0], p[1])
        if fx != fx:
            x = 0.5 * (lo + hi)
            p = a + x * (b - a)
            fx = fc.f_safe(p[0], p[1])
            if fx != fx:
                break
        if abs(fx) < 1e-13 or hi - lo < 1e-13:
            break
        if (fx > 0) == (flo > 0):
            lo, flo = x, fx
            if side == -1:
                fhi *= 0.5
            side = -1
        else:
            hi, fhi = x, fx
            if side == 1:
                flo *= 0.5
            side = 1
    return a + x * (b - a)


def _project(fc: FaceCtx, uv: np.ndarray, max_step: np.ndarray):
    """Newton-project a UV point onto f = 0 along the gradient.

    Returns (uv, ok); ok is False when it did not converge or wandered more
    than two grid cells (e.g. near a crease where f jumps instead of crossing 0).
    """
    u0, v0 = float(uv[0]), float(uv[1])
    u, v = u0, v0
    hu, hv = 1e-7 * fc.du, 1e-7 * fc.dv
    f0 = fc.f_safe(u, v)
    for _ in range(12):
        if f0 != f0 or abs(f0) < 1e-13:
            break
        fu = (fc.f_safe(u + hu, v) - fc.f_safe(u - hu, v)) / (2 * hu)
        fv = (fc.f_safe(u, v + hv) - fc.f_safe(u, v - hv)) / (2 * hv)
        g2 = fu * fu + fv * fv
        if not g2 > 0:
            break
        du, dv = -f0 * fu / g2, -f0 * fv / g2
        s = max(abs(du) / max_step[0], abs(dv) / max_step[1], 1.0)
        u += du / s
        v += dv / s
        f0 = fc.f_safe(u, v)
    ok = (f0 == f0 and abs(f0) < 1e-9
          and abs(u - u0) <= 2 * max_step[0] and abs(v - v0) <= 2 * max_step[1])
    return np.array([u, v]), ok


def _march(fc: FaceCtx, us: np.ndarray, vs: np.ndarray, F: np.ndarray):
    """Marching squares -> list of (key list, closed) polylines.

    Keys identify grid edges: ("u", i, j) is the edge from node (i, j) to
    (i+1, j); ("v", i, j) goes from (i, j) to (i, j+1).
    """
    pos = F > 0
    valid = np.isfinite(F)
    segs: list[tuple] = []
    nu, nv = F.shape
    for i in range(nu - 1):
        for j in range(nv - 1):
            if not (valid[i, j] and valid[i + 1, j] and valid[i + 1, j + 1] and valid[i, j + 1]):
                continue
            a, b, c, d = pos[i, j], pos[i + 1, j], pos[i + 1, j + 1], pos[i, j + 1]
            case = a | (b << 1) | (c << 2) | (d << 3)
            if case == 0 or case == 15:
                continue
            e0, e1, e2, e3 = ("u", i, j), ("v", i + 1, j), ("u", i, j + 1), ("v", i, j)
            if case in (5, 10):
                uc, vc = 0.5 * (us[i] + us[i + 1]), 0.5 * (vs[j] + vs[j + 1])
                centre = fc.f_safe(uc, vc) > 0
                if case == 5:  # a and c positive
                    pairs = [(e0, e1), (e2, e3)] if centre else [(e0, e3), (e1, e2)]
                else:  # b and d positive
                    pairs = [(e0, e3), (e1, e2)] if centre else [(e0, e1), (e2, e3)]
                segs.extend(pairs)
                continue
            crossing = []
            if a != b:
                crossing.append(e0)
            if b != c:
                crossing.append(e1)
            if d != c:
                crossing.append(e2)
            if a != d:
                crossing.append(e3)
            if len(crossing) == 2:
                segs.append((crossing[0], crossing[1]))

    adj: dict = {}
    for si, (k1, k2) in enumerate(segs):
        adj.setdefault(k1, []).append(si)
        adj.setdefault(k2, []).append(si)
    used = [False] * len(segs)
    lines = []
    for si in range(len(segs)):
        if used[si]:
            continue
        used[si] = True
        k1, k2 = segs[si]
        chain = [k1, k2]
        closed = False
        for direction in (0, 1):
            while True:
                tail = chain[-1] if direction == 0 else chain[0]
                nxt = None
                for sj in adj.get(tail, []):
                    if not used[sj]:
                        nxt = sj
                        break
                if nxt is None:
                    break
                used[nxt] = True
                a2, b2 = segs[nxt]
                other = b2 if a2 == tail else a2
                if direction == 0:
                    chain.append(other)
                else:
                    chain.insert(0, other)
            if len(chain) > 2 and chain[0] == chain[-1]:
                closed = True
                break
        lines.append((chain, closed))
    return lines


def face_silhouettes(fc: FaceCtx, chord_tol: float, max_len: float = float("inf")) -> list[SilPiece]:
    """Silhouette pieces of one face.  Polyline segments are kept shorter than
    `max_len` (3D) as well as within `chord_tol`, so that clipping against the
    trimmed face also notices holes a straight silhouette runs across."""
    nu, nv = fc.silhouette_grid()
    us = np.linspace(fc.u0, fc.u1, nu)
    vs = np.linspace(fc.v0, fc.v1, nv)
    # border nodes a hair inside: exactly on a periodic seam the surface returns
    # the other side's normal, which looks like a jump instead of a zero
    us[0] += 1e-7 * fc.du
    us[-1] -= 1e-7 * fc.du
    vs[0] += 1e-7 * fc.dv
    vs[-1] -= 1e-7 * fc.dv
    F = np.array([[fc.f_safe(float(u), float(v)) for v in vs] for u in us])
    lines = _march(fc, us, vs, F)
    if not lines:
        return []
    cell = np.array([fc.du / (nu - 1), fc.dv / (nv - 1)])

    cache: dict = {}

    def key_uv(k) -> np.ndarray:
        if k in cache:
            return cache[k]
        kind, i, j = k
        a = np.array([us[i], vs[j]])
        if kind == "u":
            b, fa, fb = np.array([us[i + 1], vs[j]]), F[i, j], F[i + 1, j]
        else:
            b, fa, fb = np.array([us[i], vs[j + 1]]), F[i, j], F[i, j + 1]
        p = _root(fc, a, b, fa, fb)
        fp = fc.f_safe(p[0], p[1])
        # a sign change without a zero is a crease (f jumps), not a silhouette
        cache[k] = p if fp == fp and abs(fp) < 1e-7 else None
        return cache[k]

    pieces: list[SilPiece] = []
    for keys, closed in lines:
        if closed:
            keys = keys[:-1]
        uvs = [key_uv(k) for k in keys]
        bad = [i for i, p in enumerate(uvs) if p is None]
        if bad:
            if closed:  # start right after a bad crossing, then treat as open
                r = bad[0] + 1
                uvs = uvs[r:] + uvs[:r]
                closed = False
            runs, cur = [], []
            for p in uvs:
                if p is None:
                    runs.append(cur)
                    cur = []
                else:
                    cur.append(p)
            runs.append(cur)
        else:
            runs = [uvs]
        for run in runs:
            if len(run) < 2:
                continue
            uv, xyz = _densify(fc, run, closed, chord_tol, cell, max_len)
            for piece_uv, piece_xyz, piece_closed in _clip(fc, uv, xyz, closed):
                pieces.append(SilPiece(fc.index, piece_uv, piece_xyz, piece_closed))
    return pieces


def _densify(fc: FaceCtx, uv: list, closed: bool, chord_tol: float, cell: np.ndarray,
             max_len: float = float("inf")):
    pts = [fc.point(p[0], p[1]) for p in uv]
    out_uv, out_xyz = [uv[0]], [pts[0]]
    n = len(uv)
    last = n if closed else n - 1
    for k in range(last):
        a_uv, b_uv = uv[k], uv[(k + 1) % n]
        a_p, b_p = pts[k], pts[(k + 1) % n]
        stack = [(a_uv, a_p, b_uv, b_p, 0)]
        # depth-first so the output stays ordered
        seq_uv, seq_p = [], []
        while stack:
            au, ap, bu, bp, depth = stack.pop()
            mid, ok = _project(fc, 0.5 * (au + bu), cell)
            mp = fc.point(mid[0], mid[1]) if ok else None
            if ok and depth < 12 and (np.linalg.norm(mp - 0.5 * (ap + bp)) > chord_tol
                                      or np.linalg.norm(bp - ap) > max_len):
                stack.append((mid, mp, bu, bp, depth + 1))
                stack.append((au, ap, mid, mp, depth + 1))
            else:
                seq_uv.append(bu)
                seq_p.append(bp)
        out_uv.extend(seq_uv)
        out_xyz.extend(seq_p)
    if closed:
        # last appended point equals the first one
        out_uv.pop()
        out_xyz.pop()
    return np.array(out_uv), np.array(out_xyz)


def _clip(fc: FaceCtx, uv: np.ndarray, xyz: np.ndarray, closed: bool):
    """Split a polyline into the parts inside the trimmed face."""
    inside = np.array([fc.inside(p[0], p[1]) for p in uv])
    if inside.all():
        yield uv, xyz, closed
        return
    if not inside.any():
        return
    n = len(uv)
    if closed:
        # rotate so we start on an outside point
        k0 = int(np.argmin(inside))
        uv, xyz, inside = np.roll(uv, -k0, 0), np.roll(xyz, -k0, 0), np.roll(inside, -k0)
        uv = np.vstack([uv, uv[:1]])
        xyz = np.vstack([xyz, xyz[:1]])
        inside = np.append(inside, inside[0])
        n += 1
    k = 0
    while k < n:
        if not inside[k]:
            k += 1
            continue
        start = k
        while k < n and inside[k]:
            k += 1
        stop = k  # exclusive
        seg_uv = list(uv[start:stop])
        seg_xyz = list(xyz[start:stop])
        if start > 0:
            b = _boundary(fc, uv[start], uv[start - 1])
            seg_uv.insert(0, b)
            seg_xyz.insert(0, fc.point(b[0], b[1]))
        if stop < n:
            b = _boundary(fc, uv[stop - 1], uv[stop])
            seg_uv.append(b)
            seg_xyz.append(fc.point(b[0], b[1]))
        if len(seg_uv) >= 2:
            yield np.array(seg_uv), np.array(seg_xyz), False


def _boundary(fc: FaceCtx, p_in: np.ndarray, p_out: np.ndarray) -> np.ndarray:
    lo, hi = p_in.copy(), p_out.copy()
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if fc.inside(mid[0], mid[1]):
            lo = mid
        else:
            hi = mid
        if np.abs(hi - lo).max() < 1e-12 * max(fc.du, fc.dv):
            break
    return lo


def polyline_length(xyz: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()) if len(xyz) > 1 else 0.0


__all__ = ["SilPiece", "face_silhouettes", "polyline_length"]
