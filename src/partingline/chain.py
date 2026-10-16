"""Chain parting segments into loops and build OCC wires."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from OCP.BRep import BRep_Builder
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
from OCP.TopoDS import TopoDS, TopoDS_Shape, TopoDS_Vertex

from .occ import make_compound, pnt
from .segments import Segment


@dataclass
class Loop:
    items: list[tuple[int, bool]]  # (segment index, reversed)
    nodes: list[int]               # node ids along the chain (len = items + 1)
    closed: bool
    length: float
    points: np.ndarray             # concatenated polyline
    kinds: dict = field(default_factory=dict)  # length per segment kind


@dataclass
class Chaining:
    loops: list[Loop]
    node_xyz: np.ndarray
    node_tol: np.ndarray
    junctions: list[int]   # nodes of degree > 2
    dangling: list[int]    # nodes of degree 1
    max_gap: float
    seg_nodes: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=int))


def _cluster(points: np.ndarray, tol: float) -> np.ndarray:
    """Union-find clustering of points closer than tol; returns labels."""
    n = len(points)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        d = np.linalg.norm(points[i + 1:] - points[i], axis=1)
        for j in np.nonzero(d <= tol)[0] + i + 1:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[rj] = ri
    roots = np.array([find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def _end_tangent(seg: Segment, at_start: bool) -> np.ndarray:
    """Unit direction pointing away from the node along the segment."""
    p = seg.points
    if len(p) < 2:
        return np.zeros(3)
    v = (p[1] - p[0]) if at_start else (p[-2] - p[-1])
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def chain(segments: list[Segment], gap_tol: float) -> Chaining:
    if not segments:
        return Chaining([], np.zeros((0, 3)), np.zeros(0), [], [], 0.0)
    ends = np.array([[s.start, s.end] for s in segments]).reshape(-1, 3)
    labels = _cluster(ends, gap_tol)
    n_nodes = int(labels.max()) + 1
    node_xyz = np.zeros((n_nodes, 3))
    node_tol = np.zeros(n_nodes)
    for k in range(n_nodes):
        members = ends[labels == k]
        c = members.mean(axis=0)
        node_xyz[k] = c
        node_tol[k] = float(np.linalg.norm(members - c, axis=1).max())
    seg_nodes = labels.reshape(-1, 2)  # (start node, end node) per segment

    # incidence: node -> list of (segment, at_start)
    inc: dict[int, list[tuple[int, bool]]] = {k: [] for k in range(n_nodes)}
    for si, (a, b) in enumerate(seg_nodes):
        inc[int(a)].append((si, True))
        inc[int(b)].append((si, False))

    # At nodes with more than two segment ends, pair ends by straightest
    # continuation; odd leftovers become chain ends.
    pairing: dict[tuple[int, bool], tuple[int, bool]] = {}
    junctions, dangling = [], []
    for k, ends_k in inc.items():
        deg = len(ends_k)
        if deg == 2:
            pairing[ends_k[0]] = ends_k[1]
            pairing[ends_k[1]] = ends_k[0]
            continue
        if deg == 1:
            dangling.append(k)
            continue
        junctions.append(k)
        tans = {e: _end_tangent(segments[e[0]], e[1]) for e in ends_k}
        free = list(ends_k)
        while len(free) >= 2:
            best, bi, bj = 2.0, 0, 1
            for i in range(len(free)):
                for j in range(i + 1, len(free)):
                    dot = float(tans[free[i]] @ tans[free[j]])
                    if dot < best:
                        best, bi, bj = dot, i, j
            ei, ej = free[bi], free[bj]
            pairing[ei] = ej
            pairing[ej] = ei
            free = [e for idx, e in enumerate(free) if idx not in (bi, bj)]

    used = [False] * len(segments)
    loops: list[Loop] = []

    def walk(si: int, rev: bool):
        """Follow segments from segment si (traversed reversed if rev)."""
        items, nodes = [], []
        cur, cur_rev = si, rev
        start_node = int(seg_nodes[si][1 if rev else 0])
        nodes.append(start_node)
        while True:
            used[cur] = True
            items.append((cur, cur_rev))
            exit_node = int(seg_nodes[cur][0 if cur_rev else 1])
            nodes.append(exit_node)
            # we leave through the segment's end, or its start when reversed
            nxt = pairing.get((cur, cur_rev))
            if nxt is None:
                return items, nodes, False
            nseg, n_at_start = nxt
            if nseg == si and (n_at_start != rev):
                return items, nodes, True
            if used[nseg]:
                return items, nodes, False
            # entering nseg at its start -> traverse forward
            cur, cur_rev = nseg, not n_at_start

    # Open chains first: start at unpaired segment ends.
    for si in range(len(segments)):
        for at_start in (True, False):
            if used[si] or (si, at_start) in pairing:
                continue
            items, nodes, closed = walk(si, rev=not at_start)
            loops.append(_make_loop(segments, items, nodes, closed))
    for si in range(len(segments)):
        if not used[si]:
            items, nodes, closed = walk(si, rev=False)
            loops.append(_make_loop(segments, items, nodes, closed))

    max_gap = float(node_tol.max()) * 2 if len(node_tol) else 0.0
    return Chaining(loops, node_xyz, node_tol, junctions, dangling, max_gap, seg_nodes)


def _make_loop(segments, items, nodes, closed) -> Loop:
    pts = []
    kinds: dict[str, float] = {}
    length = 0.0
    for si, rev in items:
        s = segments[si]
        p = s.points[::-1] if rev else s.points
        pts.append(p if not pts else p[1:])
        length += s.length
        kinds[s.kind] = kinds.get(s.kind, 0.0) + s.length
    points = np.vstack(pts) if pts else np.zeros((0, 3))
    # a single segment whose two ends meet is closed too
    if not closed and len(items) == 1 and nodes[0] == nodes[-1]:
        closed = True
    return Loop(items, nodes, closed, length, points, kinds)


def build_wires(segments: list[Segment], ch: Chaining) -> list[TopoDS_Shape]:
    """One wire per loop, sharing vertices at chain nodes (gap-tolerant)."""
    builder = BRep_Builder()
    vertices: dict[int, TopoDS_Vertex] = {}

    def vertex(k: int) -> TopoDS_Vertex:
        if k not in vertices:
            v = TopoDS_Vertex()
            builder.MakeVertex(v, pnt(ch.node_xyz[k]), max(1e-7, 1.01 * ch.node_tol[k] + 1e-7))
            vertices[k] = v
        return vertices[k]

    wires = []
    for loop in ch.loops:
        edges = []
        for (si, rev), n0, n1 in zip(loop.items, loop.nodes[:-1], loop.nodes[1:]):
            s = segments[si]
            va, vb = (vertex(n1), vertex(n0)) if rev else (vertex(n0), vertex(n1))
            me = BRepBuilderAPI_MakeEdge(s.curve, va, vb, s.t0, s.t1)
            if not me.IsDone():
                me = BRepBuilderAPI_MakeEdge(s.curve, s.t0, s.t1)
            if not me.IsDone():
                continue
            e = me.Edge()
            edges.append(TopoDS.Edge_s(e.Reversed()) if rev else e)
        if not edges:
            continue
        mw = BRepBuilderAPI_MakeWire()
        for e in edges:
            mw.Add(e)
        wires.append(mw.Wire() if mw.IsDone() else make_compound(edges))
    return wires
