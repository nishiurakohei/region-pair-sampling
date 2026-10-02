"""Grid-graph operations used by Hard-Maze construction and verification."""
from collections import deque

import numpy as np


def neighbors(H, V, node):
    n = H.shape[0]
    r, c = node
    if c + 1 < n and H[r, c]:
        yield r, c + 1
    if c and H[r, c - 1]:
        yield r, c - 1
    if r + 1 < n and V[r, c]:
        yield r + 1, c
    if r and V[r - 1, c]:
        yield r - 1, c


def components(H, V):
    """Return integer component labels and component sizes."""
    n = H.shape[0]
    labels = np.full((n, n), -1, dtype=np.int16)
    sizes = []
    for r0 in range(n):
        for c0 in range(n):
            if labels[r0, c0] >= 0:
                continue
            label = len(sizes)
            labels[r0, c0] = label
            q = deque([(r0, c0)])
            size = 0
            while q:
                u = q.popleft()
                size += 1
                for v in neighbors(H, V, u):
                    if labels[v] < 0:
                        labels[v] = label
                        q.append(v)
            sizes.append(size)
    return labels, np.asarray(sizes, dtype=np.int16)


def shortest_path_length(H, V, start, goal):
    """Exact unweighted distance; -1 when disconnected."""
    n = H.shape[0]
    dist = np.full((n, n), -1, dtype=np.int16)
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        if u == goal:
            return int(dist[u])
        for v in neighbors(H, V, u):
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                q.append(v)
    return -1


def all_grid_edges(n):
    """Yield canonical edges as (axis, row, col, endpoint_a, endpoint_b)."""
    for r in range(n):
        for c in range(n - 1):
            yield 0, r, c, (r, c), (r, c + 1)
    for r in range(n - 1):
        for c in range(n):
            yield 1, r, c, (r, c), (r + 1, c)


def is_open(H, V, edge):
    axis, r, c = edge[:3]
    return bool(H[r, c] if axis == 0 else V[r, c])


def with_edge_open(H, V, edge):
    h, v = H.copy(), V.copy()
    axis, r, c = edge[:3]
    (h if axis == 0 else v)[r, c] = True
    return h, v


def degrees(H, V):
    n = H.shape[0]
    d = np.zeros((n, n), dtype=np.int8)
    d[:, :-1] += H
    d[:, 1:] += H
    d[:-1, :] += V
    d[1:, :] += V
    return d


def edge_distance(edge, point):
    """Minimum Manhattan distance from either endpoint to point."""
    a, b = edge[3], edge[4]
    return min(abs(a[0] - point[0]) + abs(a[1] - point[1]),
               abs(b[0] - point[0]) + abs(b[1] - point[1]))


def local_open_density(H, V, point, radius=3):
    vals = []
    for e in all_grid_edges(H.shape[0]):
        if min(max(abs(e[3][0] - point[0]), abs(e[3][1] - point[1])),
               max(abs(e[4][0] - point[0]), abs(e[4][1] - point[1]))) <= radius:
            vals.append(is_open(H, V, e))
    return float(np.mean(vals)) if vals else 0.0


def graph_stats(H, V, start, goal):
    labels, sizes = components(H, V)
    d = degrees(H, V)
    edge_count = int(H.sum() + V.sum())
    kappa = int(len(sizes))
    hist = np.bincount(d.ravel(), minlength=5)[:5]
    return {
        "edge_count": edge_count,
        "open_edge_density": edge_count / (H.size + V.size),
        "component_count": kappa,
        "beta1": edge_count - H.shape[0] ** 2 + kappa,
        "degree_histogram": hist.astype(int),
        "mean_degree": float(d.mean()),
        "dead_ends": int((d == 1).sum()),
        "junctions": int((d >= 3).sum()),
        "start_degree": int(d[start]),
        "goal_degree": int(d[goal]),
        "start_component_size": int(sizes[labels[start]]),
        "goal_component_size": int(sizes[labels[goal]]),
        "start_local_density": local_open_density(H, V, start),
        "goal_local_density": local_open_density(H, V, goal),
    }


def shortcut_features(H, V, start, goal):
    """Coarse summaries only; component sizes are excluded as label-proximal."""
    s = graph_stats(H, V, start, goal)
    return np.asarray([
        s["edge_count"], s["open_edge_density"], s["component_count"], s["beta1"],
        *s["degree_histogram"], s["mean_degree"], s["dead_ends"], s["junctions"],
        s["start_degree"], s["goal_degree"], s["start_local_density"],
        s["goal_local_density"],
    ], dtype=np.float32)
