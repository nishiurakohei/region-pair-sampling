"""Balanced, reproducible non-(s,g) node-pair sampling for H3."""
from collections import deque
from pathlib import Path

import numpy as np

P3 = Path(__file__).resolve().parent
ROOT = P3.parent
N = 15
START, GOAL = 0, N * N - 1
ALLOWED = np.arange(1, N * N - 1, dtype=np.int16)


def component_labels(H, V):
    labels = np.full((N, N), -1, np.int16)
    label = 0
    for r0 in range(N):
        for c0 in range(N):
            if labels[r0, c0] >= 0:
                continue
            labels[r0, c0] = label
            q = deque([(r0, c0)])
            while q:
                r, c = q.popleft()
                for rr, cc, opened in ((r, c + 1, c + 1 < N and H[r, c]),
                                       (r, c - 1, c > 0 and H[r, c - 1]),
                                       (r + 1, c, r + 1 < N and V[r, c]),
                                       (r - 1, c, r > 0 and V[r - 1, c])):
                    if opened and labels[rr, cc] < 0:
                        labels[rr, cc] = label
                        q.append((rr, cc))
            label += 1
    return labels.ravel()


def ensure_component_cache(split):
    dest = ROOT / "results_phase3" / "pairs" / f"components_{split}.npy"
    if dest.exists():
        return np.load(dest, allow_pickle=False)
    z = np.load(ROOT / "data" / "H3" / f"{split}.npz", allow_pickle=False)
    labels = np.stack([component_labels(h, v) for h, v in zip(z["H"], z["V"])])
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.save(dest, labels)
    return labels


def manhattan(a, b):
    return abs(int(a) // N - int(b) // N) + abs(int(a) % N - int(b) % N)


def sample_pairs(labels, k, seed, d_min=5, epoch=0):
    """Return (pairs, targets, duplicate count); target order is balanced."""
    rng = np.random.default_rng(seed + 1000003 * epoch)
    n = len(labels)
    pairs = np.empty((n, k, 2), np.int16)
    targets = np.empty((n, k), np.uint8)
    duplicates = 0
    for i, lab in enumerate(labels):
        used = set()
        for j in range(k):
            target = (i + j + epoch) % 2
            for _ in range(10000):
                u, v = rng.choice(ALLOWED, 2, replace=False)
                key = (min(int(u), int(v)), max(int(u), int(v)))
                if (lab[u] == lab[v]) == bool(target) and manhattan(u, v) >= d_min and key not in used:
                    break
            else:
                raise RuntimeError(f"pair sampling failed: sample={i} target={target}")
            used.add(key)
            pairs[i, j] = u, v
            targets[i, j] = target
        duplicates += k - len(used)
    return pairs, targets, duplicates


def shortest_distance(H, V, u, v):
    ur, uc, vr, vc = int(u) // N, int(u) % N, int(v) // N, int(v) % N
    dist = np.full((N, N), -1, np.int16)
    dist[ur, uc] = 0
    q = deque([(ur, uc)])
    while q:
        r, c = q.popleft()
        if (r, c) == (vr, vc):
            return int(dist[r, c])
        for rr, cc, opened in ((r, c + 1, c + 1 < N and H[r, c]),
                               (r, c - 1, c > 0 and H[r, c - 1]),
                               (r + 1, c, r + 1 < N and V[r, c]),
                               (r - 1, c, r > 0 and V[r - 1, c])):
            if opened and dist[rr, cc] < 0:
                dist[rr, cc] = dist[r, c] + 1
                q.append((rr, cc))
    return -1


def fixed_pair_path(split, k):
    return ROOT / "results_phase3" / "pairs" / f"fixed_{split}_k{k}.npz"


def ensure_fixed_pairs(split, k, seed, d_min=5, distances=False):
    path = fixed_pair_path(split, k)
    if path.exists():
        return np.load(path, allow_pickle=False)
    labels = ensure_component_cache(split)
    pairs, targets, duplicates = sample_pairs(labels, k, seed, d_min)
    grid = np.vectorize(manhattan)(pairs[..., 0], pairs[..., 1]).astype(np.int16)
    graph = np.full((len(labels), k), -1, np.int16)
    if distances:
        z = np.load(ROOT / "data" / "H3" / f"{split}.npz", allow_pickle=False)
        for i in range(len(labels)):
            for j in range(k):
                if targets[i, j]:
                    graph[i, j] = shortest_distance(z["H"][i], z["V"][i], *pairs[i, j])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, pairs=pairs, targets=targets, grid_distance=grid,
                        graph_distance=graph, duplicates=duplicates, seed=seed, d_min=d_min)
    return np.load(path, allow_pickle=False)


def self_check():
    labels = ensure_component_cache("val")[:20]
    a, y, dup = sample_pairs(labels, 16, 7, 5)
    b, yb, _ = sample_pairs(labels, 16, 7, 5)
    assert np.array_equal(a, b) and np.array_equal(y, yb) and dup == 0
    assert np.all(a[..., 0] != a[..., 1]) and not np.any((a == START) | (a == GOAL))
    assert np.all([manhattan(u, v) >= 5 for u, v in a.reshape(-1, 2)])
    assert np.mean(y) == .5
    for i in range(len(labels)):
        assert np.all((labels[i, a[i, :, 0]] == labels[i, a[i, :, 1]]) == y[i])
    print("PAIR_SAMPLER_OK", a.shape, "positive_rate", y.mean())


if __name__ == "__main__":
    self_check()
