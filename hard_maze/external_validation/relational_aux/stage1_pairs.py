"""Generate fixed random-pair exact shortest-path targets from Easy-to-Hard mazes."""
import argparse, json
from collections import deque
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
NODE_PIXELS = np.array([3, 7, 11, 15, 19], dtype=np.int16)


def graph_and_endpoints(image):
    open_mask = image.max(axis=0) > 0
    adjacency = [[] for _ in range(25)]
    for i, r in enumerate(NODE_PIXELS):
        for j, c in enumerate(NODE_PIXELS):
            u = 5 * i + j
            if j < 4 and open_mask[r, c + 2]: adjacency[u].append(u + 1); adjacency[u + 1].append(u)
            if i < 4 and open_mask[r + 2, c]: adjacency[u].append(u + 5); adjacency[u + 5].append(u)
    start = goal = None
    for i, r in enumerate(NODE_PIXELS):
        for j, c in enumerate(NODE_PIXELS):
            bits = tuple(image[:, r, c].astype(int))
            if bits == (1, 0, 0): start = 5 * i + j
            elif bits == (0, 1, 0): goal = 5 * i + j
    if start is None or goal is None: raise ValueError("could not locate the two endpoint markers")
    return adjacency, start, goal


def distances(adjacency, source):
    d = np.full(25, -1, dtype=np.int16); d[source] = 0; queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in adjacency[u]:
            if d[v] < 0: d[v] = d[u] + 1; queue.append(v)
    if np.any(d < 0): raise ValueError("maze graph is disconnected")
    return d


def sample_maze(image, k, min_manhattan, rng):
    adjacency, start, goal = graph_and_endpoints(image)
    nodes = [u for u in range(25) if u not in (start, goal)]
    candidates = []
    for a, u in enumerate(nodes):
        ui, uj = divmod(u, 5)
        for v in nodes[a + 1:]:
            vi, vj = divmod(v, 5); dm = abs(ui - vi) + abs(uj - vj)
            if dm >= min_manhattan: candidates.append((u, v, dm))
    chosen = rng.choice(len(candidates), size=k, replace=False)
    cache = {}; rows = []
    for index in chosen:
        u, v, dm = candidates[index]
        if u not in cache: cache[u] = distances(adjacency, u)
        dg = int(cache[u][v]); rows.append((u, v, dg, dm, dg / dm))
    return rows, start, goal


def make_split(images, maze_indices, cfg, split_name, out):
    rng = np.random.default_rng(cfg["pair_seeds"][split_name]); rows = []
    for local_index, source_index in enumerate(maze_indices):
        pairs, start, goal = sample_maze(images[source_index], cfg["pairs_per_maze"],
                                         cfg["minimum_manhattan_distance"], rng)
        for pair_index, (u, v, dg, dm, ratio) in enumerate(pairs):
            rows.append((local_index, int(source_index), pair_index, u, v, start, goal, dg, dm,
                         dg / cfg["normalization_denominator"], ratio))
    names = ["maze_index", "source_index", "pair_index", "u", "v", "start", "goal",
             "graph_distance", "manhattan_distance", "normalized_distance", "detour_ratio"]
    arrays = {name: np.asarray([row[i] for row in rows], dtype=np.float32 if i >= 9 else np.int32)
              for i, name in enumerate(names)}
    np.savez_compressed(out / f"fixed_{split_name}_k{cfg['pairs_per_maze']}.npz", **arrays)
    dg, dm, nd, ratio = (arrays[k] for k in ("graph_distance", "manhattan_distance", "normalized_distance", "detour_ratio"))
    return {"mazes": len(maze_indices), "pairs": len(rows), "graph_distance": summary(dg),
            "normalized_distance": summary(nd), "manhattan_distance": summary(dm),
            "detour_ratio": summary(ratio), "unique_graph_distances": int(len(np.unique(dg)))}


def summary(values):
    return {"min": float(values.min()), "max": float(values.max()), "mean": float(values.mean()),
            "std": float(values.std())}


def verify(out, cfg, train_images, test_images):
    checks = {}
    files = {s: np.load(out / f"fixed_{s}_k{cfg['pairs_per_maze']}.npz") for s in ("train", "val", "test")}
    checks["endpoints_distinct"] = all(np.all(z["u"] != z["v"]) for z in files.values())
    checks["main_pair_excluded"] = all(np.all((z["u"] != z["start"]) & (z["u"] != z["goal"]) &
                                                (z["v"] != z["start"]) & (z["v"] != z["goal"])) for z in files.values())
    checks["minimum_manhattan"] = all(np.all(z["manhattan_distance"] >= cfg["minimum_manhattan_distance"]) for z in files.values())
    checks["normalization_exact"] = all(np.allclose(z["normalized_distance"], z["graph_distance"] / 9) for z in files.values())
    checks["fixed_pair_count"] = all(len(z["u"]) == cfg[f"{s}_mazes"] * cfg["pairs_per_maze"] for s, z in files.items())
    checks["train_val_maze_disjoint"] = not set(files["train"]["source_index"]).intersection(files["val"]["source_index"])
    exact = True
    for split, images in (("train", train_images), ("val", train_images), ("test", test_images)):
        z = files[split]
        for row in np.linspace(0, len(z["u"]) - 1, 100, dtype=int):
            graph, _, _ = graph_and_endpoints(images[z["source_index"][row]])
            if distances(graph, int(z["u"][row]))[int(z["v"][row])] != z["graph_distance"][row]: exact = False
    checks["bfs_targets_recomputed"] = exact
    # The raster solution has 4 pixels per room and corridor block: pixels = 4*(2*d+1).
    main_exact = True
    for index in np.linspace(0, len(train_images) - 1, 100, dtype=int):
        graph, start, goal = graph_and_endpoints(train_images[index]); d = distances(graph, start)[goal]
        solution = np.load(HERE / "data" / "easy-to-hard-data" / "maze_data_train_9" / "solutions.npy", mmap_mode="r")[index]
        if int(solution.sum()) != 4 * (2 * int(d) + 1): main_exact = False
    checks["bfs_matches_raster_solution"] = main_exact
    result = {"checks": checks, "all_ok": all(checks.values())}
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n"); return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", type=Path, default=HERE / "configs" / "stage1_aux.json")
    ap.add_argument("--out", type=Path, default=HERE.parent / "results_external" / "stage1_aux_learnability")
    args = ap.parse_args(); cfg = json.loads(args.config.read_text()); args.out.mkdir(parents=True, exist_ok=True)
    root = HERE / "data" / "easy-to-hard-data"
    train_images = np.load(root / "maze_data_train_9" / "inputs.npy", mmap_mode="r")
    test_images = np.load(root / "maze_data_test_9" / "inputs.npy", mmap_mode="r")
    order = torch.randperm(len(train_images), generator=torch.Generator().manual_seed(cfg["data_split_seed"])).numpy()
    split_indices = {"train": order[:cfg["train_mazes"]],
                     "val": order[cfg["train_mazes"]:cfg["train_mazes"] + cfg["val_mazes"]],
                     "test": np.arange(cfg["test_mazes"])}
    stats = {s: make_split(test_images if s == "test" else train_images, idx, cfg, s, args.out)
             for s, idx in split_indices.items()}
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (args.out / "pair_dataset_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    result = verify(args.out, cfg, train_images, test_images); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
