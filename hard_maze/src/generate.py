"""Targeted construction of H1--H3 counterfactual Hard-Maze pairs."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from graph import (all_grid_edges, components, degrees, edge_distance, graph_stats,
                   is_open, shortest_path_length, with_edge_open)
from maze import default_endpoints, reachable, sample_maze

ROOT = Path(__file__).resolve().parents[1]


def load_config():
    return json.loads((ROOT / "config.json").read_text())


def closed_cross_edges(H, V, labels, comp_a, comp_b):
    wanted = {int(comp_a), int(comp_b)}
    return [e for e in all_grid_edges(H.shape[0])
            if not is_open(H, V, e) and {int(labels[e[3]]), int(labels[e[4]])} == wanted]


def control_candidates(H, V, labels, comp_s, comp_g, interior_radius, start, goal):
    out = []
    forbidden = {int(comp_s), int(comp_g)}
    for e in all_grid_edges(H.shape[0]):
        if is_open(H, V, e):
            continue
        a, b = int(labels[e[3]]), int(labels[e[4]])
        if a == b or {a, b} == forbidden:
            continue
        if interior_radius is not None and (edge_distance(e, start) <= interior_radius or
                                            edge_distance(e, goal) <= interior_radius):
            continue
        out.append(e)
    return out


def choose_control(candidates, pivot, deg, rng):
    pd = sorted((int(deg[pivot[3]]), int(deg[pivot[4]])))
    scored = []
    for e in candidates:
        ed = sorted((int(deg[e[3]]), int(deg[e[4]])))
        degree_gap = abs(pd[0] - ed[0]) + abs(pd[1] - ed[1])
        boundary_gap = abs(min(e[3] + e[4]) - min(pivot[3] + pivot[4]))
        scored.append((degree_gap, boundary_gap, rng.random(), e))
    return min(scored, key=lambda x: x[:3])[3]


def find_pair(H, V, level, rng, start, goal, min_path, interior_radius):
    labels, _ = components(H, V)
    cs, cg = labels[start], labels[goal]
    if cs == cg:
        return None, "already_reachable"
    pivots = closed_cross_edges(H, V, labels, cs, cg)
    if not pivots:
        return None, "no_pivot"
    if level == "H3":
        pivots = [e for e in pivots if edge_distance(e, start) > interior_radius and
                  edge_distance(e, goal) > interior_radius]
        if not pivots:
            return None, "pivot_not_interior"
    viable = []
    for e in pivots:
        hp, vp = with_edge_open(H, V, e)
        dist = shortest_path_length(hp, vp, start, goal)
        if level != "H3" or dist >= min_path:
            viable.append((e, hp, vp, dist))
    if not viable:
        return None, "path_too_short"
    pivot, hp, vp, path_length = viable[int(rng.integers(len(viable)))]
    if level == "H1":
        return (hp, vp, H.copy(), V.copy(), pivot, None, path_length), "accepted"
    ctrls = control_candidates(H, V, labels, cs, cg,
                               interior_radius if level == "H3" else None, start, goal)
    if not ctrls:
        return None, "no_control"
    ctrl = choose_control(ctrls, pivot, degrees(H, V), rng)
    hn, vn = with_edge_open(H, V, ctrl)
    if reachable(hn, vn, start, goal):
        raise AssertionError("control edge changed reachability")
    return (hp, vp, hn, vn, pivot, ctrl, path_length), "accepted"


def edge_array(edge):
    return np.asarray(edge[:3] if edge is not None else (-1, -1, -1), dtype=np.int16)


def append_sample(store, H, V, label, pair_id, sample_id, base_seed,
                  pivot, control, path_length, level, start, goal):
    stats = graph_stats(H, V, start, goal)
    store["H"].append(H)
    store["V"].append(V)
    store["y"].append(label)
    store["sample_id"].append(sample_id)
    store["pair_id"].append(pair_id)
    store["base_seed"].append(base_seed)
    store["edge_count"].append(stats["edge_count"])
    store["component_count"].append(stats["component_count"])
    store["beta1"].append(stats["beta1"])
    store["start_degree"].append(stats["start_degree"])
    store["goal_degree"].append(stats["goal_degree"])
    store["shortest_path_length"].append(path_length if label else -1)
    store["pivotal_edge"].append(edge_array(pivot))
    store["control_edge"].append(edge_array(control))
    store["hardness_level"].append(int(level[1]))


def empty_store():
    keys = ["H", "V", "y", "sample_id", "pair_id", "base_seed", "edge_count",
            "component_count", "beta1", "start_degree", "goal_degree",
            "shortest_path_length", "pivotal_edge", "control_edge", "hardness_level"]
    return {k: [] for k in keys}


def save_store(path, store):
    arrays = {}
    for k, v in store.items():
        if k in {"H", "V"}:
            arrays[k] = np.asarray(v, dtype=bool)
        elif k in {"pivotal_edge", "control_edge"}:
            arrays[k] = np.asarray(v, dtype=np.int16)
        else:
            arrays[k] = np.asarray(v, dtype=np.int64)
    np.savez_compressed(path, **arrays)


def graph_digest(H, V):
    return hashlib.sha256(H.tobytes() + V.tobytes()).hexdigest()


def generate_split(level, split, n_samples, seed, cfg, seen):
    assert n_samples % 2 == 0
    n_pairs = n_samples // 2
    rng = np.random.default_rng(seed)
    n, p = cfg["N"], cfg["p"]
    start, goal = tuple(cfg["start"]), tuple(cfg["goal"])
    counts = {"raw_generated": 0, "candidate_near_miss": 0, "valid_pivotal_edges": 0,
              "valid_control_edges": 0, "accepted_pairs": 0}
    rejects = {}
    store = empty_store()
    t0 = time.time()
    while counts["accepted_pairs"] < n_pairs:
        counts["raw_generated"] += 1
        base_seed = int(rng.integers(0, np.iinfo(np.int64).max))
        brng = np.random.default_rng(base_seed)
        H, V = sample_maze(n, p, brng)
        digest = graph_digest(H, V)
        if digest in seen:
            rejects["duplicate_base"] = rejects.get("duplicate_base", 0) + 1
            continue
        labels, _ = components(H, V)
        if labels[start] != labels[goal]:
            pivots = closed_cross_edges(H, V, labels, labels[start], labels[goal])
            if pivots:
                counts["candidate_near_miss"] += 1
                counts["valid_pivotal_edges"] += len(pivots)
        pair, reason = find_pair(H, V, level, rng, start, goal,
                                 cfg["h3_min_path"], cfg["h3_interior_radius"])
        if pair is None:
            rejects[reason] = rejects.get(reason, 0) + 1
            continue
        hp, vp, hn, vn, pivot, ctrl, path_length = pair
        if ctrl is not None:
            counts["valid_control_edges"] += 1
        seen.add(digest)
        pair_id = counts["accepted_pairs"]
        sample0 = 2 * pair_id
        append_sample(store, hp, vp, 1, pair_id, sample0, base_seed, pivot, ctrl,
                      path_length, level, start, goal)
        append_sample(store, hn, vn, 0, pair_id, sample0 + 1, base_seed, pivot, ctrl,
                      path_length, level, start, goal)
        counts["accepted_pairs"] += 1
        if counts["accepted_pairs"] % 500 == 0:
            print(f"{level}/{split}: {counts['accepted_pairs']}/{n_pairs} pairs "
                  f"from {counts['raw_generated']} raw", flush=True)
    elapsed = time.time() - t0
    counts.update({
        "accepted_samples": n_samples,
        "acceptance_rate": n_pairs / counts["raw_generated"],
        "generation_seconds": elapsed,
        "reject_reasons": rejects,
    })
    out_dir = ROOT / "data" / level
    out_dir.mkdir(parents=True, exist_ok=True)
    save_store(out_dir / f"{split}.npz", store)
    return counts


def prepare_h0(cfg):
    source = Path(cfg["phase1_source"]) / "data"
    out_dir = ROOT / "data" / "H0"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ("train", "val", "test"):
        z = np.load(source / f"{split}.npz", allow_pickle=False)
        H, V, y = z["H"], z["V"], z["y"]
        np.savez_compressed(out_dir / f"{split}.npz", H=H, V=V, y=y,
                            sample_id=np.arange(len(y)), pair_id=np.full(len(y), -1),
                            base_seed=np.full(len(y), -1), hardness_level=np.zeros(len(y), dtype=np.int8))
        summary[split] = {"samples": int(len(y)), "positive_rate": float(y.mean())}
    return summary


def pilot(raw_per_p, cfg):
    out = {}
    start, goal = tuple(cfg["start"]), tuple(cfg["goal"])
    for p in (0.60, 0.62, 0.64, 0.66):
        rng = np.random.default_rng(9000 + int(p * 100))
        counts = {"raw": raw_per_p, "disconnected": 0, "near_miss": 0,
                  "matched": 0, "h3": 0}
        lengths, distances = [], []
        for _ in range(raw_per_p):
            H, V = sample_maze(cfg["N"], p, rng)
            labels, _ = components(H, V)
            cs, cg = labels[start], labels[goal]
            if cs == cg:
                continue
            counts["disconnected"] += 1
            pivots = closed_cross_edges(H, V, labels, cs, cg)
            if not pivots:
                continue
            counts["near_miss"] += 1
            ctrls = control_candidates(H, V, labels, cs, cg, None, start, goal)
            if ctrls:
                counts["matched"] += 1
            for e in pivots:
                hp, vp = with_edge_open(H, V, e)
                lengths.append(shortest_path_length(hp, vp, start, goal))
                distances.append(min(edge_distance(e, start), edge_distance(e, goal)))
                if (ctrls and lengths[-1] >= cfg["h3_min_path"] and
                        distances[-1] > cfg["h3_interior_radius"]):
                    counts["h3"] += 1
                    break
        out[str(p)] = {**counts,
                       "near_miss_rate": counts["near_miss"] / raw_per_p,
                       "matched_rate": counts["matched"] / raw_per_p,
                       "h3_rate": counts["h3"] / raw_per_p,
                       "path_quantiles": np.quantile(lengths, [0.5, 0.7, 0.8, 0.9]).tolist() if lengths else [],
                       "pivot_distance_quantiles": np.quantile(distances, [0.5, 0.7, 0.8, 0.9]).tolist() if distances else []}
        print(p, out[str(p)], flush=True)
    (ROOT / "results_hard_maze").mkdir(exist_ok=True)
    (ROOT / "results_hard_maze" / "pilot.json").write_text(json.dumps(out, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=int, metavar="RAW_PER_P")
    parser.add_argument("--level", choices=["H1", "H2", "H3"])
    parser.add_argument("--small", action="store_true", help="generate 200/50/100 samples")
    args = parser.parse_args()
    cfg = load_config()
    if args.pilot:
        pilot(args.pilot, cfg)
        return
    results = {"config": cfg, "H0": prepare_h0(cfg)}
    levels = [args.level] if args.level else ["H1", "H2", "H3"]
    sizes = {"train": 200, "val": 50, "test": 100} if args.small else cfg["sizes"]
    level_offset = {"H1": 100000, "H2": 200000, "H3": 300000}
    for level in levels:
        seen = set()
        results[level] = {}
        for split in ("train", "val", "test"):
            seed = cfg["split_seeds"][split] + level_offset[level]
            results[level][split] = generate_split(level, split, sizes[split], seed, cfg, seen)
    path = ROOT / "results_hard_maze" / ("generation_stats_small.json" if args.small else "generation_stats.json")
    path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
