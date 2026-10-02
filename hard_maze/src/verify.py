"""Exhaustive structural checks for generated Hard-Maze datasets."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from graph import components, shortest_path_length
from maze import reachable

ROOT = Path(__file__).resolve().parents[1]
START, GOAL = (0, 0), (14, 14)


def set_edge(H, V, edge, value):
    h, v = H.copy(), V.copy()
    axis, r, c = map(int, edge)
    (h if axis == 0 else v)[r, c] = value
    return h, v


def digest(H, V):
    return hashlib.sha256(H.tobytes() + V.tobytes()).hexdigest()


def verify_level(level, small=False):
    checks = {"splits": {}, "base_overlap": {}}
    base_sets = {}
    for split in ("train", "val", "test"):
        z = np.load(ROOT / "data" / level / f"{split}.npz", allow_pickle=False)
        H, V, y = z["H"], z["V"], z["y"]
        n = len(y)
        detail = {
            "samples": n,
            "positive_rate": float(y.mean()),
            "bfs_label_mismatches": 0,
            "positive_pivot_close_failures": 0,
            "negative_pivot_open_failures": 0,
            "negative_one_flip_failures": 0,
            "pair_edge_count_mismatches": 0,
            "pair_component_count_mismatches": 0,
            "pair_beta1_mismatches": 0,
            "pair_id_or_seed_mismatches": 0,
            "h3_path_failures": 0,
            "h3_interior_failures": 0,
        }
        for i in range(n):
            truth = int(reachable(H[i], V[i], START, GOAL))
            detail["bfs_label_mismatches"] += truth != int(y[i])
            if level == "H0":
                continue
            pivot = z["pivotal_edge"][i]
            if y[i]:
                hc, vc = set_edge(H[i], V[i], pivot, False)
                detail["positive_pivot_close_failures"] += reachable(hc, vc, START, GOAL)
                if level == "H3":
                    detail["h3_path_failures"] += shortest_path_length(H[i], V[i], START, GOAL) < 34
                    axis, r, c = map(int, pivot)
                    endpoints = ((r, c), (r, c + 1)) if axis == 0 else ((r, c), (r + 1, c))
                    ds = min(a + b for a, b in endpoints)
                    dg = min((14 - a) + (14 - b) for a, b in endpoints)
                    detail["h3_interior_failures"] += ds <= 3 or dg <= 3
            else:
                ho, vo = set_edge(H[i], V[i], pivot, True)
                fail = not reachable(ho, vo, START, GOAL)
                detail["negative_pivot_open_failures"] += fail
                detail["negative_one_flip_failures"] += fail
        if level != "H0":
            for i in range(0, n, 2):
                detail["pair_id_or_seed_mismatches"] += not (
                    z["pair_id"][i] == z["pair_id"][i + 1] and
                    z["base_seed"][i] == z["base_seed"][i + 1] and
                    y[i] == 1 and y[i + 1] == 0)
                if level in {"H2", "H3"}:
                    detail["pair_edge_count_mismatches"] += z["edge_count"][i] != z["edge_count"][i + 1]
                    detail["pair_component_count_mismatches"] += z["component_count"][i] != z["component_count"][i + 1]
                    detail["pair_beta1_mismatches"] += z["beta1"][i] != z["beta1"][i + 1]
        base_sets[split] = {digest(H[i], V[i]) for i in range(n)} if level == "H0" else {
            str(int(s)) for s in z["base_seed"][::2]}
        detail["ok"] = all(v == 0 for k, v in detail.items()
                           if k.endswith("mismatches") or k.endswith("failures"))
        checks["splits"][split] = detail
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        checks["base_overlap"][f"{a}_{b}"] = len(base_sets[a] & base_sets[b])
    checks["no_leakage_ok"] = not any(checks["base_overlap"].values())
    checks["bayes_ceiling"] = 1.0
    checks["all_ok"] = checks["no_leakage_ok"] and all(x["ok"] for x in checks["splits"].values())
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", nargs="+", default=["H0", "H1", "H2", "H3"])
    parser.add_argument("--small", action="store_true")
    args = parser.parse_args()
    out = {level: verify_level(level, args.small) for level in args.levels}
    out["all_ok"] = all(v["all_ok"] for v in out.values())
    name = "verification_small.json" if args.small else "verification.json"
    encoded = json.dumps(out, indent=2, default=lambda x: x.item())
    (ROOT / "results_hard_maze" / name).write_text(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
