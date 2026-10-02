"""Create fixed validation/test pair sets and sampling diagnostics."""
import json
from pathlib import Path

import numpy as np

from pairs import ensure_component_cache, ensure_fixed_pairs

P3 = Path(__file__).resolve().parent
ROOT = P3.parent


def summary(z):
    y, gd, graph = z["targets"], z["grid_distance"], z["graph_distance"]
    def stats(x):
        return {"min": int(x.min()), "max": int(x.max()), "mean": float(x.mean()),
                "median": float(np.median(x)),
                "quantiles": {str(q): float(np.quantile(x, q)) for q in (.1, .25, .5, .75, .9)}}
    return {"samples": int(len(y)), "pairs_per_sample": int(y.shape[1]),
            "positive_rate": float(y.mean()), "duplicates": int(z["duplicates"]),
            "grid_connected": stats(gd[y == 1]), "grid_disconnected": stats(gd[y == 0]),
            "graph_connected": stats(graph[y == 1]),
            "near_neighbor_rate_grid_le_6": float(np.mean(gd <= 6)),
            "start_or_goal_violations": int(np.sum((z["pairs"] == 0) | (z["pairs"] == 224)))}


def main():
    cfg = json.loads((P3 / "config.json").read_text())
    out = {"config": cfg, "splits": {}}
    for split, offset in (("val", 1), ("test", 2)):
        ensure_component_cache(split)
        z = ensure_fixed_pairs(split, max(cfg["k_values"]), cfg["pair_seed"] + offset,
                               cfg["d_min"], distances=True)
        out["splits"][split] = summary(z)
    ensure_component_cache("train")
    dest = ROOT / "results_phase3" / "aux_only"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "config.json").write_text(json.dumps(cfg, indent=2))
    (dest / "pair_sampling_stats.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
