"""Aggregate neural metrics, including matched-pair accuracy and probability gaps."""
import json
from pathlib import Path

import numpy as np

from dataset import MazeData

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_hard_maze"
LEVELS = ["H0", "H1", "H2", "H3"]
MODELS = ["cnn", "resnet"]
SEEDS = [1, 2, 3]


def auroc(y, p):
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    sorted_p = p[order]
    i, rank = 0, 1
    while i < len(p):
        j = i + 1
        while j < len(p) and sorted_p[j] == sorted_p[i]:
            j += 1
        ranks[order[i:j]] = (rank + rank + j - i - 1) / 2
        rank += j - i
        i = j
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(y, p, bins=15):
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)) - 1, 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def metrics(y, p, pair_id):
    p = np.clip(p.astype(float), 1e-7, 1 - 1e-7)
    pred = p > 0.5
    out = {
        "accuracy": float((pred == y).mean()),
        "auroc": auroc(y, p),
        "bce": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": ece(y, p),
        "confusion_matrix": [[int(((y == 0) & ~pred).sum()), int(((y == 0) & pred).sum())],
                             [int(((y == 1) & ~pred).sum()), int(((y == 1) & pred).sum())]],
    }
    if np.all(pair_id >= 0):
        pos, neg = np.arange(0, len(y), 2), np.arange(1, len(y), 2)
        out.update({
            "pair_accuracy": float(((pred[pos] == 1) & (pred[neg] == 0)).mean()),
            "pair_probability_gap_mean": float(np.mean(np.abs(p[pos] - p[neg]))),
            "pair_probability_gap_median": float(np.median(np.abs(p[pos] - p[neg]))),
            "pair_ranking_accuracy": float(np.mean(p[pos] > p[neg])),
        })
    return out


def aggregate(items):
    scalar_keys = [k for k, v in items[0].items() if isinstance(v, (int, float))]
    return {k: {"mean": float(np.mean([x[k] for x in items])),
                "std": float(np.std([x[k] for x in items]))} for k in scalar_keys}


def main():
    out = {"meta": {"levels": LEVELS, "models": MODELS, "seeds": SEEDS,
                    "bayes_accuracy": 1.0}, "levels": {}}
    for level in LEVELS:
        test = MazeData(level, "test")
        out["levels"][level] = {}
        for model in MODELS:
            per_seed = []
            for seed in SEEDS:
                path = RESULTS / f"pred_{level}_{model}_s{seed}.npz"
                if not path.exists():
                    raise FileNotFoundError(path)
                z = np.load(path, allow_pickle=False)
                per_seed.append(metrics(test.y.astype(int), z["prob"], test.arrays["pair_id"]))
            out["levels"][level][model] = {"per_seed": per_seed, "aggregate": aggregate(per_seed)}
            a = out["levels"][level][model]["aggregate"]["accuracy"]
            print(level, model, f"accuracy={a['mean']:.4f}±{a['std']:.4f}", flush=True)
    (RESULTS / "metrics.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
