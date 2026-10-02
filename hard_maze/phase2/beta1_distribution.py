"""Inspect and verify H3 beta1 before any Phase-2 training."""
import json
import sys
from pathlib import Path

import numpy as np

P2 = Path(__file__).resolve().parent
ROOT = P2.parent
sys.path.insert(0, str(ROOT / "src"))
from graph import components


def mutual_information_discrete(x, y):
    total = len(y)
    mi = 0.0
    for xv in np.unique(x):
        for yv in (0, 1):
            joint = np.sum((x == xv) & (y == yv)) / total
            if joint:
                px, py = np.mean(x == xv), np.mean(y == yv)
                mi += joint * np.log(joint / (px * py))
    return float(mi)


def logistic_beta1(train_b, train_y, test_b, test_y):
    mu, sd = train_b.mean(), train_b.std()
    X = np.column_stack([(train_b - mu) / sd, np.ones(len(train_b))])
    Xt = np.column_stack([(test_b - mu) / sd, np.ones(len(test_b))])
    w = np.zeros(2)
    for _ in range(500):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        w -= .1 * (X.T @ (p - train_y) / len(train_y))
    pt = 1 / (1 + np.exp(-np.clip(Xt @ w, -30, 30)))
    return float(((pt > .5) == test_y).mean())


def main():
    out, arrays = {}, {}
    for split in ("train", "val", "test"):
        z = np.load(ROOT / "data" / "H3" / f"{split}.npz", allow_pickle=False)
        b, y = z["beta1"], z["y"]
        arrays[split] = (b, y)
        idx = np.linspace(0, len(y) - 1, min(500, len(y)), dtype=int)
        errors = 0
        for i in idx:
            k = len(components(z["H"][i], z["V"][i])[1])
            errors += int(b[i] != z["H"][i].sum() + z["V"][i].sum() - 225 + k)
        counts = np.bincount(b, minlength=int(b.max()) + 1)
        out[split] = {
            "min": int(b.min()), "max": int(b.max()), "mean": float(b.mean()),
            "std": float(b.std()), "median": float(np.median(b)),
            "quantiles": {str(q): float(np.quantile(b, q)) for q in (0, .05, .25, .5, .75, .95, 1)},
            "unique_values": int(len(np.unique(b))),
            "histogram": {str(i): int(n) for i, n in enumerate(counts) if n},
            "positive_histogram": {str(i): int(np.sum(b[y == 1] == i)) for i in np.unique(b)},
            "negative_histogram": {str(i): int(np.sum(b[y == 0] == i)) for i in np.unique(b)},
            "correlation_with_reachability": float(np.corrcoef(b, y)[0, 1]),
            "mutual_information_nats": mutual_information_discrete(b, y),
            "pair_exact_match": float(np.mean(b[::2] == b[1::2])),
            "recalculation_errors": errors,
        }
    out["beta1_only_logistic_test_accuracy"] = logistic_beta1(
        arrays["train"][0], arrays["train"][1], arrays["test"][0], arrays["test"][1])
    dest = ROOT / "results_phase2" / "aux_only"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "beta1_distribution.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "train" or True}, indent=2)[:8000])


if __name__ == "__main__":
    main()
