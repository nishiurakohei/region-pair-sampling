"""Dataset summaries plus scalar-summary and local-patch shortcut baselines."""
import json
from pathlib import Path

import numpy as np

from dataset import MazeData, rasterize_batch
from graph import graph_stats, shortcut_features

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_hard_maze"
LEVELS = ["H0", "H1", "H2", "H3"]
SPLITS = ["train", "val", "test"]
START, GOAL = (0, 0), (14, 14)


def summarize(values, labels):
    out = {}
    for cls, name in ((1, "positive"), (0, "negative")):
        x = values[labels == cls]
        out[name] = {"mean": float(np.mean(x)), "std": float(np.std(x)),
                     "min": float(np.min(x)), "max": float(np.max(x))}
    return out


def collect(level, split):
    data = MazeData(level, split)
    rows, shortcut = [], []
    for H, V in zip(data.H, data.V):
        s = graph_stats(H, V, START, GOAL)
        rows.append(s)
        shortcut.append(shortcut_features(H, V, START, GOAL))
    return data, rows, np.asarray(shortcut)


def logistic_predictions(Xtr, ytr, Xva, yva, Xte, steps=300):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-6] = 1.0
    tr, va, te = [(x - mu) / sd for x in (Xtr, Xva, Xte)]
    tr = np.column_stack([tr, np.ones(len(tr))])
    va = np.column_stack([va, np.ones(len(va))])
    te = np.column_stack([te, np.ones(len(te))])
    w = np.zeros(tr.shape[1], dtype=np.float64)
    best, best_loss, stale = w.copy(), np.inf, 0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-np.clip(tr @ w, -30, 30)))
        grad = tr.T @ (p - ytr) / len(ytr)
        grad[:-1] += 1e-3 * w[:-1]
        w -= 0.08 * grad
        pv = 1 / (1 + np.exp(-np.clip(va @ w, -30, 30)))
        loss = float(-np.mean(yva * np.log(pv + 1e-8) + (1 - yva) * np.log(1 - pv + 1e-8)))
        if loss < best_loss - 1e-5:
            best, best_loss, stale = w.copy(), loss, 0
        else:
            stale += 1
            if stale >= 30:
                break
    return 1 / (1 + np.exp(-np.clip(te @ best, -30, 30)))


def local_patch_features(data, radius_cells=4):
    X = rasterize_batch(data.H, data.V)[:, 1:3]
    width = 2 * radius_cells + 1
    a = X[:, :, :width, :width]
    b = X[:, :, -width:, -width:]
    return np.concatenate([a.reshape(len(X), -1), b.reshape(len(X), -1)], axis=1)


def classification(y, p):
    pred = p > 0.5
    return {
        "accuracy": float((pred == y).mean()),
        "bce": float(-np.mean(y * np.log(p + 1e-8) + (1 - y) * np.log(1 - p + 1e-8))),
        "confusion_matrix": [[int(((y == 0) & ~pred).sum()), int(((y == 0) & pred).sum())],
                             [int(((y == 1) & ~pred).sum()), int(((y == 1) & pred).sum())]],
    }


def main():
    dataset_stats, baselines = {}, {}
    for level in LEVELS:
        cached = {split: collect(level, split) for split in SPLITS}
        all_rows = sum((cached[s][1] for s in SPLITS), [])
        all_y = np.concatenate([cached[s][0].y for s in SPLITS]).astype(int)
        keys = ["edge_count", "open_edge_density", "component_count", "beta1", "mean_degree",
                "dead_ends", "junctions", "start_degree", "goal_degree",
                "start_component_size", "goal_component_size", "start_local_density",
                "goal_local_density"]
        entry = {"sizes": {s: int(cached[s][0].n) for s in SPLITS},
                 "positive_rate": {s: float(cached[s][0].y.mean()) for s in SPLITS}}
        for key in keys:
            entry[key] = summarize(np.asarray([r[key] for r in all_rows]), all_y)
        degree = np.asarray([r["degree_histogram"] for r in all_rows])
        entry["degree_histogram_mean"] = {
            "positive": degree[all_y == 1].mean(0).tolist(),
            "negative": degree[all_y == 0].mean(0).tolist(),
        }
        if level != "H0":
            z = cached["test"][0].arrays
            pos = np.arange(0, len(z["y"]), 2)
            neg = pos + 1
            entry["test_pair_exact_match"] = {
                k: float(np.mean(z[k][pos] == z[k][neg]))
                for k in ("edge_count", "component_count", "beta1")
            }
            paths = z["shortest_path_length"][z["y"] == 1]
            entry["positive_path_length"] = {
                "mean": float(paths.mean()), "std": float(paths.std()),
                "min": int(paths.min()), "max": int(paths.max()),
            }
        dataset_stats[level] = entry

        tr, va, te = [cached[s][0] for s in SPLITS]
        ftr, fva, fte = [cached[s][2] for s in SPLITS]
        ps = logistic_predictions(ftr, tr.y, fva, va.y, fte, steps=500)
        ptr = local_patch_features(tr)
        pva = local_patch_features(va)
        pte = local_patch_features(te)
        pl = logistic_predictions(ptr, tr.y, pva, va.y, pte, steps=250)
        baselines[level] = {
            "scalar_logistic": classification(te.y, ps),
            "local_patch_logistic_radius4": classification(te.y, pl),
            "notes": "Component sizes and any direct reachability/path feature are excluded from classifiers."
        }
        print(level, baselines[level], flush=True)
    (RESULTS / "dataset_stats.json").write_text(json.dumps(dataset_stats, indent=2))
    (RESULTS / "shortcut_baselines.json").write_text(json.dumps(baselines, indent=2))


if __name__ == "__main__":
    main()
