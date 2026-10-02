"""Aggregate Phase 2-A regression metrics, figures, and decision report."""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

P2 = Path(__file__).resolve().parent
ROOT = P2.parent
sys.path.insert(0, str(ROOT / "src"))
from dataset import MazeData

OUT = ROOT / "results_phase2" / "aux_only"
FIG = OUT / "figures"


def ranks(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]: j += 1
        r[order[i:j]] = (i + 1 + j) / 2
        i = j
    return r


def score(y, p):
    err = p - y
    mse = float(np.mean(err ** 2))
    pearson = None if np.std(p) == 0 else float(np.corrcoef(y, p)[0, 1])
    spearman = None if np.std(p) == 0 else float(np.corrcoef(ranks(y), ranks(p))[0, 1])
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(err))),
            "r2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
            "pearson": pearson, "spearman": spearman,
            "rounded_accuracy": float(np.mean(np.rint(p) == y))}


def aggregate(rows):
    return {k: {"mean": float(np.mean([r[k] for r in rows])),
                "std": float(np.std([r[k] for r in rows]))} for k in rows[0]}


def savefig(fig, name):
    fig.tight_layout(); fig.savefig(FIG / name, dpi=150); plt.close(fig)


def main():
    cfg = json.loads((P2 / "config.json").read_text())
    test = MazeData("H3", "test")
    y = test.arrays["beta1"].astype(float)
    train = MazeData("H3", "train")
    constant = np.full(len(y), train.arrays["beta1"].mean())
    out = {"constant_baseline": score(y, constant), "models": {}}
    csv_rows, predictions = [], {}
    for model in cfg["models"]:
        rows = []
        for seed in cfg["aux_only_seeds"]:
            z = np.load(OUT / f"pred_{model}_s{seed}.npz", allow_pickle=False)
            m = score(y, z["pred_beta1"])
            rows.append(m); predictions[(model, seed)] = z["pred_beta1"]
            csv_rows.append({"model": model, "seed": seed, **m})
        out["models"][model] = {"per_seed": rows, "aggregate": aggregate(rows)}
    out["scalar_reference"] = {
        "description": "beta1 = edge_count - 225 + component_count (identity)",
        "r2": 1.0, "role": "reference only; not an auxiliary difficulty baseline"
    }
    (OUT / "metrics.json").write_text(json.dumps(out, indent=2))
    with (OUT / "per_seed_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_rows[0].keys()); w.writeheader(); w.writerows(csv_rows)

    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, model in zip(axes, cfg["models"]):
        p = predictions[(model, cfg["aux_only_seeds"][0])]
        ax.scatter(y, p, s=8, alpha=.35); ax.plot([y.min(), y.max()], [y.min(), y.max()], "k--")
        ax.set_title(f"{model.upper()} seed 1"); ax.set_xlabel("true β₁"); ax.set_ylabel("predicted β₁")
    savefig(fig, "fig1_true_vs_predicted.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, color in zip(cfg["models"], ("#2b6cb0", "#c05621")):
        ax.hist(predictions[(model, 1)] - y, bins=35, alpha=.5, density=True, label=model.upper(), color=color)
    ax.axvline(0, color="k", ls="--"); ax.set_xlabel("prediction residual"); ax.set_ylabel("density"); ax.legend()
    savefig(fig, "fig2_residuals.png")

    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.hist(y, bins=np.arange(y.min()-.5, y.max()+1.5), color="#4a5568")
    ax.set_xlabel("β₁"); ax.set_ylabel("test samples"); savefig(fig, "fig3_beta1_histogram.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cls, color in ((1, "#2b6cb0"), (0, "#c05621")):
        ax.hist(y[test.y == cls], bins=25, density=True, histtype="step", lw=2,
                color=color, label=f"reach y={cls}")
    ax.set_xlabel("β₁"); ax.set_ylabel("density"); ax.legend(); savefig(fig, "fig4_beta1_by_reachability.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, model in zip(axes, cfg["models"]):
        for color, seed in zip(("#2b6cb0", "#c05621", "#2f855a"), cfg["aux_only_seeds"]):
            z = np.load(OUT / f"pred_{model}_s{seed}.npz", allow_pickle=False)
            h = json.loads(str(z["history"]))
            epochs = [e["epoch"] for e in h]
            ax.plot(epochs, [e["train_mse"] for e in h], color=color, alpha=.6,
                    ls="--", label=f"seed {seed} train")
            ax.plot(epochs, [e["val_mse"] for e in h], color=color, label=f"seed {seed} val")
        ax.set_title(model.upper()); ax.set_xlabel("epoch"); ax.set_ylabel("normalized MSE")
        ax.legend(ncol=2, fontsize=8)
    savefig(fig, "fig5_training_validation_loss.png")

    best_model = max(cfg["models"], key=lambda m: out["models"][m]["aggregate"]["r2"]["mean"])
    best_r2 = out["models"][best_model]["aggregate"]["r2"]["mean"]
    proceed = best_r2 > 0
    dist = json.loads((OUT / "beta1_distribution.json").read_text())
    report = f"""# Phase 2-A — β₁ Auxiliary-only Report

## Decision

Best model: **{best_model.upper()}**, test R² = **{best_r2:.3f}**.  
Decision: **{'PROCEED to Phase 2-B' if proceed else 'PAUSE full Phase 2-B sweep'}**.

## Distribution and leakage check

- Train β₁: min={dist['train']['min']}, max={dist['train']['max']}, mean={dist['train']['mean']:.3f}, std={dist['train']['std']:.3f}, unique={dist['train']['unique_values']}.
- Pairwise exact match: train/val/test = {dist['train']['pair_exact_match']:.3f}/{dist['val']['pair_exact_match']:.3f}/{dist['test']['pair_exact_match']:.3f}.
- Correlation with reachability: {dist['test']['correlation_with_reachability']:.3f}; discrete MI={dist['test']['mutual_information_nats']:.6f} nats.
- β₁-only logistic reachability accuracy: {dist['beta1_only_logistic_test_accuracy']:.3f}.
- β₁ spot recalculation errors: {dist['train']['recalculation_errors'] + dist['val']['recalculation_errors'] + dist['test']['recalculation_errors']}.

## Regression results (3 seeds)

| model | R² | RMSE | MAE | Pearson | Spearman | rounded acc |
|---|---:|---:|---:|---:|---:|---:|
"""
    for model in cfg["models"]:
        a = out["models"][model]["aggregate"]
        report += (f"| {model.upper()} | {a['r2']['mean']:.3f} ± {a['r2']['std']:.3f} | "
                   f"{a['rmse']['mean']:.3f} | {a['mae']['mean']:.3f} | {a['pearson']['mean']:.3f} | "
                   f"{a['spearman']['mean']:.3f} | {a['rounded_accuracy']['mean']:.3f} |\n")
    report += f"""

Constant baseline R² = {out['constant_baseline']['r2']:.3f}. The scalar identity using edge count and component count has R²=1 by definition and is reference-only.

## Interpretation

{'β₁ is reproducibly learnable from the image and exceeds the constant baseline, so the predefined gate permits the full multi-task experiment.' if proceed else 'β₁ does not reproducibly exceed the constant baseline. Per protocol, only pooling/count diagnostics should follow; the full sweep is not justified.'}
"""
    (OUT / "aux_only_report.md").write_text(report)
    (OUT / "decision.json").write_text(json.dumps({"proceed": proceed, "best_model": best_model, "best_r2": best_r2}, indent=2))
    print(json.dumps({"decision": proceed, "best_model": best_model, "best_r2": best_r2,
                      "metrics": out}, indent=2)[:10000])


if __name__ == "__main__":
    main()
