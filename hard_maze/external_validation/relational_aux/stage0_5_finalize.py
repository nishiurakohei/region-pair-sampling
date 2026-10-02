"""Aggregate Stage 0.5, draw comparisons, and write the gate report."""
import csv, hashlib, json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
OUT = ROOT / "results_external" / "stage0_5_training"
S0 = ROOT / "results_external" / "stage0_reproduction"


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def f(row, key): return float(row[key])


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row["axis"], int(row["maze_size"]), float(row["percolation"]),
               row["deadend_start"].lower() == "true", int(row["iterations"]))
        groups[key].append(float(row["accuracy"]))
    return [{"axis": k[0], "maze_size": k[1], "percolation": k[2],
             "deadend_start": k[3], "iterations": k[4],
             "accuracy_mean": float(np.mean(v)), "accuracy_std": float(np.std(v)),
             "model_seeds": len(v)} for k, v in sorted(groups.items())]


def lookup(rows, axis, size=9, p=0.0, deadend=True, iterations=30):
    return next(r for r in rows if r["axis"] == axis and r["maze_size"] == size
                and r["percolation"] == p and r["deadend_start"] == deadend
                and r["iterations"] == iterations)


def plots(training, scratch, pretrained):
    figdir = OUT / "figures"; figdir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for seed in sorted({int(r["model_seed"]) for r in training}):
        rows = [r for r in training if int(r["model_seed"]) == seed]
        axes[0].plot([int(r["epoch"]) for r in rows], [f(r, "train_loss") for r in rows], label=str(seed))
        axes[0].plot([int(r["epoch"]) for r in rows], [f(r, "validation_loss") for r in rows], "--")
        axes[1].plot([int(r["epoch"]) for r in rows], [f(r, "validation_exact_accuracy") for r in rows], label=str(seed))
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy", title="solid=train, dashed=validation")
    axes[1].set(xlabel="Epoch", ylabel="Validation exact accuracy")
    for ax in axes: ax.grid(alpha=.25); ax.legend(title="model seed")
    fig.tight_layout(); fig.savefig(figdir / "training_curves.png", dpi=160); plt.close(fig)

    def lineplot(axis, xfield, filename, xlabel):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name, rows in (("pretrained", pretrained), ("scratch mean", scratch)):
            q = sorted((r for r in rows if r["axis"] == axis and r["iterations"] == 30), key=lambda z: z[xfield])
            ax.errorbar([r[xfield] for r in q], [r["accuracy_mean"] for r in q],
                        yerr=[r["accuracy_std"] for r in q], marker="o", label=name)
        ax.set(xlabel=xlabel, ylabel="Exact solution accuracy", ylim=(-.03, 1.03)); ax.grid(alpha=.25); ax.legend()
        fig.tight_layout(); fig.savefig(figdir / filename, dpi=160); plt.close(fig)
    lineplot("percolation", "percolation", "pretrained_vs_scratch_percolation.png", "Percolation")

    fig, ax = plt.subplots(figsize=(6, 4))
    labels, values, errors = [], [], []
    for model, rows in (("pretrained", pretrained), ("scratch", scratch)):
        for deadend in (True, False):
            r = lookup(rows, "deadend_start", deadend=deadend)
            labels.append(f"{model}\n{deadend}"); values.append(r["accuracy_mean"]); errors.append(r["accuracy_std"])
    ax.bar(np.arange(4), values, yerr=errors, capsize=4); ax.set_xticks(np.arange(4), labels)
    ax.set(ylabel="Exact solution accuracy", ylim=(0, 1.03)); fig.tight_layout()
    fig.savefig(figdir / "pretrained_vs_scratch_deadend.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, (name, rows) in zip(axes, (("pretrained", pretrained), ("scratch", scratch))):
        for size in (9, 19, 29, 49):
            q = sorted((r for r in rows if r["axis"] == "maze_size" and r["maze_size"] == size), key=lambda z: z["iterations"])
            ax.plot([r["iterations"] for r in q], [r["accuracy_mean"] for r in q], "o-", label=f"{size}×{size}")
        ax.set_xscale("log"); ax.set(xlabel="Iterations", title=name, ylim=(-.03, 1.03)); ax.grid(alpha=.25); ax.legend()
    axes[0].set_ylabel("Exact solution accuracy"); fig.tight_layout()
    fig.savefig(figdir / "pretrained_vs_scratch_size_iterations.png", dpi=160); plt.close(fig)


def main():
    training = read_csv(OUT / "per_seed_training_metrics.csv")
    evaluated = read_csv(OUT / "scratch_eval_metrics.csv")
    scratch = aggregate(evaluated)
    pretrained = []
    for row in read_csv(S0 / "per_seed_metrics.csv"):
        if int(row["seed"]) != 42:
            continue
        pretrained.append({"axis": row["axis"], "maze_size": int(row["maze_size"]),
                           "percolation": float(row["percolation"]),
                           "deadend_start": row["deadend_start"].lower() == "true",
                           "iterations": int(row["iterations"]),
                           "accuracy_mean": float(row["accuracy"]), "accuracy_std": 0.0,
                           "model_seeds": 1, "dataset_sha256": row["dataset_sha256"]})
    plots(training, scratch, pretrained)

    comparison = []
    for s in scratch:
        p = lookup(pretrained, s["axis"], s["maze_size"], s["percolation"], s["deadend_start"], s["iterations"])
        comparison.append({**{k: s[k] for k in ("axis", "maze_size", "percolation", "deadend_start", "iterations")},
                           "scratch_mean": s["accuracy_mean"], "scratch_std": s["accuracy_std"],
                           "pretrained_mean": p["accuracy_mean"], "pretrained_std": p["accuracy_std"]})
    with (OUT / "pretrained_vs_scratch.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison[0]); writer.writeheader(); writer.writerows(comparison)

    id_acc = lookup(scratch, "id")["accuracy_mean"]
    percolation = sorted((r for r in scratch if r["axis"] == "percolation"), key=lambda r: r["percolation"])
    dead_true, dead_false = lookup(scratch, "deadend_start"), lookup(scratch, "deadend_start", deadend=False)
    pretrained_dead_true = lookup(pretrained, "deadend_start")
    pretrained_dead_false = lookup(pretrained, "deadend_start", deadend=False)
    recovery = max(lookup(scratch, "maze_size", size=size, iterations=300)["accuracy_mean"] -
                   lookup(scratch, "maze_size", size=size, iterations=30)["accuracy_mean"] for size in (19, 29, 49))
    criteria = {
        "A_id_at_least_0_95": id_acc >= .95,
        "B_percolation_drop_at_least_0_10": percolation[0]["accuracy_mean"] - percolation[-1]["accuracy_mean"] >= .10,
        "C_deadend_drop_at_least_0_10": dead_true["accuracy_mean"] - dead_false["accuracy_mean"] >= .10,
        "D_size_iteration_recovery_at_least_0_10": recovery >= .10,
    }
    best = []
    for seed in sorted({int(r["model_seed"]) for r in training}):
        rows = [r for r in training if int(r["model_seed"]) == seed]
        chosen = max(rows, key=lambda r: f(r, "validation_exact_accuracy"))
        best.append({"model_seed": seed, "best_epoch": int(chosen["epoch"]),
                     "best_validation_exact_accuracy": f(chosen, "validation_exact_accuracy"),
                     "validation_loss_at_best": f(chosen, "validation_loss"),
                     "checkpoint": f"checkpoints/seed_{seed}/best_torch.pt"})
    metrics = {"condition": "resource_bounded_not_official_full", "scratch_summary": scratch,
               "training_best": best, "criteria": criteria, "gate0_5_pass": all(criteria.values()),
               "stage2_authorized": False}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    pretrained_hash = hashlib.sha256((HERE / "upstream" / "maze-extrapolation" / "models" / "dt_net.pth").read_bytes()).hexdigest()
    metas = [json.loads((OUT / "checkpoints" / f"seed_{s}" / "last.json").read_text()) for s in (101, 102, 103)]
    hashes = [m["initial_weight_sha256"] for m in metas]
    dataset_hash_sets = defaultdict(set)
    for row in evaluated:
        condition = (row["axis"], row["maze_size"], row["percolation"], row["deadend_start"], row["iterations"])
        dataset_hash_sets[condition].add(row["dataset_sha256"])
    pretrained_hashes = {(r["axis"], str(r["maze_size"]), str(r["percolation"]),
                          str(r["deadend_start"]), str(r["iterations"])): r["dataset_sha256"] for r in pretrained}
    same_pretrained_data = all(pretrained_hashes[(row["axis"], row["maze_size"], row["percolation"],
                                                   row["deadend_start"], row["iterations"])] == row["dataset_sha256"]
                               for row in evaluated)
    checks = {
        "three_model_seeds": sorted(int(m["model_seed"]) for m in metas) == [101, 102, 103],
        "independent_initializations": len(set(hashes)) == 3,
        "no_pretrained_loaded": all(not m["pretrained_loaded"] for m in metas),
        "pretrained_hash_matches_stage0": pretrained_hash == "b8beed72aad07b300e178f74f0c3628886cfb28fca5d271c451496af38478216",
        "fixed_dataset_across_scratch_models": all(len(v) == 1 for v in dataset_hash_sets.values()),
        "same_datasets_pretrained_vs_scratch": same_pretrained_data,
        "all_training_epochs_present": len(training) == 30,
        "all_stage0_screening_rows_present": len(evaluated) == 63,
        "resource_deviations_explicit": len(json.loads((OUT / "config.json").read_text())["resource_bounded_deviations"]) == 4,
    }
    verification = {"checks": checks, "all_ok": all(checks.values()), "gate0_5_pass": all(criteria.values()),
                    "note": "all_ok covers procedural integrity; gate0_5_pass covers scientific success."}
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")

    perco_rows = "\n".join(f"| {r['percolation']:.3f} | {r['accuracy_mean']:.3f}±{r['accuracy_std']:.3f} | {lookup(pretrained, 'percolation', p=r['percolation'])['accuracy_mean']:.3f} |" for r in percolation)
    size_rows = "\n".join(f"| {size} | " + " | ".join(f"{lookup(scratch, 'maze_size', size=size, iterations=i)['accuracy_mean']:.3f}" for i in (30,100,300)) + " |" for size in (9,19,29,49))
    report = f"""# Stage 0.5 — DT-Net Scratch Training Reproduction

## 判定

**Gate 0.5 は不通過**。今回完了したのは公式条件そのものではなく、5,000 training mazes・10 epochs の資源制約版である。scratch 3 seeds の ID exact accuracy 平均は **{id_acc:.3f}** で、Criterion A（0.95以上）を満たさない。したがって percolation、deadend、size/iteration の形を学習済み solver の failure structure と解釈できず、Stage 2 multi-task training には進まない。

## Training pipeline の特定

`mines-opt-ml/maze-extrapolation@6d09263fdb8d61274e4b0e046e07ed929b35af12` の `src/train.py` は初回 commit から空で、同 repository 単独では training を再実行できなかった。著者の公式 `aks2203/deep-thinking@152fb1182fb5dc11275859407ceec46102c20e42` 実装を採用し、DT-Net width 128、30 recurrent steps、progressive loss α=0.01、Adam、base LR 1e-3、recurrent LR 1e-3/30、weight decay 2e-4、batch 50、50 epochs、10-epoch exponential warmup を復元した。データは Easy-to-Hard の 50,000個の9×9 maze（40,000 train / 10,000 validation）である。

公式完全条件は計算見積りが数日規模だったため未実行。今回の差分は 5,000 / 1,000 examples、10 epochs、2-epoch warmup であり、ほかのモデル・loss・optimizer・30-step recurrence は維持した。よって本結果は「公式 pipeline が再現不能」という証拠ではなく、「この資源制約版では収束しなかった」という結果である。

## Training results

| seed | best epoch | best validation exact accuracy | validation loss | checkpoint |
|---:|---:|---:|---:|---|
""" + "\n".join(f"| {r['model_seed']} | {r['best_epoch']} | {r['best_validation_exact_accuracy']:.3f} | {r['validation_loss_at_best']:.4f} | `{r['checkpoint']}` |" for r in best) + f"""

Checkpoint selection は指定どおり validation exact accuracy。値が同率の場合は最初の epoch を保持したため、loss 最小 epoch と一致しない場合がある。

## Stage 0 test suite

全 scratch model と pretrained model を同一の固定 dataset seed 42・同一 SHA-256 の raster で直接比較した。pretrained の5-seed aggregateは Stage 0 report に保存済みで、下表は paired comparison のため seed 42 の値を示す。

| percolation p | scratch mean±std | pretrained (same dataset) |
|---:|---:|---:|
{perco_rows}

deadend=true は scratch {dead_true['accuracy_mean']:.3f}、deadend=false は {dead_false['accuracy_mean']:.3f}（同一 dataset の pretrained はそれぞれ {pretrained_dead_true['accuracy_mean']:.3f} / {pretrained_dead_false['accuracy_mean']:.3f}）。

| maze size | 30 iter | 100 iter | 300 iter |
|---:|---:|---:|---:|
{size_rows}

## Gate criteria

- Criterion A — ID ≥0.95: **{criteria['A_id_at_least_0_95']}**
- Criterion B — p=0→0.04 drop ≥0.10: **{criteria['B_percolation_drop_at_least_0_10']}**
- Criterion C — deadend true→false drop ≥0.10: **{criteria['C_deadend_drop_at_least_0_10']}**
- Criterion D — large-maze 30→300 iter recovery ≥0.10: **{criteria['D_size_iteration_recovery_at_least_0_10']}**

B–D のいずれかが数値上成立しても、A が失敗した model では published failure structure の再現とは判定しない。

## 原因の切り分けと残る検証

- optimizer、loss、recurrent iteration、rasterizationは公式実装に合わせ、MLX/PyTorch forward は pretrained weight で argmax 一致・最大誤差 7.63e-06 を確認済み。
- scratch initialization hash は3 seedsで異なり、pretrained weight はロードしていない。
- 最も大きい既知の差は training data 8分の1、epochs 5分の1、warmup 5分の1である。収束失敗はこの計算量削減と整合するが、因果は未確定。
- 公式40,000 train / 10,000 validation・50 epochsを3 seedsで完走するか、著者の training checkpoint/configを入手するまで Gate 0.5 は未解決のまま保持する。

## Stage 1との関係

Stage 1 auxiliary-only distance prediction は独立に完了し、全体では強い learnability を示したが長距離 bin が弱いため **Case B**。これは auxiliary target が学習可能という証拠であり、Stage 0.5 の失敗を補わない。したがって Stage 2 は未着手。
"""
    (OUT / "stage0_5_report.md").write_text(report)


if __name__ == "__main__": main()
