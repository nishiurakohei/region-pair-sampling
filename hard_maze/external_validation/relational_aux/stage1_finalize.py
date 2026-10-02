"""Create diagnostics, figures, verification, and report for Stage 1."""
import csv, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parents[1]
OUT = HERE.parent / "results_external" / "stage1_aux_learnability"


def rankdata(values):
    order = np.argsort(values, kind="mergesort"); sorted_values = values[order]; ranks = np.empty(len(values)); lo = 0
    while lo < len(values):
        hi = lo + 1
        while hi < len(values) and sorted_values[hi] == sorted_values[lo]: hi += 1
        ranks[order[lo:hi]] = (lo + hi - 1) / 2; lo = hi
    return ranks


def metrics(y, pred):
    e = pred - y; denom = np.sum((y - y.mean()) ** 2); varying = np.std(pred) > 1e-12
    return {"raw_mae": float(np.mean(np.abs(e))), "normalized_mae": float(np.mean(np.abs(e)) / 9),
            "rmse": float(np.sqrt(np.mean(e ** 2))), "r2": float(1 - np.sum(e ** 2) / denom),
            "pearson": float(np.corrcoef(y, pred)[0, 1]) if varying else 0.0,
            "spearman": float(np.corrcoef(rankdata(y), rankdata(pred))[0, 1]) if varying else 0.0}


def main():
    cfg = json.loads((OUT / "config.json").read_text()); stored = json.loads((OUT / "metrics.json").read_text())
    pred_file = np.load(OUT / "checkpoints" / "seed_401" / "test_predictions.npz")
    y, pred, manhattan, detour = (pred_file[k].astype(float) for k in ("true", "predicted", "manhattan", "detour_ratio"))
    train = np.load(OUT / "fixed_train_k16.npz"); train_y = train["graph_distance"].astype(float); train_m = train["manhattan_distance"].astype(float)
    slope, intercept = np.polyfit(train_m, train_y, 1); mean_pred = np.full_like(y, train_y.mean()); linear_pred = slope * manhattan + intercept
    stored["baselines"] = {"constant_mean": metrics(y, mean_pred), "manhattan": metrics(y, manhattan),
        "manhattan_linear": metrics(y, linear_pred), "linear_coefficients": {"slope": float(slope), "intercept": float(intercept)}}
    (OUT / "metrics.json").write_text(json.dumps(stored, indent=2) + "\n")

    bin_specs = [(1, 5, "1-5"), (6, 10, "6-10"), (11, 15, "11-15"), (16, 100, "16+")]; bin_rows = []
    for lo, hi, label in bin_specs:
        mask = (y >= lo) & (y <= hi)
        for name, values in (("aux_model", pred), ("constant_mean", mean_pred), ("manhattan", manhattan), ("manhattan_linear", linear_pred)):
            row = {"distance_bin": label, "predictor": name, "count": int(mask.sum())}
            row.update(metrics(y[mask], values[mask])); bin_rows.append(row)
    with (OUT / "distance_bin_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bin_rows[0]); w.writeheader(); w.writerows(bin_rows)
    with (OUT / "per_pair_predictions.csv").open("w", newline="") as f:
        fields = ["true_graph_distance", "predicted_graph_distance", "residual", "manhattan_distance", "detour_ratio", "distance_bin"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for yt, yp, dm, ratio in zip(y, pred, manhattan, detour):
            label = next(label for lo, hi, label in bin_specs if lo <= yt <= hi)
            w.writerow(dict(zip(fields, [yt, yp, yp - yt, dm, ratio, label])))

    figdir = OUT / "figures"; figdir.mkdir(exist_ok=True); rng = np.random.default_rng(42); sample = rng.choice(len(y), 6000, replace=False)
    fig, ax = plt.subplots(figsize=(5.5, 5)); ax.scatter(y[sample], pred[sample], s=5, alpha=.2); ax.plot([3, 23], [3, 23], "k--")
    ax.set(xlabel="True graph distance", ylabel="Predicted graph distance", title="Auxiliary distance prediction"); fig.tight_layout(); fig.savefig(figdir / "true_vs_predicted.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4)); ax.scatter(y[sample], pred[sample] - y[sample], s=5, alpha=.2); ax.axhline(0, color="k", ls="--")
    ax.set(xlabel="True graph distance", ylabel="Residual (predicted - true)"); fig.tight_layout(); fig.savefig(figdir / "residual_vs_true.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4)); labels = [x[2] for x in bin_specs]; model_mae = [next(r["raw_mae"] for r in bin_rows if r["distance_bin"] == b and r["predictor"] == "aux_model") for b in labels]
    linear_mae = [next(r["raw_mae"] for r in bin_rows if r["distance_bin"] == b and r["predictor"] == "manhattan_linear") for b in labels]
    pos = np.arange(4); ax.bar(pos - .18, model_mae, .36, label="Aux model"); ax.bar(pos + .18, linear_mae, .36, label="Linear Manhattan")
    ax.set_xticks(pos, labels); ax.set(xlabel="True-distance bin", ylabel="Raw-distance MAE"); ax.legend(); fig.tight_layout(); fig.savefig(figdir / "performance_by_distance_bin.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4.5)); hb = ax.hexbin(manhattan, y, gridsize=(6, 20), mincnt=1, cmap="viridis"); fig.colorbar(hb, ax=ax, label="Pair count")
    ax.set(xlabel="Manhattan distance", ylabel="Graph distance"); fig.tight_layout(); fig.savefig(figdir / "graph_vs_manhattan.png", dpi=170); plt.close(fig)
    iteration_metrics = stored["model"][0]["iterations"]; its = sorted(map(int, iteration_metrics)); maes = [iteration_metrics[str(i)]["raw_mae"] for i in its]; r2s = [iteration_metrics[str(i)]["r2"] for i in its]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8)); axes[0].plot(its, maes, "o-"); axes[1].plot(its, r2s, "o-")
    axes[0].set(xlabel="Test-time iterations", ylabel="Raw MAE"); axes[1].set(xlabel="Test-time iterations", ylabel="$R^2$");
    for ax in axes: ax.set_xscale("log"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(figdir / "performance_vs_iterations.png", dpi=170); plt.close(fig)

    pair_checks = json.loads((OUT / "verification.json").read_text())["checks"]
    primary = stored["model"][0]["iterations"]["30"]; linear = stored["baselines"]["manhattan_linear"]
    checks = dict(pair_checks, outputs_finite=bool(np.isfinite(pred).all()), prediction_count=len(pred) == cfg["test_mazes"] * cfg["pairs_per_maze"],
                  checkpoint_present=(OUT / "checkpoints" / "seed_401" / "best.safetensors").exists(),
                  training_completed=len(list(csv.DictReader((OUT / "checkpoints" / "seed_401" / "training_metrics.csv").open()))) == cfg["epochs"],
                  positive_r2=primary["r2"] > 0, beats_constant=primary["raw_mae"] < stored["baselines"]["constant_mean"]["raw_mae"],
                  beats_manhattan_linear=primary["raw_mae"] < linear["raw_mae"], figures_complete=len(list(figdir.glob("*.png"))) == 5)
    verification = {"checks": checks, "all_ok": all(checks.values()), "stage1_success": primary["r2"] > 0 and primary["raw_mae"] < linear["raw_mae"]}
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    long_model = next(r for r in bin_rows if r["distance_bin"] == "16+" and r["predictor"] == "aux_model")
    long_linear = next(r for r in bin_rows if r["distance_bin"] == "16+" and r["predictor"] == "manhattan_linear")
    report = f"""# Stage 1 — Random-Pair Distance Learnability\n\n## 判定\n\n**Stage 1 pass, Case B**。30 iterationsのauxiliary modelはraw MAE {primary['raw_mae']:.3f}、R² {primary['r2']:.3f}、Pearson {primary['pearson']:.3f}で、linear-Manhattan baseline（MAE {linear['raw_mae']:.3f}, R² {linear['r2']:.3f}）を明確に上回った。一方、16+ binではbaselineより良いがabsolute errorとbin内correlationが悪化し、long-distance weaknessが残る。\n\n## 設計\n\nEasy-to-Hard 9×9 mazeを40k/10k/10kに分離し、各mazeからstart/goalを除外した16 fixed pairsをsamplingした。Manhattan distanceは3以上、targetはexact BFSの graph distance / 9。pair marker channelは追加せず、元のstart/goal markerを消したgraph-only rasterからDT-Net coreでH(G)を一度計算し、h_u, h_v, |h_u-h_v|, h_u⊙h_v, global featureをscalar headへ渡した。これにより1 mazeの16 pairでcore計算を共有できる。\n\n学習は5,000 mazes（80,000 pair/epoch）×10 epochs、seed 401。これはlearnability判定用のresource-bounded conditionであり、full 40k学習ではない。\n\n## Q7–Q11\n\n- Q7: learnable。30 iterationsでR²={primary['r2']:.3f}。\n- Q8: Manhattan shortcutを上回る。linear baseline比でMAEを{linear['raw_mae']-primary['raw_mae']:.3f}削減。\n- Q9: 16+ binはmodel MAE {long_model['raw_mae']:.3f}、linear-Manhattan MAE {long_linear['raw_mae']:.3f}でbaselineより良いが、R²={long_model['r2']:.3f}、Pearson={long_model['pearson']:.3f}でabsoluteには弱い。\n- Q10: 10→30 iterationsでMAE {iteration_metrics['10']['raw_mae']:.3f}→{primary['raw_mae']:.3f}へ改善。100 iterationsでは{iteration_metrics['100']['raw_mae']:.3f}へわずかに悪化し、単調改善ではない。\n- Q11: Case BとしてStage 2を検討できるauxiliary signalは存在する。ただしproceedにはStage 0.5 passも別途必要で、distance-dependent transfer analysisを事前計画する。\n\n## 解釈の範囲\n\n結果はmaze geometryを使う非自明なdistance signalを支持する。multi-taskでmain accuracyを改善すること、またはpair marker方式と同値であることはまだ示していない。\n"""
    (OUT / "aux_learnability_report.md").write_text(report)
    print(json.dumps(verification, indent=2)); print({"long_model_mae": long_model["raw_mae"], "long_linear_mae": long_linear["raw_mae"]})


if __name__ == "__main__": main()
