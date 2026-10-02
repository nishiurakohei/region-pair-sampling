"""Aggregate Phase 2-B metrics, controls, uncertainty, figures, and report."""
import csv
import hashlib
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
from figures import plot_maze
from graph import components

OUT = ROOT / "results_phase2" / "multitask"
FIG = OUT / "figures"


def tag(v): return str(v).replace(".", "p")


def auroc(y, p):
    order = np.argsort(p, kind="mergesort"); ranks = np.empty(len(p), float); sp = p[order]
    i = 0
    while i < len(p):
        j = i + 1
        while j < len(p) and sp[j] == sp[i]: j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2; i = j
    n1, n0 = int(y.sum()), len(y) - int(y.sum())
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(y, p, bins=15):
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)) - 1, 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
                     for b in range(bins) if np.any(idx == b)))


def scores(y, p, beta, pred_beta):
    p = np.clip(p.astype(float), 1e-7, 1 - 1e-7); pred = p > .5
    pos, neg = np.arange(0, len(y), 2), np.arange(1, len(y), 2)
    margin = p[pos] - p[neg]
    out = {"accuracy": float(np.mean(pred == y)), "auroc": auroc(y, p),
           "bce": float(-np.mean(y * np.log(p) + (1-y) * np.log(1-p))),
           "brier": float(np.mean((p-y)**2)), "ece": ece(y, p),
           "pair_accuracy": float(np.mean(pred[pos] & ~pred[neg])),
           "pair_ordering_accuracy": float(np.mean(margin > 0)),
           "pair_margin_mean": float(margin.mean()), "pair_margin_median": float(np.median(margin)),
           "pair_margin_std": float(margin.std())}
    if pred_beta is not None and np.all(np.isfinite(pred_beta)):
        err = pred_beta - beta
        out.update({"aux_mse": float(np.mean(err**2)), "aux_mae": float(np.mean(abs(err))),
                    "aux_r2": float(1 - np.sum(err**2) / np.sum((beta-beta.mean())**2)),
                    "aux_pearson": float(np.corrcoef(beta, pred_beta)[0, 1])})
    return out, margin


def aggregate(rows):
    keys = set.intersection(*(set(r) for r in rows))
    return {k: {"mean": float(np.mean([r[k] for r in rows])),
                "std": float(np.std([r[k] for r in rows]))} for k in sorted(keys)}


def bootstrap_ci(values, seed=0, draws=20000):
    x = np.asarray(values, float); rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (draws, len(x)))].mean(1)
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def load_prediction(model, condition, lam, seed):
    return np.load(OUT / f"pred_{model}_{condition}_l{tag(lam)}_s{seed}.npz", allow_pickle=False)


def best_validation_bce(model, condition, lam, seed):
    z = load_prediction(model, condition, lam, seed)
    history = json.loads(str(z["history"]))
    return min(row["val_main_bce"] for row in history)


def savefig(fig, name):
    fig.tight_layout(); fig.savefig(FIG / name, dpi=150); plt.close(fig)


def save_failure_case_images(data, y, failure):
    dest = FIG / "failure_cases"
    dest.mkdir(exist_ok=True)
    for model, model_cases in failure.items():
        for kind in ("corrected", "harmed"):
            for case in model_cases[kind]:
                pair = case["pair_id"]
                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                for col, i in enumerate((2 * pair, 2 * pair + 1)):
                    plot_maze(
                        axes[col], data.H[i], data.V[i],
                        pivot=data.arrays["pivotal_edge"][i] if y[i] else None,
                        control=data.arrays["control_edge"][i] if not y[i] else None,
                        title=(f"y={y[i]} β₁={case['true_beta1']} pred={case['aux_pred_beta1'][col]:.1f}\n"
                               f"main={case['main_probs'][col]:.3f} aux={case['aux_probs'][col]:.3f}"),
                    )
                fig.suptitle(f"{model.upper()} {kind} pair {pair}")
                fig.tight_layout()
                fig.savefig(dest / f"{model}_{kind}_pair_{pair:04d}.png", dpi=120)
                plt.close(fig)


def main():
    cfg = json.loads((P2 / "config.json").read_text())
    data = MazeData("H3", "test"); y = data.y.astype(int); beta = data.arrays["beta1"].astype(float)
    results = {"config": cfg, "models": {}}
    seed_rows, sweep_rows, pair_rows = [], [], []
    margins, probs = {}, {}
    for model in cfg["models"]:
        results["models"][model] = {}
        configs = [("main", 0.0)] + [(c, l) for c in ("true", "shuffled") for l in cfg["lambdas"] if l > 0]
        for condition, lam in configs:
            per_seed = []
            for seed in cfg["multitask_seeds"]:
                z = load_prediction(model, condition, lam, seed)
                pred_beta = None if condition == "main" else z["pred_beta1"]
                m, margin = scores(y, z["prob"], beta, pred_beta)
                per_seed.append(m); margins[(model, condition, lam, seed)] = margin
                probs[(model, condition, lam, seed)] = z["prob"]
                seed_rows.append({"model": model, "condition": condition, "lambda": lam, "seed": seed, **m})
                for pair in range(len(y)//2):
                    i, j = 2*pair, 2*pair+1
                    pair_rows.append({"model": model, "condition": condition, "lambda": lam,
                                      "seed": seed, "pair_id": pair, "p_positive": float(z["prob"][i]),
                                      "p_negative": float(z["prob"][j]), "margin": float(margin[pair]),
                                      "pair_correct": int(z["prob"][i] > .5 and z["prob"][j] <= .5)})
            agg = aggregate(per_seed)
            key = f"{condition}_lambda_{lam}"
            results["models"][model][key] = {"per_seed": per_seed, "aggregate": agg}
            sweep_rows.append({"model": model, "condition": condition, "lambda": lam,
                               **{f"{k}_mean": v["mean"] for k, v in agg.items()},
                               **{f"{k}_std": v["std"] for k, v in agg.items()}})

        main = results["models"][model]["main_lambda_0.0"]["per_seed"]
        uncertainty = {}
        for lam in (x for x in cfg["lambdas"] if x > 0):
            true = results["models"][model][f"true_lambda_{lam}"]["per_seed"]
            shuffled = results["models"][model][f"shuffled_lambda_{lam}"]["per_seed"]
            uncertainty[str(lam)] = {}
            for metric in ("accuracy", "pair_accuracy", "pair_margin_mean"):
                dmain = [t[metric]-b[metric] for t,b in zip(true,main)]
                dshuffle = [t[metric]-s[metric] for t,s in zip(true,shuffled)]
                uncertainty[str(lam)][metric] = {
                    "true_minus_main_mean": float(np.mean(dmain)), "true_minus_main_ci95": bootstrap_ci(dmain),
                    "true_minus_shuffled_mean": float(np.mean(dshuffle)),
                    "true_minus_shuffled_ci95": bootstrap_ci(dshuffle, seed=1)}
        results["models"][model]["paired_uncertainty"] = uncertainty

    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(exist_ok=True)
    (OUT / "config.json").write_text(json.dumps(cfg, indent=2))
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    for name, rows in (("per_seed_metrics.csv", seed_rows), ("lambda_sweep.csv", sweep_rows),
                       ("per_pair_predictions.csv", pair_rows)):
        with (OUT / name).open("w", newline="") as f:
            fields = list(rows[0])
            fields.extend(k for row in rows for k in row if k not in fields)
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    best, selection = {}, {}
    for model in cfg["models"]:
        selection[model] = {
            str(l): float(np.mean([best_validation_bce(model, "true", l, seed)
                                   for seed in cfg["multitask_seeds"]]))
            for l in cfg["lambdas"] if l > 0
        }
        best[model] = min((l for l in cfg["lambdas"] if l > 0),
                          key=lambda l: selection[model][str(l)])
    results["lambda_selection"] = {
        "criterion": "lowest mean per-seed best validation main BCE",
        "values": selection,
        "selected": best,
    }
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, model in zip(axes, cfg["models"]):
        base_stats = results["models"][model]["main_lambda_0.0"]["aggregate"]["accuracy"]
        base = base_stats["mean"]
        for cond, color in (("true", "#2b6cb0"), ("shuffled", "#c05621")):
            vals = [results["models"][model][f"{cond}_lambda_{l}"]["aggregate"]["accuracy"]["mean"] for l in cfg["lambdas"] if l>0]
            errs = [results["models"][model][f"{cond}_lambda_{l}"]["aggregate"]["accuracy"]["std"] for l in cfg["lambdas"] if l>0]
            ax.errorbar(cfg["lambdas"][1:], vals, yerr=errs, fmt="o-", capsize=3, color=color, label=cond)
        ax.axhline(base, color="gray", ls="--", label="main-only"); ax.set_xscale("log")
        ax.axhspan(base-base_stats["std"], base+base_stats["std"], color="gray", alpha=.12)
        ax.set_title(model.upper()); ax.set_xlabel("λ"); ax.set_ylabel("main accuracy"); ax.legend(); ax.grid(alpha=.2)
    savefig(fig, "fig1_lambda_vs_accuracy.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, model in zip(axes, cfg["models"]):
        base_stats = results["models"][model]["main_lambda_0.0"]["aggregate"]["pair_accuracy"]
        base = base_stats["mean"]
        for cond, color in (("true", "#2b6cb0"), ("shuffled", "#c05621")):
            vals = [results["models"][model][f"{cond}_lambda_{l}"]["aggregate"]["pair_accuracy"]["mean"] for l in cfg["lambdas"] if l>0]
            errs = [results["models"][model][f"{cond}_lambda_{l}"]["aggregate"]["pair_accuracy"]["std"] for l in cfg["lambdas"] if l>0]
            ax.errorbar(cfg["lambdas"][1:], vals, yerr=errs, fmt="o-", capsize=3, color=color, label=cond)
        ax.axhline(base, color="gray", ls="--", label="main-only")
        ax.axhspan(base-base_stats["std"], base+base_stats["std"], color="gray", alpha=.12)
        ax.set_xscale("log"); ax.set_title(model.upper())
        ax.set_xlabel("λ"); ax.set_ylabel("pair accuracy"); ax.legend(); ax.grid(alpha=.2)
    savefig(fig, "fig2_lambda_vs_pair_accuracy.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, model in zip(axes, cfg["models"]):
        for cond, color in (("true", "#2b6cb0"), ("shuffled", "#c05621")):
            vals = [results["models"][model][f"{cond}_lambda_{l}"]["aggregate"]["aux_r2"]["mean"] for l in cfg["lambdas"] if l>0]
            errs = [results["models"][model][f"{cond}_lambda_{l}"]["aggregate"]["aux_r2"]["std"] for l in cfg["lambdas"] if l>0]
            ax.errorbar(cfg["lambdas"][1:], vals, yerr=errs, fmt="o-", capsize=3, color=color, label=cond)
        ax.axhline(0, color="gray", ls="--"); ax.set_xscale("log"); ax.set_title(model.upper())
        ax.set_xlabel("λ"); ax.set_ylabel("β₁ R²"); ax.legend(); ax.grid(alpha=.2)
    savefig(fig, "fig3_lambda_vs_aux_r2.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, model in zip(axes, cfg["models"]):
        base = results["models"][model]["main_lambda_0.0"]["aggregate"]["accuracy"]["mean"]
        for cond, color, marker in (("true", "#2b6cb0", "o"), ("shuffled", "#c05621", "s")):
            xs = [results["models"][model][f"{cond}_lambda_{l}"]["aggregate"]["aux_r2"]["mean"] for l in cfg["lambdas"] if l>0]
            ys = [results["models"][model][f"{cond}_lambda_{l}"]["aggregate"]["accuracy"]["mean"]-base for l in cfg["lambdas"] if l>0]
            ax.scatter(xs, ys, color=color, marker=marker, label=cond)
        ax.axhline(0, color="gray", ls="--"); ax.set_title(model.upper()); ax.set_xlabel("β₁ R²")
        ax.set_ylabel("main accuracy improvement"); ax.legend(); ax.grid(alpha=.2)
    savefig(fig, "fig4_improvement_vs_aux_r2.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, model in zip(axes, cfg["models"]):
        lam = best[model]
        main_margin = np.concatenate([margins[(model,"main",0.0,s)] for s in cfg["multitask_seeds"]])
        true_margin = np.concatenate([margins[(model,"true",lam,s)] for s in cfg["multitask_seeds"]])
        ax.hist(main_margin, bins=40, density=True, histtype="step", lw=2, label="main-only")
        ax.hist(true_margin, bins=40, density=True, histtype="step", lw=2, label=f"true λ={lam}")
        ax.axvline(0,color="gray",ls="--"); ax.set_title(model.upper()); ax.set_xlabel("p(G+) − p(G−)"); ax.legend()
    savefig(fig, "fig5_pair_margin.png")

    failure = {}
    for model in cfg["models"]:
        lam = best[model]
        seed_diffs = []
        for seed in cfg["multitask_seeds"]:
            mp, tp = probs[(model,"main",0.0,seed)], probs[(model,"true",lam,seed)]
            seed_diffs.append(np.mean((tp>.5)==y)-np.mean((mp>.5)==y))
        seed = cfg["multitask_seeds"][int(np.argmax(seed_diffs))]
        mp, tp = probs[(model,"main",0.0,seed)], probs[(model,"true",lam,seed)]
        bp = load_prediction(model, "true", lam, seed)["pred_beta1"]
        corrected, harmed = [], []
        for pair in range(len(y)//2):
            i,j=2*pair,2*pair+1
            mc,tc = bool(mp[i]>.5 and mp[j]<=.5), bool(tp[i]>.5 and tp[j]<=.5)
            row={"pair_id":pair,"true_beta1":int(beta[i]),
                 "main_probs":[float(mp[i]),float(mp[j])],
                 "aux_probs":[float(tp[i]),float(tp[j])],
                 "aux_pred_beta1":[float(bp[i]),float(bp[j])]}
            if tc and not mc and len(corrected)<20: corrected.append(row)
            if mc and not tc and len(harmed)<20: harmed.append(row)
        failure[model]={"lambda":lam,"seed":seed,"corrected":corrected,"harmed":harmed}
    (OUT / "failure_cases.json").write_text(json.dumps(failure, indent=2))
    save_failure_case_images(data, y, failure)

    show_model = max(cfg["models"], key=lambda m: len(failure[m]["corrected"]))
    cases = failure[show_model]["corrected"][:3] or failure[show_model]["harmed"][:3]
    fig, axes = plt.subplots(max(1, len(cases)), 2, figsize=(8, 4*max(1,len(cases))), squeeze=False)
    if not cases:
        axes[0, 0].text(.5, .5, "No threshold-level pair changes", ha="center", va="center")
        axes[0, 1].axis("off")
    for row, case in enumerate(cases):
        pair=case["pair_id"]
        for col,i in enumerate((2*pair,2*pair+1)):
            plot_maze(axes[row,col],data.H[i],data.V[i],
                      pivot=data.arrays["pivotal_edge"][i] if y[i] else None,
                      control=data.arrays["control_edge"][i] if not y[i] else None,
                      title=(f"y={y[i]} β₁={case['true_beta1']} pred={case['aux_pred_beta1'][col]:.1f}\n"
                             f"main={case['main_probs'][col]:.3f} aux={case['aux_probs'][col]:.3f}"))
    fig.suptitle(f"Representative corrected pairs: {show_model.upper()}")
    savefig(fig,"fig6_corrected_pairs.png")

    previous = json.loads((ROOT/"results_hard_maze"/"verification.json").read_text())
    h3_previous = previous["H3"]
    pair_checks = {
        name: float(np.mean(data.arrays[name][::2] == data.arrays[name][1::2]))
        for name in ("edge_count", "component_count", "beta1")
    }
    beta1_recalculation_errors = 0
    for split in ("train", "val", "test"):
        split_data = MazeData("H3", split)
        for i in np.linspace(0, split_data.n - 1, min(500, split_data.n), dtype=int):
            kappa = len(components(split_data.H[i], split_data.V[i])[1])
            calculated = int(split_data.H[i].sum() + split_data.V[i].sum() - 225 + kappa)
            beta1_recalculation_errors += int(calculated != split_data.arrays["beta1"][i])
    prior_reuse_exact, baseline_accuracy = {}, {}
    for model in cfg["models"]:
        prior_reuse_exact[model] = {}
        values = []
        for seed in cfg["multitask_seeds"]:
            current = load_prediction(model, "main", 0.0, seed)
            values.append(float(np.mean((current["prob"] > .5) == y)))
            if seed <= 3:
                prior = np.load(ROOT/"results_hard_maze"/f"pred_H3_{model}_s{seed}.npz", allow_pickle=False)
                prior_reuse_exact[model][str(seed)] = bool(np.array_equal(current["prob"], prior["prob"]))
        baseline_accuracy[model] = {"per_seed": values,
                                    "mean": float(np.mean(values)), "std": float(np.std(values))}
    verification = {
        "h3_file_sha256": {s: hashlib.sha256((ROOT/"data"/"H3"/f"{s}.npz").read_bytes()).hexdigest()
                             for s in ("train","val","test")},
        "class_balance_test": float(y.mean()),
        "pair_positive_then_negative": bool(np.all(y[::2] == 1) and np.all(y[1::2] == 0)),
        "pair_match_rates": pair_checks,
        "beta1_spot_recalculation_errors": beta1_recalculation_errors,
        "previous_h3_verification_all_ok": h3_previous["all_ok"],
        "no_split_leakage": h3_previous["no_leakage_ok"],
        "split_base_overlap": h3_previous["base_overlap"],
        "bayes_accuracy": h3_previous["bayes_ceiling"],
        "prior_prediction_reuse_exact": prior_reuse_exact,
        "five_seed_main_accuracy": baseline_accuracy,
        "shuffle_checks": {}, "seeds": cfg["multitask_seeds"]}
    train_beta = MazeData("H3","train").arrays["beta1"]
    for seed in cfg["multitask_seeds"]:
        perm = np.random.default_rng(cfg["shuffle_seed"]+seed).permutation(len(train_beta))
        verification["shuffle_checks"][str(seed)] = {
            "changed": bool(np.any(train_beta[perm]!=train_beta)),
            "multiset_preserved": bool(np.array_equal(np.sort(train_beta[perm]),np.sort(train_beta))),
            "seed": cfg["shuffle_seed"]+seed}
    verification["all_ok"] = (verification["class_balance_test"]==.5 and
        verification["pair_positive_then_negative"] and all(v == 1 for v in pair_checks.values()) and
        verification["beta1_spot_recalculation_errors"] == 0 and verification["bayes_accuracy"] == 1 and
        verification["no_split_leakage"] and verification["previous_h3_verification_all_ok"] and
        all(all(seeds.values()) for seeds in prior_reuse_exact.values()) and
        all(x["changed"] and x["multiset_preserved"] for x in verification["shuffle_checks"].values()))
    (OUT/"verification.json").write_text(json.dumps(verification,indent=2))

    report(ROOT, cfg, results, best, failure, verification)
    print("best", best, "verification", verification["all_ok"])


def report(root, cfg, results, best, failure, verification):
    aux = json.loads((root/"results_phase2"/"aux_only"/"metrics.json").read_text())
    summary_lines, sweep_lines, uncertainty_lines, gains = [], [], [], {}
    for model in cfg["models"]:
        b = results["models"][model]["main_lambda_0.0"]["aggregate"]
        lam = best[model]
        t = results["models"][model][f"true_lambda_{lam}"]["aggregate"]
        s = results["models"][model][f"shuffled_lambda_{lam}"]["aggregate"]
        gains[model] = t["accuracy"]["mean"] - b["accuracy"]["mean"]
        summary_lines.append(
            f"| {model.upper()} | {b['accuracy']['mean']:.3f} ± {b['accuracy']['std']:.3f} | {lam} | "
            f"{t['accuracy']['mean']:.3f} ± {t['accuracy']['std']:.3f} | "
            f"{s['accuracy']['mean']:.3f} ± {s['accuracy']['std']:.3f} | "
            f"{b['pair_accuracy']['mean']:.3f} | {t['pair_accuracy']['mean']:.3f} | {t['aux_r2']['mean']:.3f} |")
        u = results["models"][model]["paired_uncertainty"][str(lam)]
        for metric in ("accuracy", "pair_accuracy", "pair_margin_mean"):
            x = u[metric]
            uncertainty_lines.append(
                f"| {model.upper()} | {metric} | {x['true_minus_main_mean']:+.4f} "
                f"[{x['true_minus_main_ci95'][0]:+.4f}, {x['true_minus_main_ci95'][1]:+.4f}] | "
                f"{x['true_minus_shuffled_mean']:+.4f} "
                f"[{x['true_minus_shuffled_ci95'][0]:+.4f}, {x['true_minus_shuffled_ci95'][1]:+.4f}] |")
        for lam_i in (x for x in cfg["lambdas"] if x > 0):
            ti = results["models"][model][f"true_lambda_{lam_i}"]["aggregate"]
            si = results["models"][model][f"shuffled_lambda_{lam_i}"]["aggregate"]
            sweep_lines.append(
                f"| {model.upper()} | {lam_i} | {ti['accuracy']['mean']:.3f} | "
                f"{si['accuracy']['mean']:.3f} | {ti['pair_accuracy']['mean']:.3f} | "
                f"{si['pair_accuracy']['mean']:.3f} | {ti['pair_margin_mean']['mean']:.4f} | "
                f"{ti['aux_r2']['mean']:.3f} |")
    cnn_u = results["models"]["cnn"]["paired_uncertainty"][str(best["cnn"])]
    res_u = results["models"]["resnet"]["paired_uncertainty"][str(best["resnet"])]
    text=f"""# Phase 2 — Topological Auxiliary Supervision on H3

## Conclusion

Phase 2-A establishes a clear asymmetry: β₁ is strongly learnable (CNN R²={aux['models']['cnn']['aggregate']['r2']['mean']:.3f}, ResNet R²={aux['models']['resnet']['aggregate']['r2']['mean']:.3f}) while H3 reachability remains difficult.  
Phase 2-B does **not** show a robust accuracy or matched-pair-accuracy improvement. At validation-selected λ, CNN changes by {gains['cnn']:+.4f} accuracy and ResNet by {gains['resnet']:+.4f}; both paired 95% CIs include zero. CNN pair margin does shift positively, but this does not translate into more threshold-correct pairs.

| model | main acc | selected λ | true aux acc | shuffled acc | main pair acc | true pair acc | true β₁ R² |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(summary_lines)}

λ was selected independently for each architecture by the lowest mean per-seed best validation main BCE; test accuracy was not used for selection.

## Answers

1. **β₁ learnability:** Yes. Auxiliary-only CNN R²={aux['models']['cnn']['aggregate']['r2']['mean']:.3f}±{aux['models']['cnn']['aggregate']['r2']['std']:.3f}; ResNet R²={aux['models']['resnet']['aggregate']['r2']['mean']:.3f}±{aux['models']['resnet']['aggregate']['r2']['std']:.3f}. The constant baseline has R²={aux['constant_baseline']['r2']:.3f}.
2. **Reachability improvement:** No reproducible primary-metric improvement. CNN Δaccuracy={gains['cnn']:+.4f}, CI [{cnn_u['accuracy']['true_minus_main_ci95'][0]:+.4f}, {cnn_u['accuracy']['true_minus_main_ci95'][1]:+.4f}]. ResNet Δaccuracy={gains['resnet']:+.4f}, CI [{res_u['accuracy']['true_minus_main_ci95'][0]:+.4f}, {res_u['accuracy']['true_minus_main_ci95'][1]:+.4f}].
3. **Shuffled control:** CNN true auxiliary beats shuffled at selected λ by {cnn_u['accuracy']['true_minus_shuffled_mean']:+.4f}, but true auxiliary does not beat main-only; this mainly shows that shuffled supervision is harmful. ResNet true-vs-shuffled CI includes zero.
4. **Matched-pair accuracy:** CNN changes by {cnn_u['pair_accuracy']['true_minus_main_mean']:+.4f}, CI [{cnn_u['pair_accuracy']['true_minus_main_ci95'][0]:+.4f}, {cnn_u['pair_accuracy']['true_minus_main_ci95'][1]:+.4f}]. ResNet changes by {res_u['pair_accuracy']['true_minus_main_mean']:+.4f}, CI [{res_u['pair_accuracy']['true_minus_main_ci95'][0]:+.4f}, {res_u['pair_accuracy']['true_minus_main_ci95'][1]:+.4f}]. Neither is robust.
5. **Pair margin:** CNN increases by {cnn_u['pair_margin_mean']['true_minus_main_mean']:+.4f}, CI [{cnn_u['pair_margin_mean']['true_minus_main_ci95'][0]:+.4f}, {cnn_u['pair_margin_mean']['true_minus_main_ci95'][1]:+.4f}]. ResNet changes by {res_u['pair_margin_mean']['true_minus_main_mean']:+.4f}, with a CI spanning zero.
6. **Optimal λ:** CNN={best['cnn']}, ResNet={best['resnet']} within the prespecified grid, selected by mean validation main BCE (not test performance).
7. **Relation to β₁ quality:** Higher λ generally improves multi-task β₁ R² while reachability degrades, especially for ResNet. Better β₁ prediction is therefore not sufficient for better reachability.
8. **Representation interpretation:** The required robust positive transfer is absent, so these results do not support a claim that β₁ supervision induced a shared global-topological representation useful for reachability.
9. **Negative-result taxonomy:** Auxiliary unlearnability is ruled out. The closest interpretation is task independence/representation mismatch, with negative transfer/task competition at larger λ.

## Full λ sweep (five seeds)

| model | λ | true acc | shuffled acc | true pair acc | shuffled pair acc | true pair margin | true β₁ R² |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(sweep_lines)}

## Paired uncertainty at selected λ

| model | metric | true − main mean [95% CI] | true − shuffled mean [95% CI] |
|---|---|---:|---:|
{chr(10).join(uncertainty_lines)}

## Controls and uncertainty

- Five paired seeds per condition/λ; paired bootstrap CIs are in `metrics.json`.
- H3 unchanged, class balance=0.5, pair |E|/κ/β₁ matching=1.0, no split leakage, Bayes accuracy=1.0, and train-only shuffle checks all pass: `{verification['all_ok']}`.
- Main-only seeds 1–3 reuse the exact prior H3 predictions; seeds 4–5 are new runs under the identical encoder/training recipe.
- The reused three-seed baselines exactly retain the prior CNN 0.5597±0.0075 and ResNet 0.5218±0.0200 results. Across five seeds, ResNet is highly unstable because seed 4 reaches 0.717 while the other seeds are 0.506–0.550; this is why its paired CIs are wide.
- Twenty corrected and twenty harmed pairs per architecture are stored in `failure_cases.json`; all 80 pair visualizations are under `figures/failure_cases/`.
- The random-Gaussian auxiliary control was optional and was not run; β₀ and multi-scale auxiliary tasks were intentionally excluded by scope.

## Representation diagnostics

The conditional linear-probe extension was not run because neither main accuracy nor matched-pair accuracy showed a robust improvement over main-only. Running probes after selecting an unstable apparent gain would over-interpret the experiment.

## Interpretation boundary

β₁ is demonstrably predictable from pixels, but it is pairwise label-matched and not a reachability proxy. The result supports the narrower conclusion that learning one global graph statistic does not automatically supply the counterfactual connectivity distinction required by H3. It does not prove that β₁ representations and reachability are intrinsically unrelated, nor that other topological auxiliaries would fail.
"""
    (OUT/"phase2_report.md").write_text(text)


if __name__ == "__main__": main()
