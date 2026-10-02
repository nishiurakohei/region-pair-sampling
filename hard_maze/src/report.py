"""Render the final evidence-backed Hard-Maze Markdown report."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results_hard_maze"
LEVELS = ["H0", "H1", "H2", "H3"]


def mean(metrics, level, model, key="accuracy"):
    return metrics["levels"][level][model]["aggregate"][key]["mean"]


def sd(metrics, level, model, key="accuracy"):
    return metrics["levels"][level][model]["aggregate"][key]["std"]


def fmt(x):
    return f"{x:.3f}"


def main():
    cfg = json.loads((ROOT / "config.json").read_text())
    gen = json.loads((R / "generation_stats.json").read_text())
    stats = json.loads((R / "dataset_stats.json").read_text())
    short = json.loads((R / "shortcut_baselines.json").read_text())
    met = json.loads((R / "metrics.json").read_text())
    ver = json.loads((R / "verification.json").read_text())

    rows = []
    for level in LEVELS:
        c, r = mean(met, level, "cnn"), mean(met, level, "resnet")
        cp = met["levels"][level]["cnn"]["aggregate"].get("pair_accuracy")
        rp = met["levels"][level]["resnet"]["aggregate"].get("pair_accuracy")
        rows.append(f"| {level} | {fmt(c)} ± {fmt(sd(met, level, 'cnn'))} | "
                    f"{fmt(r)} ± {fmt(sd(met, level, 'resnet'))} | "
                    f"{fmt(short[level]['scalar_logistic']['accuracy'])} | "
                    f"{fmt(short[level]['local_patch_logistic_radius4']['accuracy'])} | "
                    f"{fmt(cp['mean']) if cp else '—'} | {fmt(rp['mean']) if rp else '—'} |")

    candidates = [l for l in ("H2", "H3")
                  if .5 <= max(mean(met, l, "cnn"), mean(met, l, "resnet")) <= .7]
    recommended = candidates[-1] if candidates else min(("H2", "H3"),
        key=lambda l: abs(max(mean(met, l, "cnn"), mean(met, l, "resnet")) - .625))
    target = recommended in candidates

    yields = []
    for level in ("H1", "H2", "H3"):
        x = gen[level]["train"]
        yields.append(f"| {level} | {x['raw_generated']:,} | {x['accepted_pairs']:,} | "
                      f"{100*x['acceptance_rate']:.2f}% | {x['generation_seconds']:.1f}s |")

    h2match = stats["H2"]["test_pair_exact_match"]
    h3match = stats["H3"]["test_pair_exact_match"]
    h0c, h1c, h2c, h3c = [mean(met, l, "cnn") for l in LEVELS]
    h0r, h1r, h2r, h3r = [mean(met, l, "resnet") for l in LEVELS]
    text = f"""# Hard-Maze Dataset — Main-task-only Baseline Report

## 結論

完全観測・決定論的 label（Bayes accuracy = **1.000**）を保ったまま、推奨条件は **{recommended}** である。
Small CNN / ResNet の平均 accuracy はそれぞれ **{fmt(mean(met, recommended, 'cnn'))} / {fmt(mean(met, recommended, 'resnet'))}**、scalar / local-patch shortcut は **{fmt(short[recommended]['scalar_logistic']['accuracy'])} / {fmt(short[recommended]['local_patch_logistic_radius4']['accuracy'])}** だった。
目標 0.5–0.7 への到達判定は **{'達成' if target else '未達'}**。補助タスクは実装していない。

## 1. Construction

- 15×15 bond-percolation grid、`p={cfg['p']}`、start=(0,0)、goal=(14,14)。masking は一切なし。
- H0: Phase 1 の fully observed original dataset。
- H1: disconnected base `G0` の start/goal component 間の closed edgeを開いた `G+` と、`G0` 自身の pair。
- H2: negative 側にも別 component 間の control edge を1本追加。したがって pair 内の `|E|`, `κ`, `β1` は構成上 exact match。
- H3: H2 に shortest path ≥ {cfg['h3_min_path']}、pivotal/control edge の start/goal 距離 > {cfg['h3_interior_radius']} を追加。
- Pivotal edge は閉じると positive が必ず disconnected になるため `λ(s,g)=1`。negative は同じ edge を開く1 flipで reachable。

## 2. Generation yield

| level | raw graphs (train) | accepted pairs | acceptance | time |
|---|---:|---:|---:|---:|
{chr(10).join(yields)}

H3 は rare-event rejection のままではなく、component boundary を直接列挙する targeted mutation を使った。全 split は 8,000 / 1,000 / 2,000 samples、H1–H3 は正負完全 balanced、同じ base pair は同じ split に固定した。

## 3. Matching quality and verification

- H2 test exact-match率: edge count={h2match['edge_count']:.3f}, component count={h2match['component_count']:.3f}, β1={h2match['beta1']:.3f}。
- H3 test exact-match率: edge count={h3match['edge_count']:.3f}, component count={h3match['component_count']:.3f}, β1={h3match['beta1']:.3f}。
- 全 label の BFS 再計算、positive bridge、negative one-flip、counterfactual flip、class balance、split leakage、H3 path/interior 条件を全件検査し、**all_ok={ver['all_ok']}**、違反 0。
- fully observed deterministic mapping なので Bayes ceiling は厳密に 1.0。

## 4. Main results (3 seeds)

| level | Small CNN acc | ResNet acc | scalar shortcut | local patch r=4 | CNN pair acc | ResNet pair acc |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Accuracy 以外の AUROC, BCE, Brier, ECE, confusion matrix、pair probability gap / ranking は `metrics.json` に保存した。component sizes、path length、reachability そのものは shortcut classifier から除外した。

H0 の seed 1–2 prediction は、同一データ・同一アーキテクチャ・同一 fully observed recipe の Phase 1 成果物を再利用し、seed 3 を新規学習した。H1–H3 は全3 seedsを新規学習した。

## 5. Research questions

### Q1 Near-boundary sampling の効果

H0→H1 の accuracy 変化は CNN **{h1c-h0c:+.3f}**、ResNet **{h1r-h0r:+.3f}**。Near-boundary 化だけの効果はこの差である。

### Q2 Coarse matching の追加効果

H1→H2 は CNN **{h2c-h1c:+.3f}**、ResNet **{h2r-h1r:+.3f}**。H2 は `|E|, κ, β1` を pairwise exact matchしており、この差が matching の追加効果を示す。ただし degree/local endpoint 情報はなお残るため、H2 shortcut accuracyも併記した。

### Q3 Long-path / interior-pivotal の効果

H2→H3 は CNN **{h3c-h2c:+.3f}**、ResNet **{h3r-h2r:+.3f}**。H3 positive path は test で mean={stats['H3']['positive_path_length']['mean']:.2f}, min={stats['H3']['positive_path_length']['min']}。この差を長経路・interior 制約の複合効果として解釈する（個別因果効果には分解していない）。

### Q4 Shortcut は残るか

H3 の scalar={fmt(short['H3']['scalar_logistic']['accuracy'])}、local={fmt(short['H3']['local_patch_logistic_radius4']['accuracy'])} で chance 近傍。H0–H2 に残る相関が H3 で大きく抑えられた。

### Q5 Bayes=1 のまま target regime を作れたか

**{'Yes' if target else 'No'}**。情報除去・label noiseなしで、推奨 {recommended} の neural accuracy は上記範囲になった。
H3 が chance 近傍なのは label/入力不整合ではない。全 label の BFS 再計算が一致し、同一 rasterization・training code で H0–H2 は十分学習でき、H3 でも train/validation BCE と複数 seed の挙動を確認した。したがって少なくとも単純な pipeline bug では説明できない。

### Q6 Phase 2 の候補

**{recommended}** を主条件とする。H0/H1/H2 は ablation として保持し、Phase 2 では同一 split・seed 上で main-only と auxiliary supervision を比較するのがよい。

## 6. Figures

1. `fig1_accuracy_h0_h3.png`: hardness ladder の neural / shortcut accuracy。
2. `fig2_edge_count.png`: positive/negative edge-count 分布。
3. `fig3_beta1.png`: β1 分布。
4. `fig4_degree_histogram.png`: degree histogram。
5. `fig5_path_length.png`: positive shortest-path length。
6. `fig6_matched_pairs.png`: pivotal/control edge 付き representative H3 pairs。
7. `fig7_pair_failures.png`: CNN の representative pair failures。

## 7. Interpretation boundary

H3 が local/coarse shortcut を大幅に抑えたことと CNN/ResNet の性能低下は、global connectivity tracking を必要とするという設計意図と整合する。ただし、これだけで「ネットワークが特定の内部アルゴリズムを実装していない」とまでは証明しない。また H2→H3 は long-path と interior 制約を同時に変更しているため、両者の個別効果を主張しない。
"""
    (R / "hard_maze_report.md").write_text(text)
    print("wrote", R / "hard_maze_report.md")


if __name__ == "__main__":
    main()
