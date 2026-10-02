# Hard-Maze Dataset — Main-task-only Baseline Report

## 結論

完全観測・決定論的 label（Bayes accuracy = **1.000**）を保ったまま、推奨条件は **H3** である。
Small CNN / ResNet の平均 accuracy はそれぞれ **0.560 / 0.522**、scalar / local-patch shortcut は **0.522 / 0.527** だった。
目標 0.5–0.7 への到達判定は **達成**。補助タスクは実装していない。

## 1. Construction

- 15×15 bond-percolation grid、`p=0.64`、start=(0,0)、goal=(14,14)。masking は一切なし。
- H0: Phase 1 の fully observed original dataset。
- H1: disconnected base `G0` の start/goal component 間の closed edgeを開いた `G+` と、`G0` 自身の pair。
- H2: negative 側にも別 component 間の control edge を1本追加。したがって pair 内の `|E|`, `κ`, `β1` は構成上 exact match。
- H3: H2 に shortest path ≥ 34、pivotal/control edge の start/goal 距離 > 3 を追加。
- Pivotal edge は閉じると positive が必ず disconnected になるため `λ(s,g)=1`。negative は同じ edge を開く1 flipで reachable。

## 2. Generation yield

| level | raw graphs (train) | accepted pairs | acceptance | time |
|---|---:|---:|---:|---:|
| H1 | 10,574 | 4,000 | 37.83% | 6.6s |
| H2 | 10,431 | 4,000 | 38.35% | 7.0s |
| H3 | 169,707 | 4,000 | 2.36% | 60.2s |

H3 は rare-event rejection のままではなく、component boundary を直接列挙する targeted mutation を使った。全 split は 8,000 / 1,000 / 2,000 samples、H1–H3 は正負完全 balanced、同じ base pair は同じ split に固定した。

## 3. Matching quality and verification

- H2 test exact-match率: edge count=1.000, component count=1.000, β1=1.000。
- H3 test exact-match率: edge count=1.000, component count=1.000, β1=1.000。
- 全 label の BFS 再計算、positive bridge、negative one-flip、counterfactual flip、class balance、split leakage、H3 path/interior 条件を全件検査し、**all_ok=True**、違反 0。
- fully observed deterministic mapping なので Bayes ceiling は厳密に 1.0。

## 4. Main results (3 seeds)

| level | Small CNN acc | ResNet acc | scalar shortcut | local patch r=4 | CNN pair acc | ResNet pair acc |
|---|---:|---:|---:|---:|---:|---:|
| H0 | 0.830 ± 0.008 | 0.821 ± 0.018 | 0.765 | 0.747 | — | — |
| H1 | 0.841 ± 0.006 | 0.920 ± 0.002 | 0.651 | 0.631 | 0.683 | 0.840 |
| H2 | 0.836 ± 0.008 | 0.911 ± 0.007 | 0.624 | 0.636 | 0.678 | 0.824 |
| H3 | 0.560 ± 0.008 | 0.522 ± 0.020 | 0.522 | 0.527 | 0.130 | 0.077 |

Accuracy 以外の AUROC, BCE, Brier, ECE, confusion matrix、pair probability gap / ranking は `metrics.json` に保存した。component sizes、path length、reachability そのものは shortcut classifier から除外した。

H0 の seed 1–2 prediction は、同一データ・同一アーキテクチャ・同一 fully observed recipe の Phase 1 成果物を再利用し、seed 3 を新規学習した。H1–H3 は全3 seedsを新規学習した。

## 5. Research questions

### Q1 Near-boundary sampling の効果

H0→H1 の accuracy 変化は CNN **+0.012**、ResNet **+0.099**。Near-boundary 化だけの効果はこの差である。

### Q2 Coarse matching の追加効果

H1→H2 は CNN **-0.005**、ResNet **-0.009**。H2 は `|E|, κ, β1` を pairwise exact matchしており、この差が matching の追加効果を示す。ただし degree/local endpoint 情報はなお残るため、H2 shortcut accuracyも併記した。

### Q3 Long-path / interior-pivotal の効果

H2→H3 は CNN **-0.276**、ResNet **-0.389**。H3 positive path は test で mean=35.65, min=34。この差を長経路・interior 制約の複合効果として解釈する（個別因果効果には分解していない）。

### Q4 Shortcut は残るか

H3 の scalar=0.522、local=0.527 で chance 近傍。H0–H2 に残る相関が H3 で大きく抑えられた。

### Q5 Bayes=1 のまま target regime を作れたか

**Yes**。情報除去・label noiseなしで、推奨 H3 の neural accuracy は上記範囲になった。
H3 が chance 近傍なのは label/入力不整合ではない。全 label の BFS 再計算が一致し、同一 rasterization・training code で H0–H2 は十分学習でき、H3 でも train/validation BCE と複数 seed の挙動を確認した。したがって少なくとも単純な pipeline bug では説明できない。

### Q6 Phase 2 の候補

**H3** を主条件とする。H0/H1/H2 は ablation として保持し、Phase 2 では同一 split・seed 上で main-only と auxiliary supervision を比較するのがよい。

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
