# Phase 3 — Random-Pair Connectivity Auxiliary Supervision on H3

## 結論

Phase 3-A の auxiliary accuracy は CNN 0.747±0.004、ResNet 0.687±0.008 で、事前基準 0.65 を両方が上回った。一方、connected pair の graph distance 21 以上では CNN 0.549、ResNet 0.489 まで低下したため、長距離 connectivity の学習は限定的である。

CNN validation pilot により K=16 を選択した。各 architecture の λ は validation main BCE のみで選択し、test accuracy は選択に使用していない。

| model | main acc | λ | true pair acc | shuffled acc | main H3 pair acc | true H3 pair acc | aux acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.560±0.006 | 1.0 | 0.590±0.007 | 0.551±0.009 | 0.130 | 0.188 | 0.746 |
| RESNET | 0.558±0.081 | 0.3 | 0.700±0.033 | 0.583±0.100 | 0.147 | 0.411 | 0.683 |

### Paired-seed evidence

- **CNN**: true−main Δaccuracy=+0.0303 (paired bootstrap 95% CI +0.0246, +0.0360); true−shuffled=+0.0389 (95% CI +0.0333, +0.0455). Δmatched-pair accuracy=+0.0582, Δmean margin=+0.0232. High-performing seed rate (>0.65): main 0/5, true pair 0/5.
- **RESNET**: true−main Δaccuracy=+0.1421 (paired bootstrap 95% CI +0.0536, +0.2096); true−shuffled=+0.1170 (95% CI +0.0414, +0.1797). Δmatched-pair accuracy=+0.2644, Δmean margin=+0.2107. High-performing seed rate (>0.65): main 1/5, true pair 5/5.

## β₁ と relational connectivity の直接比較

| condition | auxiliary learnability | main ΔAccuracy | H3 pair ΔAccuracy |
|---|---:|---:|---:|
| CNN β₁ | 0.962 R² | -0.0008 | -0.0014 |
| CNN random-pair | 0.746 acc | +0.0303 | +0.0582 |
| RESNET β₁ | 0.914 R² | +0.0132 | +0.0224 |
| RESNET random-pair | 0.683 acc | +0.1421 | +0.2644 |

## Q1–Q8

1. **Q1 — learnability:** random-pair connectivity は平均では learnable（両モデル >0.65）だが、long connected paths では near-chance であり、完全な algorithmic solution ではない。
2. **Q2 — main transfer:** **Case A** だった。true pair は main-only に対し CNN +0.0303、ResNet +0.1421 accuracy を改善し、両方の paired bootstrap 95% CI が 0 を上回った。
3. **Q3 — true vs shuffled:** true pair は shuffled に対し CNN +0.0389、ResNet +0.1170 accuracy が高く、両 CI が 0 を上回った。したがって generic regularization だけでは説明しにくい。
4. **Q4 — matched-pair accuracy:** true pair は main-only に対して CNN +0.0582、ResNet +0.2644 改善した。ordinary accuracy と同方向である。
5. **Q5 — pair margin:** mean margin は main-only に対して CNN +0.0232、ResNet +0.2107 改善した（Figure 4）。mean / median / std は `metrics.json` に保存した。
6. **Q6 — K:** CNN pilot は K=1: 0.567±0.009, K=4: 0.559±0.013, K=8: 0.549±0.011, K=16: 0.564±0.002, K=32: 0.564±0.004。K は validation BCE で選択し、単調増加の主張は test accuracy の並びだけからは行わない。
7. **Q7 — high-performing basin:** CNN は main 0/5、true pair 0/5 のままだったが、ResNet は main 1/5 から true pair 5/5（shuffled 1/5）へ増えた。これは exploratory metric であり post-hoc significance claim はしない。
8. **Q8 — representation:** seed 1 の frozen encoder では、random-pair training は CNN の reachability probe を 0.569→0.602、pair probe を 0.658→0.748、ResNet では 0.552→0.716、0.614→0.683 に改善した。一方 β₁ training は主に β₁ probe を強めた。relational alignment と整合するが、単一 seed の記述的結果であり機構の同定ではない。

| encoder | training | β₁ probe R² | reachability probe acc | random-pair probe acc |
|---|---|---:|---:|---:|
| CNN | main | 0.651 | 0.569 | 0.658 |
| CNN | beta1 | 0.945 | 0.560 | 0.667 |
| CNN | random_pair | 0.704 | 0.602 | 0.748 |
| RESNET | main | 0.275 | 0.551 | 0.614 |
| RESNET | beta1 | 0.751 | 0.559 | 0.630 |
| RESNET | random_pair | 0.357 | 0.716 | 0.683 |

## Scope, controls, and verification

- Five paired seeds for the full λ sweep; CNN K pilot used three seeds.
- H3 and its matched-pair constraints remain unchanged; verification all_ok=True.
- Fixed val/test auxiliary pairs are balanced, exclude start/goal, and have exact component-label targets. Training pairs are dynamic and epoch-seeded; shuffled targets are train-only.
- Failure cases contain 40 corrected and 40 harmed H3 pairs.
- The optional random-marker control, source-pair extension, start-goal positive control, and connected-component-membership probe were not run.

以上は、この H3・5 seed の範囲で relational auxiliary supervision が scalar β₁ より transfer しやすいことを支持する。ただし ResNet の seed 間分散、long-distance auxiliary accuracy の低下、未実施 control が残るため、algorithmic path reasoning や因果機構の証明とはしない。

詳細値は `metrics.json`、`representation_probes.json`、Figures 1–7 に保存した。観測された transfer と、そこからの機構的解釈は区別する。
