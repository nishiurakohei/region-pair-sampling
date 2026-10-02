# Phase 2 — Topological Auxiliary Supervision on H3

## Conclusion

Phase 2-A establishes a clear asymmetry: β₁ is strongly learnable (CNN R²=0.962, ResNet R²=0.914) while H3 reachability remains difficult.  
Phase 2-B does **not** show a robust accuracy or matched-pair-accuracy improvement. At validation-selected λ, CNN changes by -0.0008 accuracy and ResNet by +0.0132; both paired 95% CIs include zero. CNN pair margin does shift positively, but this does not translate into more threshold-correct pairs.

| model | main acc | selected λ | true aux acc | shuffled acc | main pair acc | true pair acc | true β₁ R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.560 ± 0.006 | 0.03 | 0.559 ± 0.007 | 0.546 ± 0.004 | 0.130 | 0.129 | 0.922 |
| RESNET | 0.558 ± 0.081 | 0.01 | 0.571 ± 0.051 | 0.547 ± 0.057 | 0.147 | 0.169 | 0.549 |

λ was selected independently for each architecture by the lowest mean per-seed best validation main BCE; test accuracy was not used for selection.

## Answers

1. **β₁ learnability:** Yes. Auxiliary-only CNN R²=0.962±0.003; ResNet R²=0.914±0.005. The constant baseline has R²=-0.002.
2. **Reachability improvement:** No reproducible primary-metric improvement. CNN Δaccuracy=-0.0008, CI [-0.0092, +0.0076]. ResNet Δaccuracy=+0.0132, CI [-0.0715, +0.0923].
3. **Shuffled control:** CNN true auxiliary beats shuffled at selected λ by +0.0131, but true auxiliary does not beat main-only; this mainly shows that shuffled supervision is harmful. ResNet true-vs-shuffled CI includes zero.
4. **Matched-pair accuracy:** CNN changes by -0.0014, CI [-0.0162, +0.0134]. ResNet changes by +0.0224, CI [-0.1300, +0.1696]. Neither is robust.
5. **Pair margin:** CNN increases by +0.0031, CI [+0.0001, +0.0062]. ResNet changes by -0.0054, with a CI spanning zero.
6. **Optimal λ:** CNN=0.03, ResNet=0.01 within the prespecified grid, selected by mean validation main BCE (not test performance).
7. **Relation to β₁ quality:** Higher λ generally improves multi-task β₁ R² while reachability degrades, especially for ResNet. Better β₁ prediction is therefore not sufficient for better reachability.
8. **Representation interpretation:** The required robust positive transfer is absent, so these results do not support a claim that β₁ supervision induced a shared global-topological representation useful for reachability.
9. **Negative-result taxonomy:** Auxiliary unlearnability is ruled out. The closest interpretation is task independence/representation mismatch, with negative transfer/task competition at larger λ.

## Full λ sweep (five seeds)

| model | λ | true acc | shuffled acc | true pair acc | shuffled pair acc | true pair margin | true β₁ R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.01 | 0.555 | 0.554 | 0.125 | 0.121 | 0.0149 | 0.796 |
| CNN | 0.03 | 0.559 | 0.546 | 0.129 | 0.106 | 0.0185 | 0.922 |
| CNN | 0.1 | 0.550 | 0.526 | 0.113 | 0.074 | 0.0148 | 0.949 |
| CNN | 0.3 | 0.550 | 0.517 | 0.111 | 0.060 | 0.0103 | 0.954 |
| CNN | 1.0 | 0.536 | 0.508 | 0.089 | 0.047 | 0.0060 | 0.952 |
| RESNET | 0.01 | 0.571 | 0.547 | 0.169 | 0.125 | 0.0662 | 0.549 |
| RESNET | 0.03 | 0.534 | 0.510 | 0.093 | 0.059 | 0.0227 | 0.804 |
| RESNET | 0.1 | 0.515 | 0.506 | 0.060 | 0.058 | 0.0032 | 0.873 |
| RESNET | 0.3 | 0.512 | 0.506 | 0.059 | 0.069 | 0.0031 | 0.865 |
| RESNET | 1.0 | 0.512 | 0.504 | 0.061 | 0.074 | 0.0031 | 0.850 |

## Paired uncertainty at selected λ

| model | metric | true − main mean [95% CI] | true − shuffled mean [95% CI] |
|---|---|---:|---:|
| CNN | accuracy | -0.0008 [-0.0092, +0.0076] | +0.0131 [+0.0073, +0.0194] |
| CNN | pair_accuracy | -0.0014 [-0.0162, +0.0134] | +0.0230 [+0.0120, +0.0356] |
| CNN | pair_margin_mean | +0.0031 [+0.0001, +0.0062] | +0.0050 [+0.0034, +0.0064] |
| RESNET | accuracy | +0.0132 [-0.0715, +0.0923] | +0.0245 [-0.0357, +0.0965] |
| RESNET | pair_accuracy | +0.0224 [-0.1300, +0.1696] | +0.0444 [-0.0684, +0.1798] |
| RESNET | pair_margin_mean | -0.0054 [-0.1499, +0.1157] | +0.0273 [-0.0482, +0.1144] |

## Controls and uncertainty

- Five paired seeds per condition/λ; paired bootstrap CIs are in `metrics.json`.
- H3 unchanged, class balance=0.5, pair |E|/κ/β₁ matching=1.0, no split leakage, Bayes accuracy=1.0, and train-only shuffle checks all pass: `True`.
- Main-only seeds 1–3 reuse the exact prior H3 predictions; seeds 4–5 are new runs under the identical encoder/training recipe.
- The reused three-seed baselines exactly retain the prior CNN 0.5597±0.0075 and ResNet 0.5218±0.0200 results. Across five seeds, ResNet is highly unstable because seed 4 reaches 0.717 while the other seeds are 0.506–0.550; this is why its paired CIs are wide.
- Twenty corrected and twenty harmed pairs per architecture are stored in `failure_cases.json`; all 80 pair visualizations are under `figures/failure_cases/`.
- The random-Gaussian auxiliary control was optional and was not run; β₀ and multi-scale auxiliary tasks were intentionally excluded by scope.

## Representation diagnostics

The conditional linear-probe extension was not run because neither main accuracy nor matched-pair accuracy showed a robust improvement over main-only. Running probes after selecting an unstable apparent gain would over-interpret the experiment.

## Interpretation boundary

β₁ is demonstrably predictable from pixels, but it is pairwise label-matched and not a reachability proxy. The result supports the narrower conclusion that learning one global graph statistic does not automatically supply the counterfactual connectivity distinction required by H3. It does not prove that β₁ representations and reachability are intrinsically unrelated, nor that other topological auxiliaries would fail.
