# Phase 2-A — β₁ Auxiliary-only Report

## Decision

Best model: **CNN**, test R² = **0.962**.  
Decision: **PROCEED to Phase 2-B**.

## Distribution and leakage check

- Train β₁: min=22, max=75, mean=47.750, std=7.420, unique=49.
- Pairwise exact match: train/val/test = 1.000/1.000/1.000.
- Correlation with reachability: 0.000; discrete MI=0.000000 nats.
- β₁-only logistic reachability accuracy: 0.500.
- β₁ spot recalculation errors: 0.

## Regression results (3 seeds)

| model | R² | RMSE | MAE | Pearson | Spearman | rounded acc |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 0.962 ± 0.003 | 1.503 | 1.191 | 0.982 | 0.979 | 0.269 |
| RESNET | 0.914 ± 0.005 | 2.272 | 1.810 | 0.957 | 0.953 | 0.175 |


Constant baseline R² = -0.002. The scalar identity using edge count and component count has R²=1 by definition and is reference-only.

## Interpretation

β₁ is reproducibly learnable from the image and exceeds the constant baseline, so the predefined gate permits the full multi-task experiment.
