# Experiment log

This file is the compact index. Reports and machine-readable results remain at their original paths so existing references do not break.

## Controlled benchmark

| Stage | Status | Main result | Config | Report / metrics |
|---|---|---|---|---|
| H3 baseline | Complete | CNN/ResNet near chance; BFS and Bayes oracle = 1.0 | `hard_maze/config.json` | `hard_maze/results_hard_maze/hard_maze_report.md` |
| Phase 2 β₁ | Negative transfer | β₁ learnable, no robust main-task gain | `hard_maze/phase2/config.json` | `hard_maze/results_phase2/multitask/phase2_report.md` |
| Phase 3 connectivity | Positive transfer with limitation | CNN ≈0.560→0.590; ResNet ≈0.558→0.700; long-distance pairs weak | `hard_maze/phase3/config.json` | `hard_maze/results_phase3/multitask/phase3_report.md` |

## External Stage 0 — pretrained DT-Net

**Status: PASS**

- Official repository: `mines-opt-ml/maze-extrapolation`
- Commit: `6d09263fdb8d61274e4b0e046e07ed929b35af12`
- Pretrained weight SHA-256: `b8beed72aad07b300e178f74f0c3628886cfb28fca5d271c451496af38478216`
- Critical dependencies: `maze-dataset==1.1.0`, `muutils==0.6.21`
- `maze-dataset==1.4.2` produced 0/10 ID accuracy with the same weight and is incompatible with this raster pipeline.

Five-dataset-seed results:

| Condition | Exact accuracy |
|---|---:|
| 9×9 ID, p=0, deadend=true | 1.000 |
| deadend=false | 0.770 |
| p=0.005 | 0.914 |
| p=0.010 | 0.854 |
| p=0.020 | 0.734 |
| p=0.030 | 0.606 |
| p=0.040 | 0.542 |

Larger mazes degrade at fixed recurrence depth and recover when test-time iterations increase. The full tail reached qualitative solving behavior through 99×99, with fewer mazes per large-size point.

- Configs: `hard_maze/external_validation/configs/stage0_screen.json`, `stage0_full_tail.json`
- Report: `hard_maze/results_external/stage0_reproduction/reproduction_report.md`
- Metrics: `hard_maze/results_external/stage0_reproduction/metrics.json`
- Verification: `hard_maze/results_external/stage0_reproduction/verification.json`

## External Stage 0.5 — scratch DT-Net

**Status: INCONCLUSIVE / GATE NOT PASSED**

The completed resource-bounded condition used 5,000 training mazes, 1,000 validation mazes, 10 epochs, two warmup epochs, and three independent model seeds. Mean 9×9 ID exact accuracy was approximately 0.003. All qualitative Gate 0.5 criteria failed because the models did not reach a solver regime.

This does **not** establish failure of the official full pipeline. The recovered official configuration is:

- 40,000 train / 10,000 validation
- 50 epochs; 10-epoch exponential warmup
- DT-Net width 128; 30 recurrent steps
- progressive loss α=0.01
- Adam; base LR 1e-3; recurrent LR 1e-3 / 30
- weight decay 2e-4; batch size 50
- training source: `aks2203/deep-thinking@152fb1182fb5dc11275859407ceec46102c20e42`

- Official config: `hard_maze/external_validation/configs/stage0_5_training.json`
- Resource config: `hard_maze/external_validation/configs/stage0_5_resource.json`
- Report: `hard_maze/results_external/stage0_5_training/stage0_5_report.md`
- Metrics / verification: `hard_maze/results_external/stage0_5_training/metrics.json`, `verification.json`

## External Stage 1 — random-pair shortest-path distance

**Status: PASS / Case B**

Design: Easy-to-Hard 9×9 mazes; 16 fixed pairs per maze; pair vertices exclude start and goal; Manhattan distance ≥3; target is exact BFS graph distance / 9. The start/goal markers are removed to form a graph-only raster. DT-Net computes H(G) once per maze, and the pair head receives `h_u`, `h_v`, `|h_u-h_v|`, `h_u*h_v`, and a global feature.

Resource-bounded training used 5,000 mazes, 10 epochs, and model seed 401.

| Metric | Model | Linear Manhattan baseline |
|---|---:|---:|
| Raw MAE, 30 iterations | 1.221 | 3.083 |
| R², 30 iterations | 0.825 | 0.054 |
| Pearson, 30 iterations | 0.909 | — |
| 16+ bin MAE | 3.098 | 9.033 |
| 16+ bin R² | -5.978 | — |
| 16+ bin Pearson | 0.092 | — |

Iteration diagnostic: MAE 2.206 at 10 iterations, 1.221 at 30, and 1.265 at 100. The task is learnable overall, but long-distance generalization remains weak.

- Config: `hard_maze/external_validation/configs/stage1_aux.json`
- Report: `hard_maze/results_external/stage1_aux_learnability/aux_learnability_report.md`
- Metrics / verification: `hard_maze/results_external/stage1_aux_learnability/metrics.json`, `verification.json`

## Next log entry

The next experiment is one full official scratch seed on the GPU machine. Do not add a Stage 2 entry unless Gate 0.5 passes. Exact instructions are in `docs/NEXT_TASK.md`.
