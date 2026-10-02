# Maze auxiliary-task research context

## Research question

Can auxiliary supervision that matches graph-geometric reasoning improve image-based reachability, and does the effect transfer from a controlled synthetic benchmark to the Knutson et al. logical-extrapolation maze benchmark?

The current evidence separates three claims: an auxiliary target can be learnable, it can improve the controlled H3 main task, and it can transfer to an external DT-Net benchmark. The first two have evidence; the third has not yet been tested because scratch DT-Net reproduction has not passed its gate.

## Controlled hard-maze benchmark (H3)

H3 is a controlled binary reachability benchmark constructed so that local and scalar shortcuts are weak. CNN and ResNet baselines remain near chance, while the exact BFS labeler and Bayes oracle achieve accuracy 1.0. The canonical report and metrics are:

- `hard_maze/results_hard_maze/hard_maze_report.md`
- `hard_maze/results_hard_maze/metrics.json`
- `hard_maze/results_hard_maze/verification.json`

## β₁ auxiliary supervision

β₁ is highly learnable from the raster input: auxiliary-only R² was 0.962 for CNN and 0.914 for ResNet. However, five paired seeds, true/shuffled controls, and validation-selected auxiliary weights did not show a robust main-task improvement. This is a negative transfer result, not an auxiliary-unlearnability result. Large auxiliary weights also showed task competition.

- `hard_maze/results_phase2/aux_only/aux_only_report.md`
- `hard_maze/results_phase2/multitask/phase2_report.md`
- `hard_maze/results_phase2/multitask/metrics.json`

## Random-pair connectivity supervision

Random-pair connectivity uses graph relations closer to the main reachability question. On H3, true auxiliary supervision improved accuracy approximately:

- CNN: 0.560 → 0.590
- ResNet: 0.558 → 0.700

The true auxiliary outperformed shuffled-label controls under paired-seed analysis. Long-distance pairs remained substantially weaker, so the result supports relational transfer but does not establish general long-range algorithmic reasoning.

- `hard_maze/results_phase3/multitask/phase3_report.md`
- `hard_maze/results_phase3/multitask/metrics.json`
- `hard_maze/results_phase3/multitask/representation_probes.json`

## Why external validation

The controlled H3 result may depend on the local architecture, data generator, or task construction. External validation therefore moved to the Knutson et al. maze logical-extrapolation benchmark and its recurrent DT-Net. This benchmark provides a pretrained reference with known percolation, endpoint, and maze-size failure modes. The plan is to reproduce the main-only scratch pipeline first, validate random-pair distance learnability independently, and only then test multi-task transfer.

## Current external-validation state

- Stage 0 pretrained reproduction: **PASS**.
- Stage 0.5 scratch reproduction: **INCONCLUSIVE / GATE NOT PASSED** under a resource-bounded run; the official full condition is untested.
- Stage 1 shortest-path auxiliary learnability: **PASS / Case B**, with long-distance weakness.
- Stage 2 multi-task transfer: **SUSPENDED** until full scratch Gate 0.5 passes.

See `docs/EXPERIMENT_LOG.md` for exact numbers and `docs/DECISIONS.md` for the go/no-go logic.
