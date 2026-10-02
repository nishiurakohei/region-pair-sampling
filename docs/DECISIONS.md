# Research decisions

## Decision 1 — suspend scratch Stage 2

Scratch multi-task Stage 2 is suspended until full official Gate 0.5 passes. A main-only model that has not learned the ID task cannot support an interpretable auxiliary-transfer comparison.

## Decision 2 — Stage 1 does not compensate for Stage 0.5

Stage 1 establishes auxiliary-only learnability (Case B), not main-task training reproducibility or transfer. Its success does not override the failed Stage 0.5 gate.

## Decision 3 — use the available high-memory GPU for full Gate 0.5

A high-memory GPU machine is available. The immediate priority is the recovered full official condition: first one scratch seed, then at least three seeds only if the first reaches validation/ID exact accuracy ≥0.95.

## Decision 4 — retain pretrained fine-tuning as fallback

Pretrained fine-tuning remains a fallback if the authoritative full scratch condition cannot be reproduced. It is not the next experiment because it would answer a different question and weaken the main-only control.

## Interpretation boundary

The completed 5k/10-epoch run is a negative result for that resource-bounded condition only. It is not evidence that the authors' 40k/50-epoch pipeline fails. OOD curves remain evaluation-only and must not select training hyperparameters or checkpoints.
