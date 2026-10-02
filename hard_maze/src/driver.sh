#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PY=${PYTHON:-"$ROOT/../.venv/bin/python"}

for level in H0 H1 H2 H3; do
  for model in cnn resnet; do
    for seed in 1 2 3; do
      pred="$ROOT/results_hard_maze/pred_${level}_${model}_s${seed}.npz"
      while [ ! -f "$pred" ]; do
        "$PY" "$ROOT/src/train.py" "$level" "$model" "$seed" --budget 150
      done
    done
  done
done
