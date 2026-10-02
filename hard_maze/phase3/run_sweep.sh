#!/bin/sh
set -eu

cd "$(dirname "$0")/../.."

while [ "$(find hard_maze/results_phase3/multitask -maxdepth 1 -name 'pred_*_*_l*_k16_s*.npz' | wc -l | tr -d ' ')" -lt 100 ]; do
  for model in cnn resnet; do
    for condition in true shuffled; do
      for lambda in 0.01 0.03 0.1 0.3 1.0; do
        for seed in 1 2 3 4 5; do
          printf '%s %s %s %s\n' "$model" "$condition" "$lambda" "$seed"
        done
      done
    done
  done | xargs -n4 -P6 sh -c \
    '.venv/bin/python hard_maze/phase3/train.py multi "$0" "$1" "$2" 16 "$3" --budget 10000'
done
