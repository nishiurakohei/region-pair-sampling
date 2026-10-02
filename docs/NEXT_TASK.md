# Next task — GPU Gate 0.5

Do not start Stage 2. Run the steps below in order and update `docs/EXPERIMENT_LOG.md` afterward.

## 1. Clone this handoff branch and create the environment

Use Python 3.11 and a CUDA-capable PyTorch 2.8.0 wheel. Select the CUDA index matching the machine's driver; the example uses CUDA 12.8.

```bash
git clone https://github.com/nishiurakohei/region-pair-sampling.git
cd region-pair-sampling
git checkout codex/handoff-gate0-5-full-gpu

python3.11 -m venv .venv-gpu
.venv-gpu/bin/python -m pip install --upgrade pip
.venv-gpu/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0
.venv-gpu/bin/python -m pip install -r hard_maze/external_validation/requirements-gpu.txt
```

Do not upgrade `maze-dataset` or `muutils`.

## 2. Restore pinned upstream sources

```bash
mkdir -p hard_maze/external_validation/upstream
git clone https://github.com/mines-opt-ml/maze-extrapolation.git hard_maze/external_validation/upstream/maze-extrapolation
git -C hard_maze/external_validation/upstream/maze-extrapolation checkout 6d09263fdb8d61274e4b0e046e07ed929b35af12
git clone https://github.com/aks2203/deep-thinking.git hard_maze/external_validation/upstream/deep-thinking
git -C hard_maze/external_validation/upstream/deep-thinking checkout 152fb1182fb5dc11275859407ceec46102c20e42
```

Confirm that `hard_maze/external_validation/upstream/maze-extrapolation/models/dt_net.pth` has SHA-256 `b8beed72aad07b300e178f74f0c3628886cfb28fca5d271c451496af38478216`.

## 3. Restore Easy-to-Hard 9×9 training data

```bash
mkdir -p hard_maze/external_validation/data/easy-to-hard-data
curl -L https://cs.umd.edu/~tomg/download/Easy_to_Hard_Datav2/maze_data_train_9.tar.gz -o hard_maze/external_validation/data/easy-to-hard-data/maze_data_train_9.tar.gz
echo '1d403b8575ff3e5fc28be7ab3982ca170bc566a2e92a3c50b4e077389c23faca  hard_maze/external_validation/data/easy-to-hard-data/maze_data_train_9.tar.gz' | sha256sum -c -
tar -xzf hard_maze/external_validation/data/easy-to-hard-data/maze_data_train_9.tar.gz -C hard_maze/external_validation/data/easy-to-hard-data
```

The test archive is not needed for the first scratch run. Its provenance is recorded in `docs/ARTIFACTS.md`.

## 4. Mandatory pretrained compatibility smoke test

```bash
MPLCONFIGDIR=/tmp/mpl-maze .venv-gpu/bin/python hard_maze/external_validation/relational_aux/pretrained_smoke.py
```

Required result: 9×9, p=0, deadend=true, 30 iterations, dataset seed 42, exact accuracy ≥0.95 (normally approximately 1.0). Stop if the weight hash or accuracy check fails. Optionally run one or two percolation points using `stage0_reproduce.py`; do not tune anything from them.

## 5. Full scratch — one seed first

```bash
.venv-gpu/bin/python hard_maze/external_validation/relational_aux/stage0_5_train.py \
  --config hard_maze/external_validation/configs/stage0_5_training.json \
  --out hard_maze/results_external/stage0_5_full \
  --seed-index 0
```

This is 40k train, 10k validation, 50 epochs, 10-epoch warmup, batch 50, 30 recurrent steps, model seed 101, dataset split seed 42, shuffle seed 1101, and progressive-sampling seed 2101.

Checkpoints are saved at epochs 1, 2, 5, 10, 20, 30, 40, and 50. `best.pt` maximizes validation exact accuracy, breaking ties by lower validation loss. `last.pt` supports resume.

## 6. Gate and expansion rule

The first-seed gate is validation/ID exact accuracy ≥0.95. Do not inspect OOD results for checkpoint selection.

- If the gate passes, run `--seed-index 1` and `--seed-index 2`, then evaluate ID/percolation/deadend/size behavior.
- If it fails, stop and report the learning curves, best epoch, environment, config, and checkpoint hashes. Do not start Stage 2.
