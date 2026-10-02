# Phase 2 — β₁ auxiliary supervision

Phase 2-A:

```bash
../../.venv/bin/python beta1_distribution.py
../../.venv/bin/python train_aux_only.py cnn 1
../../.venv/bin/python evaluate_aux_only.py
```

Phase 2-B は `train_multitask.py MODEL CONDITION LAMBDA SEED` で1構成を実行する。
`CONDITION=main` は `LAMBDA=0`、`true` / `shuffled` は正の λ を指定する。設定・seed・shuffle seed は `config.json` に固定され、各epochで checkpoint が保存される。

全予測完了後：

```bash
MPLCONFIGDIR=/tmp/mpl-cache ../../.venv/bin/python evaluate_multitask.py
```

成果物は `../results_phase2/aux_only/` と `../results_phase2/multitask/` に保存される。
