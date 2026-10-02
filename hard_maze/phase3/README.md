# Phase 3 — random-pair connectivity auxiliary supervision

既存 H3 を変更せず、start/goal を除外した balanced node pair の connectivity を auxiliary task にする。

```bash
../../.venv/bin/python prepare_pairs.py
../../.venv/bin/python train.py aux cnn true 1 16 1
../../.venv/bin/python evaluate_aux.py
```

Phase 3-B の1構成は次で実行する。

```bash
../../.venv/bin/python train.py multi cnn true 0.03 16 1
```

K pilot後に `select_k.py`、全 λ/condition完了後に `evaluate.py` と `probes.py` を実行する。設定・seedは `config.json`、成果物は `../results_phase3/` に保存される。train pairはepochごとに決定論的に再生成し、val/test pairは固定する。

選択済み K=16 の full sweep はリポジトリ root から次で再開できる。既存 prediction は skip され、未完了構成は epoch checkpoint から再開する。

```bash
hard_maze/phase3/run_sweep.sh
MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python hard_maze/phase3/evaluate.py
.venv/bin/python hard_maze/phase3/probes.py
MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python hard_maze/phase3/evaluate.py
```

最後の `evaluate.py` 再実行で representation probe を最終 report に取り込む。
