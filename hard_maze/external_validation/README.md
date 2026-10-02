# Knutson maze external validation

公式 `mines-opt-ml/maze-extrapolation` を `upstream/maze-extrapolation` に固定し、変更は `relational_aux/` に分離する。upstream clone、データ、checkpointは通常Gitに含めず、復元手順とhashを`../../docs/NEXT_TASK.md`と`../../docs/ARTIFACTS.md`に記録する。

```bash
MPLCONFIGDIR=/tmp/mpl-external .venv/bin/python relational_aux/stage0_reproduce.py
```

公式 commit は `results_external/stage0_reproduction/upstream_commit.txt`、環境は `environment.json` に記録する。2024-10-03 の upstream commit と同時期の raster 表現を再現するため `maze-dataset==1.1.0` を使用する。最新版 1.4.2 では同じ重みの 9×9 ID accuracy が 0 になり、互換性がない。

Stage 0は5 seedsで完了し、主要なpublished failure modeと9–99 size extrapolationを再現した。結果は`../results_external/stage0_reproduction/reproduction_report.md`。検証は`full_gate0_complete=true`であり、Stage 1へ進める。

Stage 0.5の資源制約版は未収束でGate不通過。Stage 1はCase Bで通過したが、Stage 2は停止中。次はGPU上でpretrained smoke testを通した後、`configs/stage0_5_training.json`を使いfull scratchを1 seedだけ実行する。詳細は`../../docs/NEXT_TASK.md`を参照。
