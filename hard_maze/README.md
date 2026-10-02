# Hard-Maze benchmark

Phase 1 の純 NumPy maze/CNN/ResNet 実装を再利用した、完全観測の reachability benchmark。

```bash
../.venv/bin/python src/generate.py --pilot 5000
../.venv/bin/python src/generate.py
../.venv/bin/python src/verify.py
../.venv/bin/python src/diagnostics.py
sh src/driver.sh
../.venv/bin/python src/evaluate.py
MPLCONFIGDIR=/tmp/mpl-cache ../.venv/bin/python src/figures.py
../.venv/bin/python src/report.py
```

設定は `config.json`、データは `data/H0`–`data/H3`、最終成果物は `results_hard_maze/` に保存される。

## Phase 2: β₁ auxiliary supervision

H3 を変更せず、β₁ 単独回帰と shared-encoder multi-task 学習を実施済み。β₁ は画像から高精度に予測できた（CNN R²=0.962、ResNet R²=0.914）が、5 seed・true/shuffled control・validation 選択 λ の比較では reachability accuracy / matched-pair accuracy の頑健な改善は確認されなかった。現時点では auxiliary unlearnability ではなく、task independence / representation mismatch と大きな λ での task competition に近い結果である。

- 最終レポート: `results_phase2/multitask/phase2_report.md`
- Phase 2-A レポート: `results_phase2/aux_only/aux_only_report.md`
- 数値: `results_phase2/multitask/metrics.json`
- 検証: `results_phase2/multitask/verification.json` (`all_ok=true`)
- 実行方法と設定: `phase2/README.md`, `phase2/config.json`

## Phase 3: random-pair connectivity auxiliary supervision

H3 上のランダムな頂点対の連結性を補助課題にした shared-encoder multi-task 学習を完了した。CNN pilot で K=16 を選び、validation main BCE により CNN λ=1.0、ResNet λ=0.3 を選択した。5 paired seeds では true pair が main-only を accuracy で CNN +0.0303、ResNet +0.1421 改善し、shuffled-label control に対する差も両モデルで正の 95% bootstrap CI を得た。β₁ 補助課題では得られなかった頑健な transfer であり、H3 では scalar topology より relational supervision の方が整合的だった。

ただし graph distance 21 以上の auxiliary accuracy は CNN 0.549、ResNet 0.489 まで低下した。結果は relational transfer の証拠であって、長距離の algorithmic path reasoning や因果機構の証明ではない。

- 最終レポート: `results_phase3/multitask/phase3_report.md`
- 数値: `results_phase3/multitask/metrics.json`
- frozen-encoder probes: `results_phase3/multitask/representation_probes.json`
- 検証: `results_phase3/multitask/verification.json` (`all_ok=true`)
- 実行方法と設定: `phase3/README.md`, `phase3/config.json`

## External validation: Knutson et al. DT-Net

Stage 0 では official pretrained DT-Net の published qualitative behavior を再現し、full Gate 0 を通過した。続く Stage 0.5 では公式 training pipeline を特定したが、実行済みなのは 5,000 train / 1,000 validation・10 epochs の資源制約版である。scratch 3 seeds の ID exact accuracy は平均 0.0033 に留まり、Gate 0.5 は不通過。公式 40,000 / 10,000・50 epochs の結果ではないため、公式 pipeline 自体の失敗とは結論しない。

Stage 1 の random-pair shortest-path distance auxiliary-only 学習は、30 iterations で raw MAE 1.221、R² 0.825 を達成し、Manhattan 系 baselines を上回った。一方、16+ distance bin には弱さが残るため Case B と判定した。Stage 0.5 と Stage 1 の両方を通過する条件を満たさず、Stage 2 multi-task training は未着手。

- Stage 0.5 report: `results_external/stage0_5_training/stage0_5_report.md`
- Stage 0.5 metrics / verification: `results_external/stage0_5_training/metrics.json`, `results_external/stage0_5_training/verification.json`
- Stage 1 report: `results_external/stage1_aux_learnability/aux_learnability_report.md`
- Stage 1 metrics / verification: `results_external/stage1_aux_learnability/metrics.json`, `results_external/stage1_aux_learnability/verification.json`
