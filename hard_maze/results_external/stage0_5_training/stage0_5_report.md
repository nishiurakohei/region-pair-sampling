# Stage 0.5 — DT-Net Scratch Training Reproduction

## 判定

**Gate 0.5 は不通過**。今回完了したのは公式条件そのものではなく、5,000 training mazes・10 epochs の資源制約版である。scratch 3 seeds の ID exact accuracy 平均は **0.003** で、Criterion A（0.95以上）を満たさない。したがって percolation、deadend、size/iteration の形を学習済み solver の failure structure と解釈できず、Stage 2 multi-task training には進まない。

## Training pipeline の特定

`mines-opt-ml/maze-extrapolation@6d09263fdb8d61274e4b0e046e07ed929b35af12` の `src/train.py` は初回 commit から空で、同 repository 単独では training を再実行できなかった。著者の公式 `aks2203/deep-thinking@152fb1182fb5dc11275859407ceec46102c20e42` 実装を採用し、DT-Net width 128、30 recurrent steps、progressive loss α=0.01、Adam、base LR 1e-3、recurrent LR 1e-3/30、weight decay 2e-4、batch 50、50 epochs、10-epoch exponential warmup を復元した。データは Easy-to-Hard の 50,000個の9×9 maze（40,000 train / 10,000 validation）である。

公式完全条件は計算見積りが数日規模だったため未実行。今回の差分は 5,000 / 1,000 examples、10 epochs、2-epoch warmup であり、ほかのモデル・loss・optimizer・30-step recurrence は維持した。よって本結果は「公式 pipeline が再現不能」という証拠ではなく、「この資源制約版では収束しなかった」という結果である。

## Training results

| seed | best epoch | best validation exact accuracy | validation loss | checkpoint |
|---:|---:|---:|---:|---|
| 101 | 6 | 0.001 | 0.6427 | `checkpoints/seed_101/best_torch.pt` |
| 102 | 9 | 0.005 | 0.4334 | `checkpoints/seed_102/best_torch.pt` |
| 103 | 1 | 0.000 | 0.6789 | `checkpoints/seed_103/best_torch.pt` |

Checkpoint selection は指定どおり validation exact accuracy。値が同率の場合は最初の epoch を保持したため、loss 最小 epoch と一致しない場合がある。

## Stage 0 test suite

全 scratch model と pretrained model を同一の固定 dataset seed 42・同一 SHA-256 の raster で直接比較した。pretrained の5-seed aggregateは Stage 0 report に保存済みで、下表は paired comparison のため seed 42 の値を示す。

| percolation p | scratch mean±std | pretrained (same dataset) |
|---:|---:|---:|
| 0.000 | 0.003±0.005 | 1.000 |
| 0.005 | 0.007±0.009 | 0.920 |
| 0.010 | 0.007±0.009 | 0.840 |
| 0.020 | 0.007±0.005 | 0.690 |
| 0.030 | 0.010±0.008 | 0.530 |
| 0.040 | 0.007±0.009 | 0.490 |

deadend=true は scratch 0.003、deadend=false は 0.000（同一 dataset の pretrained はそれぞれ 1.000 / 0.730）。

| maze size | 30 iter | 100 iter | 300 iter |
|---:|---:|---:|---:|
| 9 | 0.017 | 0.050 | 0.117 |
| 19 | 0.000 | 0.000 | 0.000 |
| 29 | 0.000 | 0.000 | 0.000 |
| 49 | 0.000 | 0.000 | 0.000 |

## Gate criteria

- Criterion A — ID ≥0.95: **False**
- Criterion B — p=0→0.04 drop ≥0.10: **False**
- Criterion C — deadend true→false drop ≥0.10: **False**
- Criterion D — large-maze 30→300 iter recovery ≥0.10: **False**

B–D のいずれかが数値上成立しても、A が失敗した model では published failure structure の再現とは判定しない。

## 原因の切り分けと残る検証

- optimizer、loss、recurrent iteration、rasterizationは公式実装に合わせ、MLX/PyTorch forward は pretrained weight で argmax 一致・最大誤差 7.63e-06 を確認済み。
- scratch initialization hash は3 seedsで異なり、pretrained weight はロードしていない。
- 最も大きい既知の差は training data 8分の1、epochs 5分の1、warmup 5分の1である。収束失敗はこの計算量削減と整合するが、因果は未確定。
- 公式40,000 train / 10,000 validation・50 epochsを3 seedsで完走するか、著者の training checkpoint/configを入手するまで Gate 0.5 は未解決のまま保持する。

## Stage 1との関係

Stage 1 auxiliary-only distance prediction は独立に完了し、全体では強い learnability を示したが長距離 bin が弱いため **Case B**。これは auxiliary target が学習可能という証拠であり、Stage 0.5 の失敗を補わない。したがって Stage 2 は未着手。
