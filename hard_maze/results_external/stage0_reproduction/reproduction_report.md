# Stage 0 — Published Baseline Reproduction

## 判定

DT-Net について、Knutson et al. の主要な定性的結果を5 seedsで再現した。9×9 training-distribution条件は exact solution accuracy 1.000、deadend-start shift と percolation shift では明確に低下し、maze size増加はtest-time iterationsの増加で回復した。

これは **full Gate 0 pass** である。論文の全size grid 9,19,…,99と30,100,300,1000,3000 iterationsをカバーし、99×99でも1000/3000 iterationsで5/5正解だった。Stage 1へ進む前提を満たした。

## Upstreamと環境

- Official repository: `mines-opt-ml/maze-extrapolation`
- Commit: `6d09263fdb8d61274e4b0e046e07ed929b35af12`
- Model: repository同梱のDT-Net pretrained weight（SHA-256 `b8beed72aad07b300e178f74f0c3628886cfb28fca5d271c451496af38478216`）
- Extension codeは`external_validation/relational_aux/`に分離し、upstreamはcleanのまま保持した。
- `maze-dataset==1.1.0`、`muutils==0.6.21`、Python 3.13、PyTorch 2.13を使用した。

upstreamのrequirementsはversion未固定である。現在の`maze-dataset==1.4.2`では同じweightの9×9 ID accuracyが0/10となったが、upstream初回commit時点の最新版1.1.0では20/20、さらに本実験500/500を達成した。したがってraster/API互換性は再現に必須である。

## 結果

| axis | condition | exact accuracy (mean±std, 5 seeds) |
|---|---|---:|
| ID | 9×9, p=0, deadend=true, 30 iter | 1.000±0.000 |
| deadend | 9×9, deadend=false, 30 iter | 0.770±0.040 |
| percolation | p=0.005 | 0.914±0.027 |
| percolation | p=0.010 | 0.854±0.010 |
| percolation | p=0.020 | 0.734±0.042 |
| percolation | p=0.030 | 0.606±0.056 |
| percolation | p=0.040 | 0.542±0.035 |

| size | 30 iter | 100 iter | 300 iter |
|---:|---:|---:|---:|
| 9 | 1.000 | 1.000 | 1.000 |
| 19 | 0.270 | 1.000 | 1.000 |
| 29 | 0.040 | 0.780 | 1.000 |
| 49 | 0.000 | 0.060 | 0.800 |

追加したfull-tail（各sizeで1 maze × 5 dataset seeds）：

| size | 30 iter | 100 iter | 300 iter | 1000 iter | 3000 iter |
|---:|---:|---:|---:|---:|---:|
| 39 | 0.00 | 0.40 | 1.00 | 1.00 | 1.00 |
| 59 | 0.00 | 0.00 | 0.40 | 1.00 | 1.00 |
| 69 | 0.00 | 0.20 | 0.80 | 1.00 | 1.00 |
| 79 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| 89 | 0.00 | 0.00 | 0.40 | 1.00 | 1.00 |
| 99 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |

deadend-start=falseの0.770は論文記載のDT-Net 0.72と同程度である。percolationの増加に伴う急減、ならびにmaze size増加時にiterations追加で回復する方向も再現した。完全な数値一致を主張するものではない。

## Scopeと次のGate

現時点で支持されるのは「published DT-Netの既知failure modeとsize extrapolationを、同時期依存と公式weightで再現できた」という範囲である。relational auxiliaryの外部転移についてはまだ結果がない。

full-tailは計算量のため論文の100 mazes/pointではなく5 mazes/pointであり、large-size accuracyの精密推定ではなく定性的再現である。一方、percolation・deadend・IDは各point 500 mazes（100×5 seeds）で評価した。Gate 0の成功基準は完全な数値再現ではなく定性的再現なので、次はrandom-pair shortest-path distance generatorとauxiliary-only learnabilityへ進める。PI-Netはcheckoutに設定・weightがないため、別途公式artifactが得られない限りDT-Netをprimary targetとする。
