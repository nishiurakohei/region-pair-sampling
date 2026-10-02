# Stage 1 — Random-Pair Distance Learnability

## 判定

**Stage 1 pass, Case B**。30 iterationsのauxiliary modelはraw MAE 1.221、R² 0.825、Pearson 0.909で、linear-Manhattan baseline（MAE 3.083, R² 0.054）を明確に上回った。一方、16+ binではbaselineより良いがabsolute errorとbin内correlationが悪化し、long-distance weaknessが残る。

## 設計

Easy-to-Hard 9×9 mazeを40k/10k/10kに分離し、各mazeからstart/goalを除外した16 fixed pairsをsamplingした。Manhattan distanceは3以上、targetはexact BFSの graph distance / 9。pair marker channelは追加せず、元のstart/goal markerを消したgraph-only rasterからDT-Net coreでH(G)を一度計算し、h_u, h_v, |h_u-h_v|, h_u⊙h_v, global featureをscalar headへ渡した。これにより1 mazeの16 pairでcore計算を共有できる。

学習は5,000 mazes（80,000 pair/epoch）×10 epochs、seed 401。これはlearnability判定用のresource-bounded conditionであり、full 40k学習ではない。

## Q7–Q11

- Q7: learnable。30 iterationsでR²=0.825。
- Q8: Manhattan shortcutを上回る。linear baseline比でMAEを1.862削減。
- Q9: 16+ binはmodel MAE 3.098、linear-Manhattan MAE 9.033でbaselineより良いが、R²=-5.978、Pearson=0.092でabsoluteには弱い。
- Q10: 10→30 iterationsでMAE 2.206→1.221へ改善。100 iterationsでは1.265へわずかに悪化し、単調改善ではない。
- Q11: Case BとしてStage 2を検討できるauxiliary signalは存在する。ただしproceedにはStage 0.5 passも別途必要で、distance-dependent transfer analysisを事前計画する。

## 解釈の範囲

結果はmaze geometryを使う非自明なdistance signalを支持する。multi-taskでmain accuracyを改善すること、またはpair marker方式と同値であることはまだ示していない。
