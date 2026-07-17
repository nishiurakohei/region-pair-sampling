# 実データ demo 実行の共有用サマリー

## 概要

この文書は、`2026-07-16` 時点の実データ demo 実行を、共同研究者に共有しやすい形でまとめたものです。

使った config:

- `configs/pair_generation_stare_demo.yaml`
- `configs/pair_corruption_evaluation_stare_demo.yaml`

実行コマンド:

```bash
python src/generate_region_pairs.py --config configs/pair_generation_stare_demo.yaml
python src/run_pair_corruption_evaluation.py --config configs/pair_corruption_evaluation_stare_demo.yaml
```

## ペア生成 demo

入力:

- `data/masks`

主な出力:

- `outputs/demo_pair_generation_stare_realdata_pairwise_check2/pairs.csv`
- `outputs/demo_pair_generation_stare_realdata_pairwise_check2/pairs.jsonl`
- `outputs/demo_pair_generation_stare_realdata_pairwise_check2/pairs_explanation.txt`
- `outputs/demo_pair_generation_stare_realdata_pairwise_check2/per_image_text/`
- `outputs/demo_pair_generation_stare_realdata_pairwise_check2/visualizations/`
- `outputs/demo_pair_generation_stare_realdata_pairwise_check2/metadata.json`

要点:

- 処理画像数は 8
- 生成ペア数は 192
- 画像ごとに txt を 1 つ出力
- ペアごとに `_vis.png` を 1 つ出力
- foreground-background の自明な cross-region pair は既定で除外

## corruption demo

主な出力:

- `outputs/demo_pair_corruption_stare_realdata/corrupted_pair_inputs.csv`
- `outputs/demo_pair_corruption_stare_realdata/corruption_review.csv`
- `outputs/demo_pair_corruption_stare_realdata/robustness_summary.csv`
- `outputs/demo_pair_corruption_stare_realdata/corrupted_images/`
- `outputs/demo_pair_corruption_stare_realdata/metadata.json`

要点:

- 固定済みペアに対して複数の corruption を適用
- `none`, `random_rectangle_mask`, `line_occlusion`, `point_neighborhood_mask`, `gaussian_noise`, `blur` を確認
- 構造変化や label 変化を CSV に記録

## この demo の限界

- 処理した最初の 8 画像は同じ STARE 系列に偏っていました。
- そのため、この run は一般評価ではなく end-to-end の動作確認として読むのが適切です。

## 共同研究者に見てもらうとよい場所

まずは次を見ると状況が伝わりやすいです。

1. `outputs/demo_pair_generation_stare_realdata_pairwise_check2/pairs.csv`
2. `outputs/demo_pair_generation_stare_realdata_pairwise_check2/per_image_text/`
3. `outputs/demo_pair_generation_stare_realdata_pairwise_check2/visualizations/`
4. `outputs/demo_pair_corruption_stare_realdata/robustness_summary.csv`
