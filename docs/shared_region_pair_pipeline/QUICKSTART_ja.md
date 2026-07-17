# クイックスタート

## 目的

この文書は、共同研究者が最短でこのパイプラインを動かすための手順です。

## 1. 依存関係を入れる

```bash
python -m pip install -r requirements.txt
```

必要なのは主に `numpy`, `Pillow`, `scipy`, `matplotlib` です。

## 2. マスク画像を置く

最も簡単なのは `data/masks/` に二値マスク画像を置く方法です。

別の場所を使う場合は、config の `input_dir` を変更してください。

## 3. 二点ペアを生成する

実行例:

```bash
python src/generate_region_pairs.py --config configs/pair_generation_stare_demo.yaml
```

この段階で次が行われます。

- 画像ごとに一定数のペアを選ぶ
- 既定では foreground-background の自明なペアを除外する
- 幾何特徴を計算する
- 画像ごとの txt を出力する
- ペアごとの可視化 PNG を出力する

主な出力:

- `pairs.csv`
- `pairs.jsonl`
- `pairs_explanation.txt`
- `per_image_text/<元画像名>.txt`
- `visualizations/<元画像名>_<pair_id>_vis.png`
- `metadata.json`

## 4. 生成結果を見る

最初は次の順で見るのがおすすめです。

1. `metadata.json`
2. `pairs.csv`
3. `per_image_text/`
4. `visualizations/`

## 5. 必要なら corruption 評価を行う

```bash
python src/run_pair_corruption_evaluation.py --config configs/pair_corruption_evaluation_stare_demo.yaml
```

この段階では、すでに固定されたペアに対してマスキングやノイズを加えます。

## 6. 新しいデータセットに移すときにまず見る項目

まずは config の以下を調整してください。

- `input_dir`
- `output_dir`
- `num_pairs_per_image`
- `mask_channel`
- `binarization_mode`
- `sample_foreground`
- `sample_background`
- `pair_component_mode`

## 7. 共有時の最短導線

共同研究者に渡すときは、次の 3 つを一緒に案内すると十分です。

1. この `QUICKSTART_ja.md`
2. [`HANDOFF_ja.md`](HANDOFF_ja.md)
3. [`DEMO_RUN_SUMMARY_ja.md`](DEMO_RUN_SUMMARY_ja.md)
