# 二点ペア生成・corruption パイプライン共有ガイド

## 目的

このガイドは、現在のパイプラインを共同研究者に共有しやすい形で整理するためのものです。

まず共有の渡し方を確認したい場合は [`HANDOFF_ja.md`](HANDOFF_ja.md) を見てください。
最短で動かしたい場合は [`QUICKSTART_ja.md`](QUICKSTART_ja.md) を見てください。
具体例を見たい場合は [`DEMO_RUN_SUMMARY_ja.md`](DEMO_RUN_SUMMARY_ja.md) を見てください。

このパイプラインで扱っている一連の流れは次の通りです。
1. 二値マスク画像を入力する
2. 二点ペアを生成する
3. 各ペアについて連結性や幾何特徴を記録する
4. 固定済みのペアに対してマスキングやノイズを加える
5. 後段の学習や解析に使える形で出力する

## 最初に見るべきファイル

共同研究者がまず触る入口は次の 4 つです。

- 二点ペア生成 CLI: [`src/generate_region_pairs.py`](../../src/generate_region_pairs.py)
- 二点ペア生成コア: [`src/connectivity_experiment/region_pair_generation.py`](../../src/connectivity_experiment/region_pair_generation.py)
- corruption 評価 CLI: [`src/run_pair_corruption_evaluation.py`](../../src/run_pair_corruption_evaluation.py)
- corruption 評価コア: [`src/connectivity_experiment/pair_corruption_evaluation.py`](../../src/connectivity_experiment/pair_corruption_evaluation.py)

## 共有向けの構造

```text
docs/shared_region_pair_pipeline/
  README.md
  README_en.md
  README_ja.md
  HANDOFF_en.md
  HANDOFF_ja.md
  QUICKSTART_en.md
  QUICKSTART_ja.md
  DEMO_RUN_SUMMARY_en.md
  DEMO_RUN_SUMMARY_ja.md

configs/
  pair_generation_default.yaml
  pair_generation_stare_demo.yaml
  pair_corruption_evaluation_default.yaml
  pair_corruption_evaluation_stare_demo.yaml

scripts/
  create_collaborator_bundle.py

src/
  generate_region_pairs.py
  run_pair_corruption_evaluation.py
  connectivity_experiment/
    config.py
    io_utils.py
    region_pair_sampling.py
    region_pair_generation.py
    pair_corruption_evaluation.py

outputs/
  demo_pair_generation_stare_realdata_pairwise_check2/
  demo_pair_corruption_stare_realdata/
  share_bundles/
    collaborator_handoff_<date>/
```

## 一連の流れ

### Step 1. 二値マスクを用意する

- 入力は `data/masks/` のような二値マスク画像ディレクトリです。
- グレースケールでも多チャンネルでも内部では `foreground/background` の bool mask に変換されます。
- `min_component_size` で小さい連結成分を除去できます。

関連コード:
- [`src/connectivity_experiment/region_pair_generation.py`](../../src/connectivity_experiment/region_pair_generation.py)
- [`src/connectivity_experiment/region_pair_sampling.py`](../../src/connectivity_experiment/region_pair_sampling.py)

### Step 2. 二点ペアを生成し、特徴量を記録する

実行コマンド:

```bash
python src/generate_region_pairs.py --config configs/pair_generation_stare_demo.yaml
```

この段階では次を行います。
- 有効点をサンプリングする
- 二点ペアを作る
- 可能な限り画像ごとの最終ペア数を固定する
- 自明な foreground-background の cross-region pair を既定で除外する
- `connected / disconnected` を付与する
- 幾何特徴を計算する
- 決定論的な explanation と review flag を付ける
- CSV / JSONL / TXT / metadata を出力する

主な出力:
- `pairs.csv`
- `pairs.jsonl`
- `pairs_explanation.txt`
- `pairs_for_human_review.csv`
- `per_image_text/<元画像名>.txt`
- `visualizations/<元画像名>_<pair_id>_vis.png`
- `metadata.json`

主な記録項目:
- `connected`
- `euclidean_distance`
- `geodesic_distance`
- `geodesic_euclidean_ratio`
- `boundary_crossing_count`
- `distance_to_boundary_min`
- `difficulty_bucket`
- `reason_codes`
- `why_selected`
- `review_reasons`

### Step 3. ペアレベルの出力を見る

後段の解析で主に使うのは `pairs.csv` です。

見る順番の目安:
1. `metadata.json` で run 全体の条件を確認する
2. `pairs.csv` で構造化データを見る
3. `per_image_text/` で画像ごとの txt を見る
4. `visualizations/` でペアごとの `_vis.png` 重ね描き画像を見る
5. `pairs_explanation.txt` で全体をまとめた説明を見る
6. `pairs_for_human_review.csv` で手確認が必要なペアを見る

### Step 4. 固定済みペアに対してマスキングやノイズを加える

実行コマンド:

```bash
python src/run_pair_corruption_evaluation.py --config configs/pair_corruption_evaluation_stare_demo.yaml
```

この段階では次を行います。
- すでに作成済みの `pairs.csv` を読む
- 二点ペア自体は固定する
- 画像またはマスクに corruption を加える
- corruption 後の連結性を補助情報として再計算する
- 構造変化が起きたかを記録する
- corruption レベルの summary を出力する

現在の corruption method:
- `none`
- `random_rectangle_mask`
- `random_circle_mask`
- `line_occlusion`
- `point_neighborhood_mask`
- `salt_and_pepper`
- `gaussian_noise`
- `blur`
- `erosion`
- `dilation`

主な出力:
- `corrupted_pair_inputs.csv`
- `corrupted_pair_inputs.jsonl`
- `corruption_explanation.txt`
- `corruption_review.csv`
- `robustness_summary.csv`
- `metadata.json`

### Step 5. 後段の実験で使う

このパイプラインは後段モデルを固定していません。

典型的な使い方:
- `pairs.csv` を使って classifier の学習データを作る
- 幾何特徴と難易度の関係を調べる
- corruption が構造に与える影響を調べる
- 後からモデル予測と結合して robustness 解析を行う

## 入力画像に対する柔軟性

現在の pair generation loader は、血管マスクだけでなく、道路セグメンテーションやコンクリートひび割れセグメンテーションも想定して使えるようにしています。

主に使う設定項目:
- `mask_channel: auto`
- `binarization_mode: auto`
- `foreground_threshold`
- `foreground_values`
- `invert_mask`
- `use_alpha_channel`
- `auto_invert_if_foreground_dominates`

実際の意味:
- `mask_channel: auto` は、RGB のどのチャンネルに前景が載っているか固定でない場合に使えます。
- `binarization_mode: auto` は、`0/1`、`0/255`、単純な色付き segmentation を吸収しやすくします。
- `foreground_values` は、特定の画素値だけを前景として扱いたい場合に使えます。
- `invert_mask` や `auto_invert_if_foreground_dominates` は、データセットごとに前景・背景の極性が異なる場合に有効です。

## コードマップ

### 二点ペア生成レイヤ

- [`src/generate_region_pairs.py`](../../src/generate_region_pairs.py)
  CLI の入口です。
- [`src/connectivity_experiment/region_pair_generation.py`](../../src/connectivity_experiment/region_pair_generation.py)
  データセット非依存の pair 生成、explanation、review 出力を担います。
- [`src/connectivity_experiment/region_pair_sampling.py`](../../src/connectivity_experiment/region_pair_sampling.py)
  幾何特徴計算やサンプリングの共通部品です。

### corruption 評価レイヤ

- [`src/run_pair_corruption_evaluation.py`](../../src/run_pair_corruption_evaluation.py)
  CLI の入口です。
- [`src/connectivity_experiment/pair_corruption_evaluation.py`](../../src/connectivity_experiment/pair_corruption_evaluation.py)
  固定ペアへの corruption、構造 summary、review 判定を担います。

### 共通ユーティリティ

- [`src/connectivity_experiment/config.py`](../../src/connectivity_experiment/config.py)
  軽量 YAML パーサです。
- [`src/connectivity_experiment/io_utils.py`](../../src/connectivity_experiment/io_utils.py)
  CSV / JSON / JSONL の入出力補助です。

### テスト

- [`tests/test_region_pair_generation.py`](../../tests/test_region_pair_generation.py)
- [`tests/test_pair_corruption_evaluation.py`](../../tests/test_pair_corruption_evaluation.py)
- [`tests/test_config_loader.py`](../../tests/test_config_loader.py)

## default config と demo config

汎用の出発点として使うもの:
- [`configs/pair_generation_default.yaml`](../../configs/pair_generation_default.yaml)
- [`configs/pair_corruption_evaluation_default.yaml`](../../configs/pair_corruption_evaluation_default.yaml)

実データで一度動かした具体例として使うもの:
- [`configs/pair_generation_stare_demo.yaml`](../../configs/pair_generation_stare_demo.yaml)
- [`configs/pair_corruption_evaluation_stare_demo.yaml`](../../configs/pair_corruption_evaluation_stare_demo.yaml)

## デモ run の参照先

実データを使った end-to-end のデモ run はここにあります。
- [`outputs/reports/stare_realdata_demo_run_2026-07-16.md`](../../outputs/reports/stare_realdata_demo_run_2026-07-16.md)

このレポートには次がまとまっています。
- 何を実行したか
- どの config を使ったか
- 何件のペアが出たか
- どの corruption method を使ったか
- どのような構造変化が観測されたか

## 最小共有セット

リポジトリ全体を渡さずに最小限だけ共有するなら、次を含めると実用的です。

- `docs/shared_region_pair_pipeline/`
- `configs/pair_generation_default.yaml`
- `configs/pair_generation_stare_demo.yaml`
- `configs/pair_corruption_evaluation_default.yaml`
- `configs/pair_corruption_evaluation_stare_demo.yaml`
- `src/generate_region_pairs.py`
- `src/run_pair_corruption_evaluation.py`
- `src/connectivity_experiment/config.py`
- `src/connectivity_experiment/io_utils.py`
- `src/connectivity_experiment/region_pair_sampling.py`
- `src/connectivity_experiment/region_pair_generation.py`
- `src/connectivity_experiment/pair_corruption_evaluation.py`
- `outputs/reports/stare_realdata_demo_run_2026-07-16.md`

## 共同研究者向けメモ

- pair generation 側はディレクトリ単位で dataset 非依存です。通常は `input_dir` を変えるだけで新しいマスク集合に載せられます。
- corruption 側は、すでに `pairs.csv` ができていることを前提にしています。
- corruption 評価の正解ラベルはデフォルトでは元画像上のラベルを使い、corruption 後の構造変化は補助情報として別に保持します。
- 現在の demo はパイプラインの実演としては十分ですが、実データ全体に対する代表的な結論ではありません。
