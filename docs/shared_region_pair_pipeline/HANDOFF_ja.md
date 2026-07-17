# 共同研究者への共有手順

## 目的

この文書は、このパイプラインを共同研究者へ渡すときの実務的な入口です。

共有対象は大きく 2 つに分かれます。

- `GitHub` で共有するもの: コード、config、説明書
- `Dropbox` で共有するもの: 実データ、サンプル出力、重い画像群

## 推奨方針

基本方針は次です。

- 普段の共同作業は `GitHub` を使う
- 実データや生成済み PNG 群の受け渡しは `Dropbox` を使う

この分け方だと、最新版のコード管理と、重いファイルの配布を分離できます。

## GitHub で共有する最小構成

少なくとも次があれば、共同研究者はコードを読み、config を直し、再実行できます。

- `README.md`
- `requirements.txt`
- `docs/shared_region_pair_pipeline/`
- `configs/pair_generation_default.yaml`
- `configs/pair_generation_stare_demo.yaml`
- `configs/pair_corruption_evaluation_default.yaml`
- `configs/pair_corruption_evaluation_stare_demo.yaml`
- `src/generate_region_pairs.py`
- `src/run_pair_corruption_evaluation.py`
- `src/connectivity_experiment/`

最初に見てもらう順番は次がおすすめです。

1. [`QUICKSTART_ja.md`](QUICKSTART_ja.md)
2. [`DEMO_RUN_SUMMARY_ja.md`](DEMO_RUN_SUMMARY_ja.md)
3. [`README_ja.md`](README_ja.md)

## Dropbox で共有するもの

Dropbox には、GitHub に載せにくいものを置きます。

- `data/masks/` のような実データ
- `outputs/demo_pair_generation_stare_realdata_pairwise_check2/`
- `outputs/demo_pair_corruption_stare_realdata/`
- その他の可視化画像や解析結果

## Dropbox 用 bundle の自動生成

このリポジトリには、共同研究者向け bundle を自動で作るスクリプトがあります。

実行コマンド:

```bash
python scripts/create_collaborator_bundle.py
```

既定では次のようなフォルダができます。

```text
outputs/share_bundles/collaborator_handoff_<date>/
```

この bundle には次が入ります。

- 説明書
- 必要な code
- 主要 config
- 最低限の test
- 利用可能なら demo 出力例

## 共同研究者に送るときの一言テンプレート

以下のように伝えると分かりやすいです。

```text
まず docs/shared_region_pair_pipeline/QUICKSTART_ja.md を見てください。
コード本体は src/generate_region_pairs.py と src/run_pair_corruption_evaluation.py です。
実データやサンプル出力は Dropbox 側に置いてあります。
```

## 注意点

- `data/masks/` は Git 管理しない前提です。
- `outputs/` も通常は Git 管理しません。
- そのため、GitHub だけでは実データや sample output は渡りません。
- Dropbox bundle を作るときは、相手に見せたい sample output が手元に揃っている状態で実行してください。
