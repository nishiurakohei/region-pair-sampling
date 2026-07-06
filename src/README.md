# Source Layout

CLI と共通ロジックを分けています。

- `generate_dataset.py`: curved dataset v1 生成
- `generate_dataset.py`: curved dataset と multi-scene query dataset の生成
- `run_region_pair_sampling.py`: 既存マスクから valid point と二点ペア特徴量を出力
- `train_classifier.py`: NumPy CNN の学習と評価
- `score_difficulty.py`: 難易度 proxy の再計算
- `visualize_samples.py`: サンプル一覧画像の出力
- `generate_curves.py`: package 内の曲線生成関数への入口
- `render_shapes.py`: package 内のレンダリング関数への入口
- `connectivity_oracles.py`: package 内の連結性判定関数への入口
- `connectivity_experiment/config.py`: 設定読込
- `connectivity_experiment/generate_curves.py`: 直線、Bezier、C字型中心線の生成
- `connectivity_experiment/render_shapes.py`: 3 チャンネル画像レンダリングと crescent 半径プロファイル
- `connectivity_experiment/connectivity_oracles.py`: 幾何・ピクセル連結判定
- `connectivity_experiment/generation.py`: family 選択、正負生成、multi-scene query 生成、メタデータ集約
- `connectivity_experiment/geometry.py`: 連結判定と経路指標
- `connectivity_experiment/model.py`: `small` / `large` CNN 実装
- `connectivity_experiment/pixel_baseline.py`: ピクセル連結ベースライン
- `connectivity_experiment/region_pair_sampling.py`: distance transform、点サンプリング、pair metric、可視化
- `connectivity_experiment/visualize.py`: コンタクトシート作成

必要に応じて、ここから `processed` 用前処理や追加モデルを増やせます。
