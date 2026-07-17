from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from connectivity_experiment.pair_corruption_evaluation import (
    _select_input_pairs,
    run_pair_corruption_evaluation_from_config,
)


class PairCorruptionEvaluationTest(unittest.TestCase):
    def test_pair_selection_strategy_round_robin_image_method(self) -> None:
        rows = [
            {"pair_id": "a_random_1", "image_id": "a", "sampling_method": "random"},
            {"pair_id": "a_random_2", "image_id": "a", "sampling_method": "random"},
            {"pair_id": "a_geo_1", "image_id": "a", "sampling_method": "geodesic_farthest"},
            {"pair_id": "b_random_1", "image_id": "b", "sampling_method": "random"},
            {"pair_id": "b_geo_1", "image_id": "b", "sampling_method": "geodesic_farthest"},
            {"pair_id": "b_geo_2", "image_id": "b", "sampling_method": "geodesic_farthest"},
        ]
        selected = _select_input_pairs(rows, max_pairs=4, strategy="round_robin_image_method")
        self.assertEqual(
            [row["pair_id"] for row in selected],
            ["a_random_1", "a_geo_1", "b_random_1", "b_geo_1"],
        )

    def test_pair_selection_strategy_balances_labels(self) -> None:
        rows = [
            {"pair_id": "a_pos_1", "image_id": "a", "sampling_method": "random", "connected": 1},
            {"pair_id": "a_pos_2", "image_id": "a", "sampling_method": "geo", "connected": 1},
            {"pair_id": "b_pos_1", "image_id": "b", "sampling_method": "random", "connected": 1},
            {"pair_id": "a_neg_1", "image_id": "a", "sampling_method": "random", "connected": 0},
            {"pair_id": "b_neg_1", "image_id": "b", "sampling_method": "geo", "connected": 0},
            {"pair_id": "c_neg_1", "image_id": "c", "sampling_method": "random", "connected": 0},
        ]
        selected = _select_input_pairs(rows, max_pairs=4, strategy="balanced_label_round_robin_image_method")
        labels = [int(row["connected"]) for row in selected]
        self.assertEqual(sum(labels), 2)
        self.assertEqual(len(labels) - sum(labels), 2)

    def test_corruption_evaluation_writes_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mask_dir = root / "masks"
            output_dir = root / "pair_corruption"
            mask_dir.mkdir(parents=True, exist_ok=True)

            mask = np.zeros((40, 40), dtype=np.uint8)
            mask[18:22, 6:34] = 255
            Image.fromarray(mask, mode="L").save(mask_dir / "sample.png")

            pairs_csv_path = root / "pairs.csv"
            with pairs_csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "pair_id",
                        "image_id",
                        "mask_path",
                        "point_a_y",
                        "point_a_x",
                        "point_b_y",
                        "point_b_x",
                        "connected",
                        "euclidean_distance",
                        "geodesic_distance",
                        "geodesic_euclidean_ratio",
                        "boundary_crossing_count",
                        "distance_to_boundary_min",
                        "region_value_i",
                        "region_value_j",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "pair_id": "pair_0001",
                        "image_id": "sample",
                        "mask_path": str(mask_dir / "sample.png"),
                        "point_a_y": 19,
                        "point_a_x": 10,
                        "point_b_y": 19,
                        "point_b_x": 28,
                        "connected": 1,
                        "euclidean_distance": 18.0,
                        "geodesic_distance": 18.0,
                        "geodesic_euclidean_ratio": 1.0,
                        "boundary_crossing_count": 0,
                        "distance_to_boundary_min": 2.0,
                        "region_value_i": 1,
                        "region_value_j": 1,
                    }
                )

            config = {
                "input_pairs_csv": str(pairs_csv_path),
                "output_dir": str(output_dir),
                "seed": 0,
                "label_source": "original",
                "mask_channel": 0,
                "foreground_threshold": 1,
                "corruption": {
                    "methods": ["none", "point_neighborhood_mask", "gaussian_noise"],
                    "num_trials_per_method": 2,
                    "point_neighborhood_mask": {"radius": 4, "target": ["point_a"]},
                    "gaussian_noise": {"sigma": 0.6},
                },
            }

            result = run_pair_corruption_evaluation_from_config(config=config, repo_root=root)

            self.assertTrue((output_dir / "corrupted_pair_inputs.csv").exists())
            self.assertTrue((output_dir / "corrupted_pair_inputs.jsonl").exists())
            self.assertTrue((output_dir / "corruption_explanation.txt").exists())
            self.assertTrue((output_dir / "corruption_review.csv").exists())
            self.assertTrue((output_dir / "robustness_summary.csv").exists())
            self.assertTrue((output_dir / "metadata.json").exists())

            with (output_dir / "corrupted_pair_inputs.csv").open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 5)
            required_columns = {
                "pair_id",
                "connected_original",
                "connected_after_corruption",
                "label_changed_by_corruption",
                "corruption_method",
                "corruption_trial_id",
                "corruption_params_json",
                "corrupted_image_path",
                "foreground_ratio_original",
                "foreground_ratio_corrupted",
                "changed_pixel_ratio",
                "component_count_original",
                "component_count_corrupted",
                "needs_human_review",
                "review_reasons",
            }
            self.assertTrue(required_columns.issubset(rows[0].keys()))

            targeted_rows = [row for row in rows if row["corruption_method"] == "point_neighborhood_mask"]
            self.assertTrue(any(int(row["label_changed_by_corruption"]) == 1 for row in targeted_rows))
            self.assertTrue(any(int(row["needs_human_review"]) == 1 for row in targeted_rows))

            with (output_dir / "robustness_summary.csv").open("r", newline="", encoding="utf-8") as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertEqual({row["corruption_method"] for row in summary_rows}, {"none", "point_neighborhood_mask", "gaussian_noise"})

            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["num_corrupted_samples"], len(rows))
            self.assertEqual(len(result["records"]), len(rows))


if __name__ == "__main__":
    unittest.main()
