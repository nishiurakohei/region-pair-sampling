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

from connectivity_experiment.region_pair_generation import generate_region_pairs_from_config, load_binary_mask


class RegionPairGenerationTest(unittest.TestCase):
    def test_generate_region_pairs_writes_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mask_dir = root / "masks"
            output_dir = root / "generated_pairs"
            mask_dir.mkdir(parents=True, exist_ok=True)

            mask_connected = np.zeros((48, 48), dtype=np.uint8)
            mask_connected[12:36, 10:18] = 255
            mask_connected[30:38, 10:34] = 255
            Image.fromarray(mask_connected, mode="L").save(mask_dir / "connected.png")

            mask_disconnected = np.zeros((48, 48), dtype=np.uint8)
            mask_disconnected[8:20, 8:20] = 255
            mask_disconnected[28:40, 28:40] = 255
            Image.fromarray(mask_disconnected, mode="L").save(mask_dir / "disconnected.png")

            config = {
                "input_dir": str(mask_dir),
                "output_dir": str(output_dir),
                "num_pairs_per_image": 6,
                "seed": 0,
                "positive_ratio": 0.5,
                "min_component_size": 4,
                "candidate_pool_size": 256,
                "points_per_method": 8,
                "mask_channel": 0,
                "foreground_threshold": 1,
                "valid_threshold": 1.0,
                "methods": ["random", "euclidean_farthest", "geodesic_farthest"],
            }

            result = generate_region_pairs_from_config(config=config, repo_root=root)

            self.assertTrue((output_dir / "pairs.csv").exists())
            self.assertTrue((output_dir / "pairs.jsonl").exists())
            self.assertTrue((output_dir / "pairs_explanation.txt").exists())
            self.assertTrue((output_dir / "pairs_for_human_review.csv").exists())
            self.assertTrue((output_dir / "metadata.json").exists())
            self.assertTrue((output_dir / "per_image_text" / "connected.txt").exists())
            self.assertTrue((output_dir / "per_image_text" / "disconnected.txt").exists())
            self.assertGreater(len(result["records"]), 0)

            with (output_dir / "pairs.csv").open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreater(len(rows), 0)
            connected_rows = [row for row in rows if Path(row["mask_path"]).stem == "connected"]
            disconnected_rows = [row for row in rows if Path(row["mask_path"]).stem == "disconnected"]
            self.assertTrue(connected_rows)
            self.assertTrue(disconnected_rows)
            first_connected = connected_rows[0]
            first_disconnected = disconnected_rows[0]
            self.assertTrue(
                (
                    output_dir
                    / "visualizations"
                    / f"connected_{first_connected['pair_id']}_vis.png"
                ).exists()
            )
            self.assertTrue(
                (
                    output_dir
                    / "visualizations"
                    / f"disconnected_{first_disconnected['pair_id']}_vis.png"
                ).exists()
            )
            required_columns = {
                "pair_id",
                "image_id",
                "mask_path",
                "sampling_method",
                "point_a_y",
                "point_a_x",
                "point_b_y",
                "point_b_x",
                "connected",
                "euclidean_distance",
                "boundary_crossing_count",
                "distance_to_boundary_min",
                "difficulty_bucket",
                "reason_codes",
                "rule_score",
                "why_selected",
                "needs_human_review",
                "review_reasons",
            }
            self.assertTrue(required_columns.issubset(rows[0].keys()))
            self.assertIn(1, {int(row["connected"]) for row in rows})
            self.assertIn(0, {int(row["connected"]) for row in rows})

            explanation_text = (output_dir / "pairs_explanation.txt").read_text(encoding="utf-8")
            self.assertIn("reason_codes:", explanation_text)
            self.assertIn("why_selected:", explanation_text)

            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["num_pairs"], len(rows))
            self.assertIn("generation_summary", metadata)
            self.assertTrue(metadata["per_image_text_dir"].endswith("per_image_text"))
            self.assertTrue(metadata["visualization_dir"].endswith("visualizations"))

    def test_auto_mask_loading_handles_binary_and_rgb_segmentations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            grayscale_path = root / "binary01.png"
            rgb_path = root / "rgb_green.png"

            grayscale = np.zeros((16, 16), dtype=np.uint8)
            grayscale[4:12, 6:10] = 1
            Image.fromarray(grayscale, mode="L").save(grayscale_path)

            rgb = np.zeros((16, 16, 3), dtype=np.uint8)
            rgb[3:13, 5:11, 1] = 255
            Image.fromarray(rgb, mode="RGB").save(rgb_path)

            grayscale_mask = load_binary_mask(grayscale_path, mask_channel="auto", binarization_mode="auto")
            rgb_mask = load_binary_mask(rgb_path, mask_channel="auto", binarization_mode="auto")

            self.assertEqual(int(grayscale_mask.sum()), 32)
            self.assertEqual(int(rgb_mask.sum()), 60)

    def test_background_foreground_cross_region_pairs_are_not_selected_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mask_dir = root / "masks"
            output_dir = root / "generated_pairs"
            mask_dir.mkdir(parents=True, exist_ok=True)

            rgb_mask = np.zeros((48, 48, 3), dtype=np.uint8)
            rgb_mask[10:18, 10:18, 1] = 255
            rgb_mask[28:36, 28:36, 1] = 255
            Image.fromarray(rgb_mask, mode="RGB").save(mask_dir / "roads.png")

            config = {
                "input_dir": str(mask_dir),
                "output_dir": str(output_dir),
                "num_pairs_per_image": 4,
                "strict_num_pairs_per_image": True,
                "seed": 3,
                "positive_ratio": 0.5,
                "min_component_size": 4,
                "candidate_pool_size": 256,
                "points_per_method": 6,
                "mask_channel": "auto",
                "binarization_mode": "auto",
                "valid_threshold": 1.0,
                "sample_foreground": True,
                "sample_background": True,
                "methods": ["random"],
            }

            result = generate_region_pairs_from_config(config=config, repo_root=root)
            self.assertEqual(len(result["records"]), 4)
            rows = [record.to_csv_record() for record in result["records"]]
            self.assertTrue(all(row["pair_region_type"] != "cross_region" for row in rows))
            self.assertTrue(all(int(row["same_region_value"]) == 1 for row in rows))


if __name__ == "__main__":
    unittest.main()
