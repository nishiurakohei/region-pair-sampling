from __future__ import annotations

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

from connectivity_experiment.io_utils import read_csv, write_csv
from connectivity_experiment.region_pair_example_visualization import build_example_groups, save_example_grids


class RegionPairExampleVisualizationTest(unittest.TestCase):
    def test_build_example_groups_and_save(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "mask.png"
            mask = np.zeros((16, 16), dtype=np.uint8)
            mask[4:12, 4:12] = 255
            Image.fromarray(mask, mode="L").save(image_path)

            records = [
                {
                    "pair_id": f"pair_{index}",
                    "image_path": str(image_path),
                    "row_i": "5",
                    "col_i": "5",
                    "row_j": "10",
                    "col_j": "10",
                    "connected": "1" if index % 2 == 0 else "0",
                    "sampling_method": "random" if index < 3 else "hard_pair_targeted",
                    "loss": str(0.1 * index),
                    "pred_prob_connected": "0.75",
                    "pred_label": "1",
                    "is_correct": "1" if index % 3 else "0",
                    "euclidean_distance": "7.0",
                    "geodesic_distance": "8.0" if index % 2 == 0 else "",
                    "geodesic_euclidean_ratio": "1.1" if index % 2 == 0 else "",
                    "boundary_crossing_count": str(index % 4),
                }
                for index in range(6)
            ]
            grouped = build_example_groups(records, per_group=2, random_seed=0)
            self.assertEqual(set(grouped.keys()), {"hard_pair_targeted", "random"})
            self.assertEqual(len(grouped["random"]["random_examples"]), 2)
            self.assertEqual(len(grouped["hard_pair_targeted"]["high_loss_examples"]), 2)

            predictions_path = temp_path / "predictions.csv"
            write_csv(predictions_path, records)
            out_dir = temp_path / "example_pairs"
            output_paths = save_example_grids(predictions_path, out_dir=out_dir, per_group=2, n_cols=2, random_seed=0)
            self.assertTrue(any(path.name == "example_pairs_report.md" for path in output_paths))
            self.assertTrue((out_dir / "random_random_examples.png").exists())
            self.assertTrue((out_dir / "hard_pair_targeted_high_loss_examples.png").exists())
            self.assertTrue((out_dir / "random_random_examples.csv").exists())
            self.assertTrue((out_dir / "hard_pair_targeted_high_loss_examples.csv").exists())

            csv_rows = read_csv(out_dir / "random_random_examples.csv")
            self.assertEqual(len(csv_rows), 2)
            self.assertEqual(csv_rows[0]["example_type"], "random_examples")
            self.assertIn("component_gap_distance", csv_rows[0])
            self.assertIn("max_outside_segment_length", csv_rows[0])
            self.assertIn("difficulty_bucket", csv_rows[0])

            report_text = (out_dir / "example_pairs_report.md").read_text(encoding="utf-8")
            self.assertIn("PNG", report_text)
            self.assertIn("CSV", report_text)
            self.assertIn("random_random_examples.csv", report_text)


if __name__ == "__main__":
    unittest.main()
