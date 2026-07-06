from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from connectivity_experiment.region_pair_cnn import (
    PairRecordDataset,
    SimplePairCNN,
    assign_image_splits,
    input_channels_for_mode,
    resolve_pos_weight_value,
)


class RegionPairCNNTest(unittest.TestCase):
    def test_assign_image_splits_covers_all_images(self) -> None:
        records = [{"image_id": f"img_{index}", "connected": "0"} for index in range(10)]
        split_by_image = assign_image_splits(records, seed=42, train_fraction=0.6, val_fraction=0.2)
        self.assertEqual(len(split_by_image), 10)
        self.assertEqual(set(split_by_image.values()), {"train", "val", "test"})

    def test_pair_record_dataset_renders_three_channels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "mask.png"
            mask = np.zeros((16, 16), dtype=np.uint8)
            mask[4:12, 5:10] = 255
            Image.fromarray(mask, mode="L").save(image_path)
            records = [
                {
                    "image_path": str(image_path),
                    "row_i": "6",
                    "col_i": "6",
                    "row_j": "2",
                    "col_j": "2",
                    "connected": "0",
                }
            ]
            dataset = PairRecordDataset(records, input_size=16, marker_radius=1)
            image_tensor, label, _ = dataset[0]
            self.assertEqual(tuple(image_tensor.shape), (3, 16, 16))
            self.assertEqual(float(label), 0.0)
            self.assertGreater(float(image_tensor[0].sum()), 0.0)
            self.assertGreater(float(image_tensor[1].sum()), 0.0)
            self.assertGreater(float(image_tensor[2].sum()), 0.0)

    def test_pair_record_dataset_supports_input_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "mask.png"
            mask = np.zeros((16, 16), dtype=np.uint8)
            mask[4:12, 5:10] = 255
            Image.fromarray(mask, mode="L").save(image_path)
            records = [
                {
                    "image_path": str(image_path),
                    "row_i": "6",
                    "col_i": "6",
                    "row_j": "2",
                    "col_j": "2",
                    "connected": "1",
                }
            ]
            for input_mode, expected_channels in [
                ("mask_point_i_point_j", 3),
                ("mask_point_i_point_j_line", 4),
                ("point_i_point_j_only", 2),
                ("mask_only", 1),
                ("mask_merged_points", 2),
            ]:
                dataset = PairRecordDataset(records, input_size=16, marker_radius=1, input_mode=input_mode)
                image_tensor, _, _ = dataset[0]
                self.assertEqual(tuple(image_tensor.shape), (expected_channels, 16, 16))
                self.assertEqual(input_channels_for_mode(input_mode), expected_channels)
                if input_mode == "mask_point_i_point_j_line":
                    self.assertGreater(float(image_tensor[3].sum()), 0.0)

    def test_simple_pair_cnn_supports_pool_sizes(self) -> None:
        for pool_size in [4, 8, 16]:
            model = SimplePairCNN(input_channels=4, pool_size=pool_size)
            output = model(torch.zeros((2, 4, 64, 64), dtype=torch.float32))
            self.assertEqual(tuple(output.shape), (2,))

    def test_resolve_pos_weight_value(self) -> None:
        self.assertEqual(resolve_pos_weight_value(pos_count=10, neg_count=30, pos_weight_mode="balanced"), 3.0)
        self.assertEqual(resolve_pos_weight_value(pos_count=10, neg_count=30, pos_weight_mode="one"), 1.0)


if __name__ == "__main__":
    unittest.main()
