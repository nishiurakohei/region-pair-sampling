from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from connectivity_experiment.region_pair_sampling import (
    attach_region_pair_difficulty,
    build_pair_records,
    build_valid_mask,
    compute_distance_transform,
    sample_points,
    select_pair_records,
)
from connectivity_experiment.region_pair_diagnostics import filter_easy_only_records_v2


class RegionPairSamplingTest(unittest.TestCase):
    def test_valid_mask_shrinks_with_threshold(self) -> None:
        mask = np.zeros((20, 20), dtype=bool)
        mask[4:16, 4:16] = True
        distance_map = compute_distance_transform(mask)
        valid_low = build_valid_mask(mask, distance_map, min_distance_to_boundary=1.0)
        valid_high = build_valid_mask(mask, distance_map, min_distance_to_boundary=3.0)

        self.assertLess(int(valid_high.sum()), int(valid_low.sum()))
        self.assertTrue(np.all(valid_low <= mask))
        self.assertTrue(np.all(valid_high <= mask))

    def test_pair_metrics_capture_nonconvexity(self) -> None:
        mask = np.zeros((40, 40), dtype=bool)
        mask[5:35, 5:10] = True
        mask[30:35, 5:30] = True
        mask[5:35, 25:30] = True
        distance_map = compute_distance_transform(mask)
        background_distance_map = compute_distance_transform(~mask)

        records = build_pair_records(
            mask=mask,
            distance_map=distance_map,
            points=[(8, 7), (8, 27)],
            image_id="toy",
            image_path="toy.png",
            valid_threshold=1.0,
            sampling_method="manual",
            background_distance_map=background_distance_map,
            pair_component_mode="same_component",
        )
        attach_region_pair_difficulty(records)

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(int(record["connected"]), 1)
        self.assertGreater(float(record["geodesic_distance"]), float(record["euclidean_distance"]))
        self.assertGreater(float(record["geodesic_euclidean_ratio"]), 1.0)
        self.assertGreaterEqual(int(record["boundary_crossing_count"]), 2)
        self.assertEqual(record["point_region_i"], "foreground")
        self.assertEqual(record["feature_triplet"][2], int(record["boundary_crossing_count"]))
        self.assertIn(record["difficulty_bucket"], {"Hard", "Very hard", "Medium"})

    def test_background_and_cross_region_pairs(self) -> None:
        mask = np.zeros((24, 24), dtype=bool)
        mask[8:16, 10:14] = True
        distance_map = compute_distance_transform(mask)
        background_distance_map = compute_distance_transform(~mask)

        points = [
            {"row": 10, "col": 11, "region_value": 1, "region_name": "foreground"},
            {"row": 2, "col": 2, "region_value": 0, "region_name": "background"},
            {"row": 20, "col": 20, "region_value": 0, "region_name": "background"},
        ]
        records = build_pair_records(
            mask=mask,
            distance_map=distance_map,
            background_distance_map=background_distance_map,
            points=points,
            image_id="toy_bg",
            image_path="toy_bg.png",
            valid_threshold=1.0,
            sampling_method="manual",
            pair_component_mode="all_pairs",
        )

        self.assertEqual(len(records), 3)
        cross_region_records = [record for record in records if record["pair_region_type"] == "cross_region"]
        background_records = [record for record in records if record["pair_region_type"] == "background_background"]
        self.assertEqual(len(cross_region_records), 2)
        self.assertEqual(len(background_records), 1)
        self.assertTrue(all(int(record["connected"]) == 0 for record in cross_region_records))
        self.assertTrue(all(record["geodesic_distance"] == "" for record in cross_region_records))
        self.assertEqual(int(background_records[0]["connected"]), 1)
        self.assertGreater(float(background_records[0]["geodesic_distance"]), 0.0)

    def test_sampling_methods_stay_inside_valid_mask(self) -> None:
        mask = np.zeros((48, 48), dtype=bool)
        mask[8:40, 8:40] = True
        distance_map = compute_distance_transform(mask)
        valid_mask = build_valid_mask(mask, distance_map, min_distance_to_boundary=3.0)
        rng = np.random.default_rng(42)

        for method in [
            "random",
            "euclidean_fps",
            "geodesic_fps",
            "feature_diversity",
            "euclidean_farthest",
            "geodesic_farthest",
            "boundary_crossing_targeted",
            "boundary_crossing_stratified",
            "feature_space_diverse",
            "hard_pair_targeted",
            "hard_pair_targeted_v2",
        ]:
            points = sample_points(
                valid_mask=valid_mask,
                distance_map=distance_map,
                method=method,
                n_points=8,
                candidate_pool_limit=2000,
                rng=rng,
            )
            self.assertEqual(len(points), 8)
            self.assertTrue(all(bool(valid_mask[row, col]) for row, col in points))

    def test_pair_selection_methods_return_requested_count(self) -> None:
        records = []
        for index in range(24):
            records.append(
                {
                    "pair_id": f"pair_{index}",
                    "connected": 1 if index % 2 == 0 else 0,
                    "euclidean_distance": float(5 + index),
                    "geodesic_distance": float(10 + index) if index % 2 == 0 else "",
                    "geodesic_euclidean_ratio": float(1.0 + index / 20.0) if index % 2 == 0 else "",
                    "boundary_crossing_count": index % 5,
                    "distance_to_boundary_min": float(1 + (index % 4)),
                    "line_inside_fraction": float((index % 6) / 5.0),
                    "component_gap_distance": float(index % 3) if index % 2 == 1 else "",
                    "max_outside_segment_length": float(index % 7),
                    "hard_pair_score": float(index),
                    "sampling_method": "manual",
                }
            )
        rng = np.random.default_rng(123)
        for method in [
            "boundary_crossing_targeted",
            "boundary_crossing_stratified",
            "feature_space_diverse",
            "hard_pair_targeted",
            "hard_pair_targeted_v2",
        ]:
            chosen = select_pair_records(records, method=method, target_count=10, rng=rng, feature_bin_count=4)
            self.assertEqual(len(chosen), 10)
            self.assertEqual(len({record["pair_id"] for record in chosen}), 10)

    def test_attach_region_pair_difficulty_marks_easy_and_hard_v2_flags(self) -> None:
        records = [
            {
                "pair_id": "easy_pos",
                "image_id": "img_a",
                "connected": 1,
                "euclidean_distance": 2.0,
                "geodesic_distance": 2.2,
                "geodesic_euclidean_ratio": 1.1,
                "boundary_crossing_count": 0,
                "distance_to_boundary_min": 4.0,
                "distance_to_boundary_i": 4.0,
                "distance_to_boundary_j": 4.0,
                "line_inside_fraction": 1.0,
                "component_gap_distance": "",
                "max_outside_segment_length": 0.0,
            },
            {
                "pair_id": "hard_pos",
                "image_id": "img_a",
                "connected": 1,
                "euclidean_distance": 5.0,
                "geodesic_distance": 20.0,
                "geodesic_euclidean_ratio": 4.0,
                "boundary_crossing_count": 3,
                "distance_to_boundary_min": 1.0,
                "distance_to_boundary_i": 1.0,
                "distance_to_boundary_j": 1.0,
                "line_inside_fraction": 0.8,
                "component_gap_distance": "",
                "max_outside_segment_length": 0.0,
            },
            {
                "pair_id": "easy_neg",
                "image_id": "img_a",
                "connected": 0,
                "euclidean_distance": 18.0,
                "geodesic_distance": "",
                "geodesic_euclidean_ratio": "",
                "boundary_crossing_count": 1,
                "distance_to_boundary_min": 3.0,
                "distance_to_boundary_i": 3.0,
                "distance_to_boundary_j": 3.0,
                "line_inside_fraction": 0.0,
                "component_gap_distance": "",
                "max_outside_segment_length": 6.0,
            },
            {
                "pair_id": "hard_neg",
                "image_id": "img_a",
                "connected": 0,
                "euclidean_distance": 6.0,
                "geodesic_distance": "",
                "geodesic_euclidean_ratio": "",
                "boundary_crossing_count": 2,
                "distance_to_boundary_min": 1.0,
                "distance_to_boundary_i": 1.0,
                "distance_to_boundary_j": 1.0,
                "line_inside_fraction": 0.9,
                "component_gap_distance": 1.0,
                "max_outside_segment_length": 1.0,
            },
        ]
        attach_region_pair_difficulty(records)
        by_id = {record["pair_id"]: record for record in records}

        self.assertEqual(int(by_id["easy_pos"]["is_easy_positive"]), 1)
        self.assertEqual(int(by_id["easy_neg"]["is_easy_negative"]), 1)
        self.assertEqual(int(by_id["easy_pos"]["easy_positive_v2"]), 1)
        self.assertEqual(int(by_id["easy_neg"]["easy_negative_v2"]), 1)
        self.assertEqual(int(by_id["hard_pos"]["is_hard_positive_v2"]), 1)
        self.assertEqual(int(by_id["hard_neg"]["is_hard_negative_v2"]), 1)
        self.assertGreater(float(by_id["hard_pos"]["hard_pair_score_v2"]), float(by_id["easy_pos"]["hard_pair_score_v2"]))

    def test_filter_easy_only_records_v2_balances_by_image(self) -> None:
        records = [
            {"pair_id": "img_a_pos_1", "image_id": "img_a", "easy_positive_v2": 1, "easy_negative_v2": 0, "connected": 1},
            {"pair_id": "img_a_pos_2", "image_id": "img_a", "easy_positive_v2": 1, "easy_negative_v2": 0, "connected": 1},
            {"pair_id": "img_a_neg_1", "image_id": "img_a", "easy_positive_v2": 0, "easy_negative_v2": 1, "connected": 0},
            {"pair_id": "img_b_pos_1", "image_id": "img_b", "easy_positive_v2": 1, "easy_negative_v2": 0, "connected": 1},
            {"pair_id": "img_b_neg_1", "image_id": "img_b", "easy_positive_v2": 0, "easy_negative_v2": 1, "connected": 0},
            {"pair_id": "img_b_neg_2", "image_id": "img_b", "easy_positive_v2": 0, "easy_negative_v2": 1, "connected": 0},
            {"pair_id": "img_c_pos_only", "image_id": "img_c", "easy_positive_v2": 1, "easy_negative_v2": 0, "connected": 1},
        ]
        chosen = filter_easy_only_records_v2(records, seed=0)
        self.assertEqual(sum(int(record["connected"]) for record in chosen), 2)
        self.assertEqual(len(chosen) - sum(int(record["connected"]) for record in chosen), 2)
        self.assertEqual({record["image_id"] for record in chosen}, {"img_a", "img_b"})


if __name__ == "__main__":
    unittest.main()
