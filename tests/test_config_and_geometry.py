from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from connectivity_experiment.config import load_config
from connectivity_experiment.geometry import Circle, build_adjacency, summarize_graph


class ConfigAndGeometryTest(unittest.TestCase):
    def test_load_simple_yaml(self) -> None:
        config = load_config(ROOT / "configs" / "default.yaml")
        self.assertEqual(config["dataset"]["image_size"], 128)
        self.assertEqual(config["dataset"]["radius_range"], [2, 8])
        curved = load_config(ROOT / "configs" / "curved_v1.yaml")
        self.assertIn("shape_families", curved["dataset"])
        self.assertEqual(curved["dataset"]["input_channels"], ["shape", "point_A", "point_B"])
        multi_scene = load_config(ROOT / "configs" / "multi_magatama_scene_v1.yaml")
        self.assertEqual(multi_scene["dataset"]["dataset_mode"], "scene_query")
        self.assertTrue(multi_scene["dataset"]["scene_require_cross_source_positive"])
        region_pairs = load_config(ROOT / "configs" / "region_pair_sampling_v1.yaml")
        self.assertEqual(region_pairs["sampling"]["methods"][0], "random")
        self.assertEqual(region_pairs["paths"]["output_dir"], "outputs/region_pair_sampling_v1")
        stare_subset_v2 = load_config(ROOT / "configs" / "region_pair_sampling_stare_subset_v2.yaml")
        self.assertTrue(stare_subset_v2["sampling"]["sample_background"])
        self.assertEqual(stare_subset_v2["sampling"]["pair_component_mode"], "all_pairs")
        instruction_subset = load_config(ROOT / "configs" / "region_pair_sampling_stare_instruction_subset_v1.yaml")
        self.assertIn("feature_space_diverse", instruction_subset["sampling"]["methods"])
        self.assertEqual(instruction_subset["sampling"]["n_pairs_per_image"], 96)
        instruction_subset_v2 = load_config(ROOT / "configs" / "region_pair_sampling_stare_instruction_subset_v2.yaml")
        self.assertIn("boundary_crossing_stratified", instruction_subset_v2["sampling"]["methods"])
        self.assertIn("hard_pair_targeted_v2", instruction_subset_v2["sampling"]["methods"])
        cnn_subset_v2 = load_config(ROOT / "configs" / "region_pair_cnn_stare_instruction_subset_v2.yaml")
        self.assertEqual(cnn_subset_v2["model"]["input_mode"], "mask_point_i_point_j")
        diagnostics_subset_v2 = load_config(ROOT / "configs" / "region_pair_followup_diagnostics_stare_instruction_subset_v2.yaml")
        self.assertEqual(len(diagnostics_subset_v2["diagnostics"]["ablation_input_modes"]), 4)
        positive_failure = load_config(ROOT / "configs" / "region_pair_positive_failure_diagnostics_stare_instruction_subset_v2.yaml")
        self.assertIn("mask_point_i_point_j_line", positive_failure["comparison"]["input_modes"])
        self.assertEqual(positive_failure["comparison"]["pool_sizes"], [4, 8])

    def test_connected_graph_summary(self) -> None:
        circles = [
            Circle(10.0, 10.0, 4.0, "target", 0),
            Circle(16.0, 10.0, 4.0, "path", 1),
            Circle(22.0, 10.0, 4.0, "target", 2),
        ]
        summary = summarize_graph(circles, 0, 2)
        self.assertEqual(summary["connected"], 1)
        self.assertEqual(summary["shortest_path_steps"], 2)
        self.assertGreater(summary["connection_margin"], 0)

    def test_disconnected_graph_summary(self) -> None:
        circles = [
            Circle(10.0, 10.0, 3.0, "target", 0),
            Circle(20.0, 10.0, 2.0, "path", 1),
            Circle(34.0, 10.0, 3.0, "target", 2),
        ]
        adjacency = build_adjacency(circles)
        self.assertEqual(adjacency[0], [])
        summary = summarize_graph(circles, 0, 2)
        self.assertEqual(summary["connected"], 0)
        self.assertLess(summary["connection_margin"], 0)


if __name__ == "__main__":
    unittest.main()
