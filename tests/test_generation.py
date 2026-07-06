from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from connectivity_experiment.config import load_config
from connectivity_experiment.generation import generate_sample_pair, generate_scene_query_samples


class GenerationTest(unittest.TestCase):
    def test_generate_sample_pair(self) -> None:
        config = load_config(ROOT / "configs" / "curved_v1.yaml")
        positive, negative = generate_sample_pair(config, split="train", pair_index=0, base_seed=42)

        self.assertEqual(positive["connected"], 1)
        self.assertEqual(negative["connected"], 0)
        self.assertEqual(positive["metadata"]["connected"], 1)
        self.assertEqual(negative["metadata"]["connected"], 0)
        self.assertGreater(float(positive["metadata"]["connection_margin"]), 0.0)
        self.assertLess(float(negative["metadata"]["signed_connection_margin"]), 0.0)
        self.assertIn(positive["metadata"]["shape_family"], {"straight", "bezier_bend", "c_shape"})
        self.assertEqual(positive["image"].mode, "RGB")
        self.assertIn(int(positive["metadata"]["connected_pixel_8"]), {0, 1})

    def test_generate_scene_query_samples(self) -> None:
        config = load_config(ROOT / "configs" / "multi_magatama_scene_v1.yaml")
        samples = generate_scene_query_samples(
            config=config,
            split="train",
            scene_index=0,
            base_seed=42,
            target_positive=3,
            target_negative=3,
        )

        self.assertGreaterEqual(len(samples), 2)
        self.assertTrue(all(sample["image"].mode == "RGB" for sample in samples))
        self.assertTrue(all("scene_id" in sample["metadata"] for sample in samples))
        self.assertTrue(all("same_source_pair" in sample["metadata"] for sample in samples))
        self.assertTrue(any(int(sample["metadata"]["same_source_pair"]) == 0 and int(sample["connected"]) == 1 for sample in samples))
        self.assertTrue(any(int(sample["connected"]) == 0 for sample in samples))


if __name__ == "__main__":
    unittest.main()
