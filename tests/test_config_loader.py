from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from connectivity_experiment.config import load_config


class ConfigLoaderTest(unittest.TestCase):
    def test_load_config_supports_block_lists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "methods:",
                        "  - random",
                        "  - geodesic_farthest",
                        "point_neighborhood_mask:",
                        "  target:",
                        "    - point_a",
                        "    - point_b",
                    ]
                ),
                encoding="utf-8",
            )
            config = load_config(config_path)

        self.assertEqual(config["methods"], ["random", "geodesic_farthest"])
        self.assertEqual(config["point_neighborhood_mask"]["target"], ["point_a", "point_b"])


if __name__ == "__main__":
    unittest.main()
