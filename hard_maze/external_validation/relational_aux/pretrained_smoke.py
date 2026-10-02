"""Fail-fast compatibility check for the pinned pretrained DT-Net environment."""
import argparse, hashlib, json, sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
UP = HERE / "upstream" / "maze-extrapolation"
sys.path.insert(0, str(UP))
from src.models.dt_net import DTNet
from stage0_reproduce import evaluate, make_data

EXPECTED_WEIGHT = "b8beed72aad07b300e178f74f0c3628886cfb28fca5d271c451496af38478216"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE.parent / "results_external" / "pretrained_smoke.json")
    parser.add_argument("--num-mazes", type=int, default=100)
    args = parser.parse_args()
    weight = UP / "models" / "dt_net.pth"
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    if digest != EXPECTED_WEIGHT:
        raise RuntimeError(f"pretrained SHA-256 mismatch: {digest}")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    saved = torch.load(weight, map_location="cpu", weights_only=True)["net"]
    model = DTNet().eval().to(device)
    model.load_state_dict({key.replace("module.", ""): value for key, value in saved.items()}, strict=True)
    x, y, dataset_hash = make_data(9, args.num_mazes, 0.0, True, 42, HERE / "data")
    accuracy = evaluate(model, x, y, [30], 20, device)[30][0]
    result = {"passed": accuracy >= .95, "accuracy": accuracy, "threshold": .95,
              "num_mazes": args.num_mazes, "dataset_seed": 42, "iterations": 30,
              "device": str(device), "torch": torch.__version__,
              "weight_sha256": digest, "dataset_sha256": dataset_hash}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if not result["passed"]:
        raise SystemExit("pretrained compatibility smoke test failed")


if __name__ == "__main__":
    main()
