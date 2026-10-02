"""MLX backend for the published DT-Net; weights round-trip to PyTorch."""
from pathlib import Path
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
UP = HERE / "upstream" / "maze-extrapolation"
sys.path.insert(0, str(UP))
from src.models.dt_net import DTNet as TorchDTNet


class BasicBlock(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1, bias=False)

    def __call__(self, x):
        return nn.relu(self.conv2(nn.relu(self.conv1(x))) + x)


class Base(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.projection = nn.Conv2d(3, width, 3, padding=1, bias=False)
        self.head1 = nn.Conv2d(width, 32, 3, padding=1, bias=False)
        self.head2 = nn.Conv2d(32, 8, 3, padding=1, bias=False)
        self.head3 = nn.Conv2d(8, 2, 3, padding=1, bias=False)

    def project(self, x): return nn.relu(self.projection(x))

    def head(self, x): return self.head3(nn.relu(self.head2(nn.relu(self.head1(x)))))


class Recurrent(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.input = nn.Conv2d(width + 3, width, 3, padding=1, bias=False)
        self.block0 = BasicBlock(width)
        self.block1 = BasicBlock(width)

    def __call__(self, z, x):
        z = self.input(mx.concatenate([z, x], axis=-1))
        return self.block1(self.block0(z))


class DTNet(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.base = Base(width)
        self.recur = Recurrent(width)

    def __call__(self, x, iterations, interim=None, capture=0):
        z = self.base.project(x) if interim is None else interim
        captured = None
        for step in range(iterations):
            z = self.recur(z, x)
            if step + 1 == capture: captured = z
        return self.base.head(z), z, captured


MAPPING = {
    "projection.0.weight": ("base", "projection"),
    "recur_block.0.weight": ("recur", "input"),
    "recur_block.1.0.conv1.weight": ("recur", "block0", "conv1"),
    "recur_block.1.0.conv2.weight": ("recur", "block0", "conv2"),
    "recur_block.1.1.conv1.weight": ("recur", "block1", "conv1"),
    "recur_block.1.1.conv2.weight": ("recur", "block1", "conv2"),
    "head.0.weight": ("base", "head1"),
    "head.2.weight": ("base", "head2"),
    "head.4.weight": ("base", "head3"),
}


def _module(model, path):
    value = model
    for name in path: value = getattr(value, name)
    return value


def import_torch_state(model, state):
    clean = {k.replace("module.", ""): v for k, v in state.items()}
    for key, path in MAPPING.items():
        _module(model, path).weight = mx.array(clean[key].detach().cpu().numpy().transpose(0, 2, 3, 1))
    mx.eval(model.parameters())


def export_torch_state(model):
    state = {}
    for key, path in MAPPING.items():
        weight = np.array(_module(model, path).weight).transpose(0, 3, 1, 2).copy()
        state[key] = torch.from_numpy(weight)
    return state


def torch_initialized(model_seed):
    torch.manual_seed(model_seed)
    torch_model = TorchDTNet()
    model = DTNet()
    import_torch_state(model, torch_model.state_dict())
    return model


def self_check():
    state = torch.load(UP / "models" / "dt_net.pth", map_location="cpu", weights_only=True)["net"]
    torch_model = TorchDTNet().eval()
    torch_model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    model = DTNet(); import_torch_state(model, state)
    rng = np.random.default_rng(7); x = rng.normal(size=(2, 3, 24, 24)).astype(np.float32)
    with torch.inference_mode(): expected = torch_model(torch.from_numpy(x), 30)[0].permute(0, 2, 3, 1).numpy()
    actual = np.array(model(mx.array(x.transpose(0, 2, 3, 1)), 30)[0])
    error = float(np.max(np.abs(expected - actual)))
    assert error < 2e-4, error
    roundtrip = export_torch_state(model)
    assert all(torch.equal(roundtrip[k], v) for k, v in {k.replace("module.", ""): v for k, v in state.items()}.items())
    print({"forward_max_abs_error": error, "argmax_equal": bool(np.array_equal(expected.argmax(-1), actual.argmax(-1))), "weight_roundtrip_exact": True})


if __name__ == "__main__": self_check()
