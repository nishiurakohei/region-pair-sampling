"""Phase-1 encoders with reachability and beta1 scalar heads."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from models import BasicBlock, Sequential
from nn import BN2d, Conv2d, GAP, Linear, MaxPool2d, ReLU


class HeadsMixin:
    def heads_forward(self, h, train):
        self.last_h = h
        return self.reach.forward(h, train).ravel(), self.aux.forward(h, train).ravel()

    def heads_backward(self, greach, gaux):
        return self.reach.backward(greach[:, None]) + self.aux.backward(gaux[:, None])

    def params(self):
        d = self.encoder_params(); d.update(self.reach.params()); d.update(self.aux.params()); return d

    def grads(self):
        d = self.encoder_grads(); d.update(self.reach.grads()); d.update(self.aux.grads()); return d

    def zero_grad(self):
        self.encoder_zero_grad(); self.reach.zero_grad(); self.aux.zero_grad()


class MultiCNN(HeadsMixin):
    def __init__(self, rng):
        self.seq = Sequential([
            Conv2d(6, 16, 3, 1, 1, rng, "c1"), BN2d(16, "b1"), ReLU(), MaxPool2d(2),
            Conv2d(16, 32, 3, 1, 1, rng, "c2"), BN2d(32, "b2"), ReLU(), MaxPool2d(2),
            Conv2d(32, 64, 3, 1, 1, rng, "c3"), BN2d(64, "b3"), ReLU(), GAP(),
        ])
        self.reach, self.aux = Linear(64, 1, rng, "reach"), Linear(64, 1, rng, "aux")

    def forward(self, x, train=True): return self.heads_forward(self.seq.forward(x, train), train)
    def backward(self, gr, ga): self.seq.backward(self.heads_backward(gr, ga))
    def encoder_params(self): return self.seq._collect("params")
    def encoder_grads(self): return self.seq._collect("grads")
    def encoder_zero_grad(self):
        for layer in self.seq.layers: layer.zero_grad()
    def bn_state(self):
        d = {}
        for layer in self.seq.layers:
            if isinstance(layer, BN2d): d.update(layer.state())
        return d
    def load_bn_state(self, state):
        for layer in self.seq.layers:
            if isinstance(layer, BN2d): layer.load_state(state)


class MultiResNet(HeadsMixin):
    def __init__(self, rng):
        channels, blocks = (16, 32, 64), (2, 2, 2)
        self.stem_c = Conv2d(6, channels[0], 3, 1, 1, rng, "stem.c")
        self.stem_b, self.stem_r = BN2d(channels[0], "stem.b"), ReLU()
        self.blocks = []
        prev = channels[0]
        for si, (count, width) in enumerate(zip(blocks, channels)):
            for bi in range(count):
                self.blocks.append(BasicBlock(prev, width, 2 if bi == 0 else 1, rng, f"s{si}b{bi}"))
                prev = width
        self.gap = GAP()
        self.reach, self.aux = Linear(prev, 1, rng, "reach"), Linear(prev, 1, rng, "aux")

    def encoder_layers(self):
        layers = [self.stem_c, self.stem_b]
        for block in self.blocks: layers += block.layers()
        return layers
    def forward(self, x, train=True):
        h = self.stem_r.forward(self.stem_b.forward(self.stem_c.forward(x, train), train))
        for block in self.blocks: h = block.forward(h, train)
        return self.heads_forward(self.gap.forward(h, train), train)
    def backward(self, gr, ga):
        g = self.gap.backward(self.heads_backward(gr, ga))
        for block in reversed(self.blocks): g = block.backward(g)
        self.stem_c.backward(self.stem_b.backward(self.stem_r.backward(g)))
    def encoder_params(self):
        d = {}
        for layer in self.encoder_layers(): d.update(layer.params())
        return d
    def encoder_grads(self):
        d = {}
        for layer in self.encoder_layers(): d.update(layer.grads())
        return d
    def encoder_zero_grad(self):
        for layer in self.encoder_layers(): layer.zero_grad()
        self.stem_r.zero_grad(); self.gap.zero_grad()
    def bn_state(self):
        d = {}
        for layer in self.encoder_layers():
            if isinstance(layer, BN2d): d.update(layer.state())
        return d
    def load_bn_state(self, state):
        for layer in self.encoder_layers():
            if isinstance(layer, BN2d): layer.load_state(state)


def build(name, rng):
    return MultiCNN(rng) if name == "cnn" else MultiResNet(rng)
