"""SmallCNN and a ResNet-18-topology residual network (adapted for 29x29 input
and pure-numpy/CPU compute). Both output logits of shape (B,)."""
import numpy as np
from nn import Conv2d, BN2d, ReLU, MaxPool2d, GAP, Linear


class Sequential:
    def __init__(self, layers):
        self.layers = layers
    def forward(self, x, train=True):
        for l in self.layers:
            x = l.forward(x, train)
        return x
    def backward(self, g):
        for l in reversed(self.layers):
            g = l.backward(g)
        return g
    def _collect(self, meth):
        d = {}
        for l in self.layers:
            d.update(getattr(l, meth)())
        return d


class SmallCNN:
    def __init__(self, cin=6, rng=None):
        self.seq = Sequential([
            Conv2d(cin, 16, 3, 1, 1, rng, "c1"), BN2d(16, "b1"), ReLU(), MaxPool2d(2),   # 29->14
            Conv2d(16, 32, 3, 1, 1, rng, "c2"), BN2d(32, "b2"), ReLU(), MaxPool2d(2),    # 14->7
            Conv2d(32, 64, 3, 1, 1, rng, "c3"), BN2d(64, "b3"), ReLU(),                  # 7
            GAP(),
        ])
        self.fc = Linear(64, 1, rng, "fc")
    def forward(self, x, train=True):
        h = self.seq.forward(x, train)
        return self.fc.forward(h, train).ravel()
    def backward(self, glogit):
        g = self.fc.backward(glogit[:, None])
        self.seq.backward(g)
    def params(self):
        d = self.seq._collect("params"); d.update(self.fc.params()); return d
    def grads(self):
        d = self.seq._collect("grads"); d.update(self.fc.grads()); return d
    def zero_grad(self):
        for l in self.seq.layers: l.zero_grad()
        self.fc.zero_grad()
    def bn_state(self):
        d = {}
        for l in self.seq.layers:
            if isinstance(l, BN2d): d.update(l.state())
        return d
    def load_bn_state(self, z):
        for l in self.seq.layers:
            if isinstance(l, BN2d): l.load_state(z)


class BasicBlock:
    def __init__(self, cin, cout, stride, rng, pfx):
        self.c1 = Conv2d(cin, cout, 3, stride, 1, rng, f"{pfx}.c1")
        self.b1 = BN2d(cout, f"{pfx}.b1"); self.r1 = ReLU()
        self.c2 = Conv2d(cout, cout, 3, 1, 1, rng, f"{pfx}.c2")
        self.b2 = BN2d(cout, f"{pfx}.b2"); self.r2 = ReLU()
        self.proj = (stride != 1 or cin != cout)
        if self.proj:
            self.cs = Conv2d(cin, cout, 1, stride, 0, rng, f"{pfx}.cs")
            self.bs = BN2d(cout, f"{pfx}.bs")
    def forward(self, x, train=True):
        h = self.r1.forward(self.b1.forward(self.c1.forward(x, train), train))
        h = self.b2.forward(self.c2.forward(h, train), train)
        s = self.bs.forward(self.cs.forward(x, train), train) if self.proj else x
        return self.r2.forward(h + s)
    def backward(self, g):
        g = self.r2.backward(g)
        gs = g  # into shortcut
        gh = g  # into main
        gh = self.c2.backward(self.b2.backward(gh))
        gh = self.c1.backward(self.b1.backward(self.r1.backward(gh)))
        if self.proj:
            gsx = self.cs.backward(self.bs.backward(gs))
        else:
            gsx = gs
        return gh + gsx
    def layers(self):
        ls = [self.c1, self.b1, self.c2, self.b2]
        if self.proj: ls += [self.cs, self.bs]
        return ls


class ResNet:
    def __init__(self, cin=6, blocks=(2, 2, 2, 2), channels=(16, 32, 64, 128), rng=None):
        self.stem_c = Conv2d(cin, channels[0], 3, 1, 1, rng, "stem.c")
        self.stem_b = BN2d(channels[0], "stem.b"); self.stem_r = ReLU()
        self.blocks = []
        prev = channels[0]
        for si, (nb, ch) in enumerate(zip(blocks, channels)):
            for bi in range(nb):
                stride = 2 if bi == 0 else 1   # downsample at first block of every stage
                self.blocks.append(BasicBlock(prev, ch, stride, rng, f"s{si}b{bi}"))
                prev = ch
        self.gap = GAP()
        self.fc = Linear(prev, 1, rng, "fc")
    def forward(self, x, train=True):
        h = self.stem_r.forward(self.stem_b.forward(self.stem_c.forward(x, train), train))
        for blk in self.blocks:
            h = blk.forward(h, train)
        h = self.gap.forward(h, train)
        return self.fc.forward(h, train).ravel()
    def backward(self, glogit):
        g = self.fc.backward(glogit[:, None])
        g = self.gap.backward(g)
        for blk in reversed(self.blocks):
            g = blk.backward(g)
        self.stem_c.backward(self.stem_b.backward(self.stem_r.backward(g)))
    def _all_layers(self):
        ls = [self.stem_c, self.stem_b]
        for blk in self.blocks: ls += blk.layers()
        ls += [self.fc]
        return ls
    def params(self):
        d = {}
        for l in self._all_layers(): d.update(l.params())
        return d
    def grads(self):
        d = {}
        for l in self._all_layers(): d.update(l.grads())
        return d
    def zero_grad(self):
        for l in self._all_layers(): l.zero_grad()
        self.stem_r.zero_grad(); self.gap.zero_grad()
    def bn_state(self):
        d = {}
        for l in self._all_layers():
            if isinstance(l, BN2d): d.update(l.state())
        return d
    def load_bn_state(self, z):
        for l in self._all_layers():
            if isinstance(l, BN2d): l.load_state(z)


def build(name, rng):
    if name == "cnn":
        return SmallCNN(6, rng)
    if name == "resnet":
        return ResNet(6, blocks=(2, 2, 2), channels=(16, 32, 64), rng=rng)
    raise ValueError(name)


def count_params(model):
    return int(sum(v.size for v in model.params().values()))
