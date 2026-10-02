"""Phase-1 encoders with a global main head and spatial node-pair head."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from models import BasicBlock, Sequential
from nn import BN2d, Conv2d, Linear, MaxPool2d, ReLU

N = 15


class PairHeads:
    def heads_forward(self, fmap, pairs, train):
        self.fmap_shape = fmap.shape
        self.global_h = fmap.mean(axis=(2, 3))
        main = self.main.forward(self.global_h, train).ravel()
        if pairs is None:
            return main, None
        b, k = pairs.shape[:2]
        h, w = fmap.shape[2:]
        u, v = pairs[..., 0], pairs[..., 1]
        self.ur = np.rint((u // N) * (h - 1) / (N - 1)).astype(int)
        self.uc = np.rint((u % N) * (w - 1) / (N - 1)).astype(int)
        self.vr = np.rint((v // N) * (h - 1) / (N - 1)).astype(int)
        self.vc = np.rint((v % N) * (w - 1) / (N - 1)).astype(int)
        bi = np.arange(b)[:, None]
        self.hu = fmap[bi, :, self.ur, self.uc]
        self.hv = fmap[bi, :, self.vr, self.vc]
        repeated_global = np.broadcast_to(self.global_h[:, None], (b, k, self.global_h.shape[1]))
        feat = np.concatenate((self.hu, self.hv, np.abs(self.hu-self.hv),
                               self.hu*self.hv, repeated_global), axis=2)
        pair_logits = self.pair.forward(feat.reshape(b*k, -1), train).reshape(b, k)
        return main, pair_logits

    def heads_backward(self, gmain, gpair):
        b, c, h, w = self.fmap_shape
        gmap = np.zeros(self.fmap_shape, np.float32)
        gglobal = self.main.backward(gmain[:, None])
        if gpair is not None:
            k = gpair.shape[1]
            gf = self.pair.backward(gpair.reshape(-1, 1)).reshape(b, k, 5*c)
            a, d, diff, prod, glob = np.split(gf, 5, axis=2)
            sign = np.sign(self.hu-self.hv)
            gu = a + diff*sign + prod*self.hv
            gv = d - diff*sign + prod*self.hu
            gglobal += glob.sum(axis=1)
            bi = np.arange(b)
            for j in range(k):
                np.add.at(gmap, (bi, slice(None), self.ur[:, j], self.uc[:, j]), gu[:, j])
                np.add.at(gmap, (bi, slice(None), self.vr[:, j], self.vc[:, j]), gv[:, j])
        gmap += gglobal[:, :, None, None] / (h*w)
        return gmap

    def params(self):
        d = self.encoder_params(); d.update(self.main.params()); d.update(self.pair.params()); return d
    def grads(self):
        d = self.encoder_grads(); d.update(self.main.grads()); d.update(self.pair.grads()); return d
    def zero_grad(self):
        self.encoder_zero_grad(); self.main.zero_grad(); self.pair.zero_grad()


class PairCNN(PairHeads):
    def __init__(self, rng):
        self.seq = Sequential([
            Conv2d(6,16,3,1,1,rng,"c1"), BN2d(16,"b1"), ReLU(), MaxPool2d(2),
            Conv2d(16,32,3,1,1,rng,"c2"), BN2d(32,"b2"), ReLU(), MaxPool2d(2),
            Conv2d(32,64,3,1,1,rng,"c3"), BN2d(64,"b3"), ReLU(),
        ])
        self.main = Linear(64,1,rng,"main")
        self.pair = Linear(5*64,1,rng,"pair")
    def forward(self, x, pairs=None, train=True): return self.heads_forward(self.seq.forward(x,train),pairs,train)
    def backward(self, gm, gp): self.seq.backward(self.heads_backward(gm,gp))
    def encoder_params(self): return self.seq._collect("params")
    def encoder_grads(self): return self.seq._collect("grads")
    def encoder_zero_grad(self):
        for x in self.seq.layers: x.zero_grad()
    def bn_state(self):
        d={}
        for x in self.seq.layers:
            if isinstance(x,BN2d): d.update(x.state())
        return d
    def load_bn_state(self,state):
        for x in self.seq.layers:
            if isinstance(x,BN2d): x.load_state(state)


class PairResNet(PairHeads):
    def __init__(self,rng):
        channels, counts=(16,32,64),(2,2,2)
        self.stem_c=Conv2d(6,16,3,1,1,rng,"stem.c")
        self.stem_b,self.stem_r=BN2d(16,"stem.b"),ReLU()
        self.blocks=[]; prev=16
        for si,(count,width) in enumerate(zip(counts,channels)):
            for bi in range(count):
                self.blocks.append(BasicBlock(prev,width,2 if bi==0 else 1,rng,f"s{si}b{bi}")); prev=width
        self.main=Linear(prev,1,rng,"main")
        self.pair=Linear(5*prev,1,rng,"pair")
    def encoder_layers(self):
        out=[self.stem_c,self.stem_b]
        for block in self.blocks: out += block.layers()
        return out
    def forward(self,x,pairs=None,train=True):
        h=self.stem_r.forward(self.stem_b.forward(self.stem_c.forward(x,train),train))
        for block in self.blocks: h=block.forward(h,train)
        return self.heads_forward(h,pairs,train)
    def backward(self,gm,gp):
        g=self.heads_backward(gm,gp)
        for block in reversed(self.blocks): g=block.backward(g)
        self.stem_c.backward(self.stem_b.backward(self.stem_r.backward(g)))
    def encoder_params(self):
        d={}
        for x in self.encoder_layers(): d.update(x.params())
        return d
    def encoder_grads(self):
        d={}
        for x in self.encoder_layers(): d.update(x.grads())
        return d
    def encoder_zero_grad(self):
        for x in self.encoder_layers(): x.zero_grad()
        self.stem_r.zero_grad()
    def bn_state(self):
        d={}
        for x in self.encoder_layers():
            if isinstance(x,BN2d): d.update(x.state())
        return d
    def load_bn_state(self,state):
        for x in self.encoder_layers():
            if isinstance(x,BN2d): x.load_state(state)


def build(name,rng): return PairCNN(rng) if name=="cnn" else PairResNet(rng)


def self_check():
    rng=np.random.default_rng(1); x=rng.normal(size=(3,6,29,29)).astype(np.float32)
    pairs=np.array([[[1,223],[16,208]],[[2,222],[17,207]],[[3,221],[18,206]]],np.int16)
    for name in ("cnn","resnet"):
        m=build(name,np.random.default_rng(2)); main,pair=m.forward(x,pairs,True)
        assert main.shape==(3,) and pair.shape==(3,2) and np.isfinite(pair).all()
        m.zero_grad(); m.backward(np.ones(3,np.float32)/3,np.ones((3,2),np.float32)/6)
        assert all(np.isfinite(g).all() for g in m.grads().values())
    print("PAIR_MODELS_OK")


if __name__=="__main__": self_check()
