"""Minimal pure-numpy deep-learning stack: Conv2d (im2col via sliding_window_view),
BatchNorm2d, ReLU, MaxPool2d(2), global average pool, Linear, residual blocks,
Adam, BCE-with-logits. Enough to build a small CNN and a ResNet-style net.

Convention: forward(x, train) caches what backward needs. backward(grad) returns
grad wrt input and accumulates parameter grads into self.grads. Params/grads are
exposed via params()/grads() as dict name->array (shared object identity)."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

He = lambda shape, fan_in, rng: rng.standard_normal(shape).astype(np.float32) * np.float32(np.sqrt(2.0 / fan_in))


class Conv2d:
    def __init__(self, cin, cout, k=3, stride=1, pad=1, rng=None, name="conv"):
        self.k, self.stride, self.pad, self.name = k, stride, pad, name
        self.W = He((cout, cin, k, k), cin * k * k, rng)
        self.b = np.zeros(cout, dtype=np.float32)
        self.dW = np.zeros_like(self.W); self.db = np.zeros_like(self.b)

    def forward(self, x, train=True):
        B, C, H, Wd = x.shape
        k, s, p = self.k, self.stride, self.pad
        xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
        win = sliding_window_view(xp, (k, k), axis=(2, 3))      # B,C,Ho',Wo',k,k
        win = win[:, :, ::s, ::s, :, :]
        Ho, Wo = win.shape[2], win.shape[3]
        # cols: (B*Ho*Wo, C*k*k)
        cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(B * Ho * Wo, C * k * k)
        Wm = self.W.reshape(self.W.shape[0], -1)               # (Cout, C*k*k)
        out = cols @ Wm.T + self.b                              # (B*Ho*Wo, Cout)
        out = out.reshape(B, Ho, Wo, -1).transpose(0, 3, 1, 2)
        if train:
            self.cache = (cols, x.shape, Ho, Wo)
        return out

    def backward(self, grad):
        cols, xshape, Ho, Wo = self.cache
        B, C, H, Wd = xshape
        k, s, p = self.k, self.stride, self.pad
        Cout = self.W.shape[0]
        g = grad.transpose(0, 2, 3, 1).reshape(-1, Cout)       # (B*Ho*Wo, Cout)
        Wm = self.W.reshape(Cout, -1)
        self.dW += (g.T @ cols).reshape(self.W.shape)
        self.db += g.sum(axis=0)
        dcols = g @ Wm                                          # (B*Ho*Wo, C*k*k)
        # col2im
        dcols = dcols.reshape(B, Ho, Wo, C, k, k).transpose(0, 3, 4, 5, 1, 2)
        dxp = np.zeros((B, C, H + 2 * p, Wd + 2 * p), dtype=np.float32)
        for i in range(k):
            for j in range(k):
                dxp[:, :, i:i + s * Ho:s, j:j + s * Wo:s] += dcols[:, :, i, j]
        return dxp[:, :, p:p + H, p:p + Wd]

    def params(self): return {f"{self.name}.W": self.W, f"{self.name}.b": self.b}
    def grads(self): return {f"{self.name}.W": self.dW, f"{self.name}.b": self.db}
    def zero_grad(self): self.dW[...] = 0; self.db[...] = 0


class BN2d:
    def __init__(self, C, name="bn", mom=0.9):
        self.g = np.ones(C, dtype=np.float32); self.beta = np.zeros(C, dtype=np.float32)
        self.rm = np.zeros(C, dtype=np.float32); self.rv = np.ones(C, dtype=np.float32)
        self.dg = np.zeros(C, dtype=np.float32); self.dbeta = np.zeros(C, dtype=np.float32)
        self.mom = mom; self.name = name; self.eps = 1e-5

    def forward(self, x, train=True):
        if train:
            mu = x.mean(axis=(0, 2, 3))
            var = x.var(axis=(0, 2, 3))
            self.rm = self.mom * self.rm + (1 - self.mom) * mu
            self.rv = self.mom * self.rv + (1 - self.mom) * var
            xc = x - mu[None, :, None, None]
            std = np.sqrt(var + self.eps)[None, :, None, None]
            xn = xc / std
            self.cache = (xn, std, x.shape)
        else:
            xn = (x - self.rm[None, :, None, None]) / np.sqrt(self.rv + self.eps)[None, :, None, None]
        return self.g[None, :, None, None] * xn + self.beta[None, :, None, None]

    def backward(self, grad):
        xn, std, shape = self.cache
        B, C, H, W = shape
        M = B * H * W
        self.dg += (grad * xn).sum(axis=(0, 2, 3))
        self.dbeta += grad.sum(axis=(0, 2, 3))
        dxn = grad * self.g[None, :, None, None]
        dxn_sum = dxn.sum(axis=(0, 2, 3))[None, :, None, None]
        dxn_xn = (dxn * xn).sum(axis=(0, 2, 3))[None, :, None, None]
        dx = (dxn - dxn_sum / M - xn * dxn_xn / M) / std
        return dx

    def params(self): return {f"{self.name}.g": self.g, f"{self.name}.beta": self.beta}
    def grads(self): return {f"{self.name}.g": self.dg, f"{self.name}.beta": self.dbeta}
    def zero_grad(self): self.dg[...] = 0; self.dbeta[...] = 0
    def state(self): return {f"{self.name}.rm": self.rm, f"{self.name}.rv": self.rv}
    def load_state(self, d): self.rm = d[f"{self.name}.rm"]; self.rv = d[f"{self.name}.rv"]


class ReLU:
    def forward(self, x, train=True):
        self.mask = x > 0
        return x * self.mask
    def backward(self, grad): return grad * self.mask
    def params(self): return {}
    def grads(self): return {}
    def zero_grad(self): pass


class MaxPool2d:
    def __init__(self, k=2):
        self.k = k
    def forward(self, x, train=True):
        B, C, H, W = x.shape
        k = self.k
        Ho, Wo = H // k, W // k
        xr = x[:, :, :Ho * k, :Wo * k].reshape(B, C, Ho, k, Wo, k)
        xr = xr.transpose(0, 1, 2, 4, 3, 5).reshape(B, C, Ho, Wo, k * k)
        idx = xr.argmax(axis=-1)
        out = np.take_along_axis(xr, idx[..., None], axis=-1)[..., 0]
        self.cache = (idx, x.shape, Ho, Wo)
        return out
    def backward(self, grad):
        idx, shape, Ho, Wo = self.cache
        B, C, H, W = shape
        k = self.k
        dxr = np.zeros((B, C, Ho, Wo, k * k), dtype=np.float32)
        np.put_along_axis(dxr, idx[..., None], grad[..., None], axis=-1)
        dxr = dxr.reshape(B, C, Ho, Wo, k, k).transpose(0, 1, 2, 4, 3, 5)
        dx = np.zeros(shape, dtype=np.float32)
        dx[:, :, :Ho * k, :Wo * k] = dxr.reshape(B, C, Ho * k, Wo * k)
        return dx
    def params(self): return {}
    def grads(self): return {}
    def zero_grad(self): pass


class GAP:  # global average pool -> (B,C)
    def forward(self, x, train=True):
        self.shape = x.shape
        return x.mean(axis=(2, 3))
    def backward(self, grad):
        B, C, H, W = self.shape
        return np.broadcast_to(grad[:, :, None, None] / (H * W), self.shape).astype(np.float32)
    def params(self): return {}
    def grads(self): return {}
    def zero_grad(self): pass


class Linear:
    def __init__(self, cin, cout, rng, name="fc"):
        self.W = He((cin, cout), cin, rng); self.b = np.zeros(cout, dtype=np.float32)
        self.dW = np.zeros_like(self.W); self.db = np.zeros_like(self.b); self.name = name
    def forward(self, x, train=True):
        self.x = x
        return x @ self.W + self.b
    def backward(self, grad):
        self.dW += self.x.T @ grad
        self.db += grad.sum(axis=0)
        return grad @ self.W.T
    def params(self): return {f"{self.name}.W": self.W, f"{self.name}.b": self.b}
    def grads(self): return {f"{self.name}.W": self.dW, f"{self.name}.b": self.db}
    def zero_grad(self): self.dW[...] = 0; self.db[...] = 0


# ---------------------------------------------------------------------------
def bce_with_logits(logits, y):
    """logits,(B,) y,(B,). Returns loss (scalar), grad wrt logits (B,)."""
    z = logits.ravel()
    # stable: max(z,0) - z*y + log(1+exp(-|z|))
    loss = np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z))))
    p = 1.0 / (1.0 + np.exp(-z))
    grad = (p - y) / y.shape[0]
    return loss, grad


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), wd=0.0):
        self.params = params  # dict name->array
        self.lr, self.b1, self.b2, self.wd = lr, betas[0], betas[1], wd
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0
    def step(self, grads):
        self.t += 1
        b1, b2 = self.b1, self.b2
        for k, p in self.params.items():
            g = grads[k]
            if self.wd and (k.endswith(".W")):
                p *= (1 - self.lr * self.wd)
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * (g * g)
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + 1e-8)
    def state(self):
        d = {"t": np.array(self.t)}
        for k in self.params:
            d[f"m::{k}"] = self.m[k]; d[f"v::{k}"] = self.v[k]
        return d
    def load_state(self, d):
        self.t = int(d["t"])
        for k in self.params:
            self.m[k] = d[f"m::{k}"]; self.v[k] = d[f"v::{k}"]
