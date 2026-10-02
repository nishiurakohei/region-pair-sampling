"""Fully observed six-channel rasterization and Hard-Maze data loading."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N = 15
L = 2 * N - 1
C = 6
START, GOAL = (0, 0), (N - 1, N - 1)

_nr, _nc = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
NODE_R, NODE_C = (2 * _nr).ravel(), (2 * _nc).ravel()
_hr, _hc = np.meshgrid(np.arange(N), np.arange(N - 1), indexing="ij")
HR, HC = (2 * _hr).ravel(), (2 * _hc + 1).ravel()
_vr, _vc = np.meshgrid(np.arange(N - 1), np.arange(N), indexing="ij")
VR, VC = (2 * _vr + 1).ravel(), (2 * _vc).ravel()


def rasterize_batch(H, V):
    """Same Phase-1 representation with every edge observed (unknown channel is zero)."""
    b = H.shape[0]
    x = np.zeros((b, C, L, L), dtype=np.float32)
    x[:, 0, NODE_R, NODE_C] = 1.0
    hf, vf = H.reshape(b, -1), V.reshape(b, -1)
    x[:, 1, HR, HC] = hf
    x[:, 2, HR, HC] = ~hf
    x[:, 1, VR, VC] = vf
    x[:, 2, VR, VC] = ~vf
    x[:, 4, 0, 0] = 1.0
    x[:, 5, -1, -1] = 1.0
    return x


class MazeData:
    def __init__(self, level, split):
        z = np.load(ROOT / "data" / level / f"{split}.npz", allow_pickle=False)
        self.arrays = {k: z[k] for k in z.files}
        self.H, self.V = self.arrays["H"], self.arrays["V"]
        self.y = self.arrays["y"].astype(np.float32)
        self.n = len(self.y)

    def raster_all(self):
        return rasterize_batch(self.H, self.V)
