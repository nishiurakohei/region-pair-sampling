"""
Maze generation, exact reachability, masking, rasterization, and Bayes oracle.

Latent maze G = (V, E): N x N grid of cells. Undirected edges between 4-adjacent
cells. Each edge independently open with probability p (Bernoulli bond percolation).
Main label y = 1{ start <-> goal } computed by exact BFS over OPEN edges.

Edge storage (canonical, seed-reproducible):
  H : bool array (N, N-1)   H[r,c] = edge between (r,c) and (r,c+1)   (horizontal)
  V : bool array (N-1, N)   V[r,c] = edge between (r,c) and (r+1,c)   (vertical)
True = open/passable.

Masking produces "known" masks of identical shape (True = observed, False = unknown).

Rasterization -> (C, L, L) float32 image, L = 2N-1. Cells at even/even pixels,
edges at the pixel between the two incident cells. Channels:
  0 node      : 1 at every cell pixel (constant grid geometry aid)
  1 known_open: 1 at edge pixel if observed AND open
  2 known_wall: 1 at edge pixel if observed AND closed
  3 unknown   : 1 at edge pixel if NOT observed   <-- distinct from wall
  4 start     : 1 at start cell pixel
  5 goal      : 1 at goal cell pixel
"""
import numpy as np
from collections import deque

N_DEFAULT = 15
N_CHANNELS = 6


# ----------------------------------------------------------------------------
# generation
# ----------------------------------------------------------------------------
def sample_maze(N, p, rng):
    """Sample a latent maze (H, V) via Bernoulli(p) bond percolation."""
    H = rng.random((N, N - 1)) < p
    V = rng.random((N - 1, N)) < p
    return H, V


def reachable(H, V, start, goal):
    """Exact BFS reachability over open edges. start/goal are (r,c) tuples."""
    N = H.shape[0]
    if start == goal:
        return True
    seen = np.zeros((N, N), dtype=bool)
    seen[start] = True
    q = deque([start])
    while q:
        r, c = q.popleft()
        # right neighbor (r,c)-(r,c+1) via H[r,c]
        if c + 1 < N and H[r, c] and not seen[r, c + 1]:
            if (r, c + 1) == goal:
                return True
            seen[r, c + 1] = True
            q.append((r, c + 1))
        # left neighbor (r,c-1)-(r,c) via H[r,c-1]
        if c - 1 >= 0 and H[r, c - 1] and not seen[r, c - 1]:
            if (r, c - 1) == goal:
                return True
            seen[r, c - 1] = True
            q.append((r, c - 1))
        # down neighbor (r,c)-(r+1,c) via V[r,c]
        if r + 1 < N and V[r, c] and not seen[r + 1, c]:
            if (r + 1, c) == goal:
                return True
            seen[r + 1, c] = True
            q.append((r + 1, c))
        # up neighbor (r-1,c)-(r,c) via V[r-1,c]
        if r - 1 >= 0 and V[r - 1, c] and not seen[r - 1, c]:
            if (r - 1, c) == goal:
                return True
            seen[r - 1, c] = True
            q.append((r - 1, c))
    return False


def default_endpoints(N):
    return (0, 0), (N - 1, N - 1)


# ----------------------------------------------------------------------------
# masking
# ----------------------------------------------------------------------------
def sample_mask(N, m, rng):
    """Return (Hknown, Vknown): True where edge state is OBSERVED. Each edge
    hidden independently with probability m."""
    Hknown = rng.random((N, N - 1)) >= m
    Vknown = rng.random((N - 1, N)) >= m
    return Hknown, Vknown


# ----------------------------------------------------------------------------
# rasterization
# ----------------------------------------------------------------------------
def rasterize(H, V, Hknown, Vknown, start, goal):
    """Build (C, L, L) float32 observation image."""
    N = H.shape[0]
    L = 2 * N - 1
    img = np.zeros((N_CHANNELS, L, L), dtype=np.float32)
    # channel 0: nodes
    for r in range(N):
        for c in range(N):
            img[0, 2 * r, 2 * c] = 1.0
    # horizontal edges (r,c)-(r,c+1) at pixel (2r, 2c+1)
    for r in range(N):
        for c in range(N - 1):
            pr, pc = 2 * r, 2 * c + 1
            if Hknown[r, c]:
                if H[r, c]:
                    img[1, pr, pc] = 1.0
                else:
                    img[2, pr, pc] = 1.0
            else:
                img[3, pr, pc] = 1.0
    # vertical edges (r,c)-(r+1,c) at pixel (2r+1, 2c)
    for r in range(N - 1):
        for c in range(N):
            pr, pc = 2 * r + 1, 2 * c
            if Vknown[r, c]:
                if V[r, c]:
                    img[1, pr, pc] = 1.0
                else:
                    img[2, pr, pc] = 1.0
            else:
                img[3, pr, pc] = 1.0
    img[4, 2 * start[0], 2 * start[1]] = 1.0
    img[5, 2 * goal[0], 2 * goal[1]] = 1.0
    return img


# ----------------------------------------------------------------------------
# Bayes oracle helpers
# ----------------------------------------------------------------------------
def completion_lower_upper(H, V, Hknown, Vknown, start, goal):
    """Return (certain_pos, certain_neg).
    lower = unknown -> closed; if reachable then certainly reachable.
    upper = unknown -> open;   if NOT reachable then certainly unreachable."""
    H_lower = H & Hknown            # unknown => closed
    V_lower = V & Vknown
    if reachable(H_lower, V_lower, start, goal):
        return True, False          # certainly reachable
    H_upper = H | (~Hknown)         # unknown => open
    V_upper = V | (~Vknown)
    if not reachable(H_upper, V_upper, start, goal):
        return False, True          # certainly unreachable
    return False, False             # ambiguous


def mc_posterior(H, V, Hknown, Vknown, start, goal, p, K, rng):
    """Monte-Carlo P(y=1 | X): resample unknown edges ~ Bernoulli(p), keep
    observed edges fixed, average reachability over K completions."""
    N = H.shape[0]
    Hunk = ~Hknown
    Vunk = ~Vknown
    nH = int(Hunk.sum())
    nV = int(Vunk.sum())
    hits = 0
    for _ in range(K):
        Hc = H.copy()
        Vc = V.copy()
        if nH:
            Hc[Hunk] = rng.random(nH) < p
        if nV:
            Vc[Vunk] = rng.random(nV) < p
        if reachable(Hc, Vc, start, goal):
            hits += 1
    return hits / K
