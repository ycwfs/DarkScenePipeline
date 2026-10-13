"""Plain-torch ST-GCN for COCO-17 skeleton sequences, adapted from yysijie/st-gcn (A-2.0).

Model input per person: (N, 3, T, V). The person dimension M is handled OUTSIDE the net:
each person runs through the same weights and their logits are merged by a presence weight
(skel_train.merge_persons), so absent-person slots stay inert. The graph is COCO-17 (the
reference repo only ships openpose/ntu layouts); adjacency follows the reference's
hop-distance gaussian kernel with per-kernel degree normalisation.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# COCO-17 joints: 0 nose, 1/2 L/R eye, 3/4 L/R ear, 5/6 L/R shoulder, 7/8 L/R elbow,
# 9/10 L/R wrist, 11/12 L/R hip, 13/14 L/R knee, 15/16 L/R ankle
COCO_EDGES = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
              (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
              (3, 5), (4, 6)]
FLIP_MAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]   # L<->R permutation


def coco_adjacency(V=17, max_hop=1, sigma=1.0):
    """(K=max_hop+1, V, V) kernel branches: exp(-hop²/2σ²) masked to hop<=h, deg-normalised."""
    A = np.eye(V, dtype=np.float32)
    for i, j in COCO_EDGES:
        A[i, j] = A[j, i] = 1
    hop = np.full((V, V), np.inf, np.float32)
    np.fill_diagonal(hop, 0)
    reach = np.eye(V, dtype=np.float32)
    for d in range(1, max_hop + 1):                     # boolean matmul: one more hop
        reach = (reach > 0).astype(np.float32) @ (A > 0).astype(np.float32)
        hop = np.where((reach > 0) & np.isinf(hop), float(d), hop)
    kernel = np.zeros((max_hop + 1, V, V), np.float32)
    for h in range(max_hop + 1):
        k = np.where(hop <= h, np.exp(-(hop ** 2) / (2 * sigma ** 2)), 0.0)
        kernel[h] = k / (k.sum(1, keepdims=True) + 1e-6)
    return kernel


class STGCNBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, kt=9, dropout=0.1):
        super().__init__()
        self.gcn = nn.Conv2d(cin, cout, 1)
        self.bn1 = nn.BatchNorm2d(cout)
        # layout (N,C,V,T): spatial = H(V), temporal = W(T) -> kernel (1,kt), stride on W
        self.tcn = nn.Sequential(
            nn.Conv2d(cout, cout, (1, kt), (1, stride), padding=(0, (kt - 1) // 2)),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.stride = stride
        self.res_conv = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.res_bn = nn.BatchNorm2d(cout) if cin != cout else None
        self.drop = nn.Dropout(dropout)

    def forward(self, x, A):
        y = self.gcn(x)                                          # (N,C',V,T)
        y = F.relu(self.bn1(torch.einsum("ncit,kio->ncot", y, A).contiguous()))
        y = self.tcn(y)
        r = self.res_conv(x)
        if r.shape != y.shape:
            r = F.adaptive_avg_pool2d(r, y.shape[2:])
        if self.res_bn is not None:
            r = self.res_bn(r)
        return self.drop(F.relu(y + r))


class STGCN(nn.Module):
    """(N, 3, T, V) -> (N, num_classes). Four blocks, ~0.4M params at 64 channels."""

    def __init__(self, in_ch=3, num_classes=10, V=17, channels=(64, 64, 64, 128)):
        super().__init__()
        self.register_buffer("A", torch.from_numpy(coco_adjacency(V)))
        self.bn_in = nn.BatchNorm1d(in_ch)
        mods, cin = [], in_ch
        for i, c in enumerate(channels):
            mods.append(STGCNBlock(cin, c, stride=2 if i >= 2 else 1))
            cin = c
        self.blocks = nn.ModuleList(mods)
        self.fc = nn.Linear(cin, num_classes)
        nn.init.normal_(self.fc.weight, 0, 0.01)

    def forward(self, x):
        """x: (N, 3, T, V) -> logits (N, num_classes)."""
        N, C, T, V = x.shape
        x = self.bn_in(x.reshape(N, C, T * V)).reshape(N, C, T, V)
        x = x.permute(0, 1, 3, 2).contiguous()                  # (N,C,V,T) for 2d convs
        for blk in self.blocks:
            x = blk(x, self.A)
        return self.fc(x.mean((2, 3)))


def merge_persons(logits, presence):
    """logits (N, M, C), presence (N, M) in [0,1] -> presence-weighted sum over M."""
    w = presence / presence.sum(1, keepdim=True).clamp_min(1e-6)
    return (logits * w.unsqueeze(-1)).sum(1)


def normalize_window(sk, conf_thr=0.3):
    """sk (T, M, 17, 3) xy-norm+conf -> (M, 3, T, 17) float32, hip-centred, torso-scaled.

    Per frame: center = mid hip (falls back to mid shoulder, else the frame stays zero);
    scale = torso length (falls back to the visible bounding-box diagonal), eps-guarded.
    Person presence = max over frames of that person's mean keypoint score.
    """
    T, M, V, _ = sk.shape
    xy, cf = sk[..., :2], sk[..., 2]
    out = np.zeros((M, 3, T, V), np.float32)
    pres = np.zeros(M, np.float32)
    for m in range(M):
        if T:
            pres[m] = float(cf[:, m].mean(1).max())
        for t in range(T):
            p = cf[t, m]
            vis = p > conf_thr
            if vis.sum() < 3:
                continue
            pts = xy[t, m]
            if vis[11] and vis[12]:
                hip = (pts[11] + pts[12]) / 2
            elif vis[5] and vis[6]:
                hip = (pts[5] + pts[6]) / 2
            else:
                continue
            if vis[5] and vis[6] and vis[11] and vis[12]:
                torso = float(np.linalg.norm((pts[5] + pts[6]) / 2 - (pts[11] + pts[12]) / 2))
            else:
                visp = pts[vis]
                torso = float(np.linalg.norm(visp.max(0) - visp.min(0)))
            s = max(torso, 1e-3)
            out[m, 0, t, vis] = (pts[vis, 0] - hip[0]) / s
            out[m, 1, t, vis] = (pts[vis, 1] - hip[1]) / s
            out[m, 2, t, vis] = p[vis]
    return out, pres


def augment(x, rng):
    """x (M, 3, T, 17): rotate/scale/hflip/jitter the xy channels; score channel untouched."""
    ang = rng.uniform(-0.35, 0.35)
    c, s = np.float32(np.cos(ang)), np.float32(np.sin(ang))
    x0, x1 = x[:, 0].copy(), x[:, 1].copy()
    x[:, 0] = (x0 * c - x1 * s) * rng.uniform(0.85, 1.15)
    x[:, 1] = (x0 * s + x1 * c) * rng.uniform(0.85, 1.15)
    if rng.random() < 0.5:
        x[:, 0] = -x[:, 0]
        x = x[..., FLIP_MAP]
    x[:, :2] += rng.normal(0, 0.02, x[:, :2].shape).astype(np.float32)
    return x
