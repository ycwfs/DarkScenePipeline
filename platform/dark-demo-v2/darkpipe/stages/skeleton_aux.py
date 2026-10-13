"""COCO-17 skeleton branch for the behavior recognizer: RTMO-s pose -> ST-GCN -> weighted
fusion with the RGB branch, plus a configurable no-person policy (NO_PERSON_MODES).

Why a second modality at all: the two branches fail on DISJOINT inputs. The RGB branch
occasionally reports a named behaviour at 0.85+ confidence on empty or irrelevant scenes
(the 65-clip person-walk suite measured 1-16 fired windows per model generation), while the
skeleton branch physically cannot fire without a person -- its keypoint scores double as a
presence signal. Conversely the skeleton branch is blind to anything that is not a body
posture (it cannot tell drinking from waving as well as RGB can at night). Weighted fusion
p = alpha*p_rgb + (1-alpha)*p_skel, evaluated per fusion weight across three field suites
(behaviour windows / person-walk false-positive clips / IR night positives), put the stack at
alpha=0.70: 1/365 false positives, 18/18 positives, old-val top1 0.9831 -- the same operating
point the RGB-only model needs alpha 1.0 for, with the skeleton branch holding the gate.

Ported verbatim (network, adjacency, normalization, person merge, gate threshold) from the
evaluation code in compare/behavior/skeleton/ so the served numbers ARE the measured numbers:
stgcn.py (STGCN/coco_adjacency/merge_persons/normalize_window), skeleton_cache.py (RTMO
wrapper), fusion_eval.py (TAU=0.5 gate, per-window protocol). The pose model runs on the
ENHANCED frame as this stage receives it -- the same frames, at proc_max_side resolution, the
field suites were measured on.

Failure policy is fail-soft by design: load() never raises. A site missing rtmlib, missing
onnxruntime (or having only the CPU build, which cannot meet the <=1 s latency budget), or
missing the weight files degrades to pure RGB -- byte-for-byte the v1 behaviour -- with one
loud Chinese startup line saying why. The one hard gate is the pose latency probe: onnxruntime
silently falls back to CPU when its CUDA providers cannot load, and 32 CPU pose inferences per
window would not just miss the accuracy target, they would blow the latency spec.
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import BEHAVIORS

# COCO-17 joints: 0 nose, 1/2 L/R eye, 3/4 L/R ear, 5/6 L/R shoulder, 7/8 L/R elbow,
# 9/10 L/R wrist, 11/12 L/R hip, 13/14 L/R knee, 15/16 L/R ankle
COCO_EDGES = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
              (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
              (3, 5), (4, 6)]

POSE_ONNX = "rtmo-s_end2end.onnx"                      # shipped in ckpts/ next to the torch weights
SKEL_CKPT = "stgcn_behavior_32f_realactall_full.pth"
POSE_SCORE_THR = 0.3
# Person gate: a window whose best person's mean keypoint score stays under TAU counts as
# no-person and takes the `no_person` policy (SkeletonAux.fuse). Picked from the field
# measurements: real person windows sit at 0.5+ while the RTMO "ghost person" false
# detections on empty scenes score below it (compare/behavior/skeleton probe_pose_ab.py).
TAU = 0.5
# What a no-person window reports. The hard "other" (v2.0 behaviour) is the strongest false
# positive brake -- but it is also a recall hazard: a real behaviour whose person the pose
# model MISSED (extreme dark, heavy occlusion, person at the frame edge) is forced to
# `other` even when the RGB branch is sure. The two softer policies keep a path for the RGB
# evidence:
#   "other"  force one-hot `other` (v2.0).
#   "soft"   the absent skeleton still votes `other`, but only with weight (1-alpha):
#            p = alpha*p_rgb + (1-alpha)*onehot(other). A weak RGB hallucination is damped
#            below the 0.65 fire threshold, while a strong one survives (firing needs
#            p_rgb >= 0.65/alpha, i.e. 0.93 at alpha=0.7) -- missed-person recall is kept
#            for exactly the windows where RGB is most confident.
#   "rgb"    the skeleton abstains; the window reports the pure RGB distribution (v1).
NO_PERSON_MODES = ("other", "soft", "rgb")
# Chinese for the startup log line only (identity stays the English mode string).
NO_PERSON_ZH = {"other": "强制other", "soft": "骨架投other票", "rgb": "纯RGB"}


def coco_adjacency(V=17, max_hop=1, sigma=1.0):
    """(K=max_hop+1, V, V) kernel branches: exp(-hop^2/2sigma^2) masked to hop<=h, deg-normalised."""
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
    """(N, 3, T, V) -> (N, num_classes). Four blocks, ~0.4M params at 64 channels.

    The person dimension M is handled OUTSIDE the net: each person runs through the same
    weights and their logits are merged by a presence weight (merge_persons), so absent-person
    slots stay inert.
    """

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


class SkeletonAux:
    """One skeleton branch, owned by a BehaviorRecognizer. `load()` is the only entry point
    that can fail and it reports by returning a reason string instead of raising."""

    #: measured RTMO-s on a RTX 3090: 8.7 ms/frame (YOLO26m-pose for comparison: 16.1). A CPU
    #: onnxruntime fallback runs 8-15x that and would turn one window into >1.5 s of pose work,
    #: so anything above POSE_MS_BUDGET at load time disables the branch rather than shipping
    #: a deployment that misses its latency spec silently.
    POSE_MS_BUDGET = 25.0

    def __init__(self, ckpt_dir, device, alpha, no_person="soft"):
        self.ckpt_dir = ckpt_dir
        self.device = "cuda:0" if str(device) == "cuda" else str(device)
        self.alpha = float(alpha)
        self.no_person = no_person if no_person in NO_PERSON_MODES else "soft"
        self.onnx = os.path.join(ckpt_dir, POSE_ONNX)
        self.skel_ckpt = os.path.join(ckpt_dir, SKEL_CKPT)
        self.pose_ms = 0.0
        self.pose = None
        self.net = None

    def load(self):
        """-> "" on success, else a reason string (already phrased for the startup log)."""
        missing = [p for p in (self.onnx, self.skel_ckpt) if not os.path.exists(p)]
        if missing:
            return (f"[recognize] 骨架分支关闭：权重缺失 {missing}。"
                    "算法路径不受影响，按纯 RGB 模型运行")
        try:
            from rtmlib import RTMO
            self.pose = RTMO(onnx_model=self.onnx, model_input_size=(640, 640),
                             backend="onnxruntime", device=self.device,
                             score_thr=POSE_SCORE_THR)
        except Exception as e:                           # noqa: BLE001 -- fail-soft by contract
            return (f"[recognize] 骨架分支关闭：姿态模型不可用（{type(e).__name__}: {e}）。"
                    "运行环境需要 rtmlib 与 onnxruntime-gpu（见包根 requirements-skeleton.txt），"
                    "当前按纯 RGB 模型运行")
        try:
            # Price the probe at steady state, not first call: the CUDA EP's first inference
            # pays session init (measured 884 ms), the second runs at the real per-frame cost
            # (measured 8.9 ms). Timing only the first call would misread GPU as CPU and
            # disable the branch on a perfectly healthy card.
            dummy = np.zeros((472, 840, 3), np.uint8)    # proc_max_side=840 enhanced frame
            self.pose(dummy)                             # session init, untimed
            t0 = time.perf_counter()
            self.pose(dummy)
            self.pose_ms = (time.perf_counter() - t0) * 1000
            if self.pose_ms > self.POSE_MS_BUDGET:
                self.pose = None
                return (f"[recognize] 骨架分支关闭：姿态推理实测 {self.pose_ms:.0f} ms/帧，"
                        f"超过 {self.POSE_MS_BUDGET:.0f} ms 预算（onnxruntime 没用上 GPU？"
                        "需要 onnxruntime-gpu 而非 CPU 版）。按纯 RGB 模型运行")
            self.net = STGCN(num_classes=10)
            sd = torch.load(self.skel_ckpt, map_location="cpu")
            self.net.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd
                                     else sd)
            self.net = self.net.to(self.device).eval()
        except Exception as e:                           # noqa: BLE001
            self.pose, self.net = None, None
            return (f"[recognize] 骨架分支关闭：ST-GCN 加载失败（{type(e).__name__}: {e}）。"
                    "按纯 RGB 模型运行")
        return ""

    def warmup(self):
        """Absorb the first ST-GCN CUDA forward (cuBLAS/cuDNN setup) at startup; the pose
        session was already priced by the load() probe."""
        self.probs([np.zeros((472, 840, 3), np.uint8)] * 32)

    def probs(self, frames_bgr):
        """frames (enhanced BGR uint8) -> (p_skel (10,) float32, presence float in [0,1])."""
        T = len(frames_bgr)
        sk = np.zeros((T, 2, 17, 3), np.float32)
        H, W = frames_bgr[0].shape[:2]
        for t in range(T):
            kpts, scores = self.pose(frames_bgr[t])
            if len(kpts) == 0:
                continue
            per = scores.mean(1)                          # person-level score
            for m, i in enumerate(np.argsort(per)[::-1][:2]):
                sk[t, m, :, :2] = kpts[i] / np.array([W, H], np.float32)
                sk[t, m, :, 2] = scores[i]
        x, pres = normalize_window(sk)
        xt = torch.from_numpy(x).unsqueeze(0).to(self.device)
        with torch.no_grad():
            lg = torch.stack([self.net(xt[:, m]) for m in range(xt.shape[1])], 1)[0]
        p = merge_persons(lg, torch.tensor(pres).unsqueeze(0).to(self.device))[0].softmax(0)
        return p.float().cpu().numpy(), float(pres.max())

    def fuse(self, pr, ps, pres):
        """No-person policy (see NO_PERSON_MODES) + weighted blend. `ps` is only meaningful
        when the pose model actually found someone -- below the gate the network would be
        reading an all-zero skeleton, so every no-person policy ignores it."""
        if pres >= TAU:
            return self.alpha * pr + (1.0 - self.alpha) * ps
        if self.no_person == "rgb":
            return pr
        out = np.zeros_like(pr)
        out[BEHAVIORS.index("other")] = 1.0
        if self.no_person == "other":
            return out
        return self.alpha * pr + (1.0 - self.alpha) * out      # "soft"
