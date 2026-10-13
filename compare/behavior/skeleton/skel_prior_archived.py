"""Skeleton geometric-prior veto: hard posture requirements on top of the fused decision.

User directive (2026-10-07): RGB may say "wave" but if the pose never lifted the hand above
the head, the window must not report wave; drink requires a hand at the neck/face; chase
requires large motion amplitude. Implemented as a post-fusion veto so the fused
distribution (alpha-blend + no-person policy, see skeleton_aux.py) stays the single source
of probability and the prior can only ever *demote* labels, never invent one.

Design rules, all measured on the 765-window independent new-val skeleton cache
(compare/behavior/skeleton/skel_prior_study.py, 2026-10-07):
  - veto only on POSITIVE geometric evidence of violation; missing/occluded joints yield
    NaN features whose comparisons are all False -> the rule abstains (fail-soft: the pose
    branch must never take recall away, it only got fused in because it protects precision);
  - the whole prior abstains when presence < TAU (no person, no geometric evidence);
  - thresholds sit at the true class's 5th percentile or below, so >=95% of true windows
    pass by construction.

Measured pass/veto table (rows = ground truth, "pass" = rule would keep a window named so):
    rule    true-class pass      veto rate when a WRONG window is named so
    wave    wave 60/60 (100%)   chase 96%  other 89%  drink 78%  pick_up 71%
    drink   drink 93/94 (99%)   wave 90%   other 91%  chase 78%  pick_up 62%
    chase   chase 68/69 (99%)   other 22%  wave 52%   drink 55%  pick_up  4%
(chase's modest 22% on `other` is the known semantic overlap: the chase positives are
single-person fast-walk, so a walk window carries real motion amplitude. The veto still
removes the static/fidgeting walk windows, and never touches true chase.)

window_features() is byte-identical to skel_prior_study.py's -- the study and this stage
must stay a synced pair, like skeleton_aux.py and its NVIDIA-Box port.
"""
import numpy as np

# Kept locally so this module stays pure-numpy and importable outside the package
# (fusion_eval harness, NVIDIA-Box port). MUST stay in sync with skeleton_aux.BEHAVIORS
# (the label order is a frozen interface -- see constants.py) and skeleton_aux.TAU.
BEHAVIORS = ["wave", "throw", "chase", "fall", "fight", "talk", "drink", "pick_up",
             "shake_hands", "other"]
TAU = 0.5                                    # person gate, same as skeleton_aux

# COCO-17 keypoint indices (same layout skeleton_aux stores: x/W, y/H in [0,1], score).
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16

CONF_THR = 0.3          # same visibility bar as normalize_window
FAST_JOINTS = (L_ELB, R_ELB, L_WRI, R_WRI, L_KNE, R_KNE, L_ANK, R_ANK)

# Thresholds from skel_prior_study.py over cache/skel_realact_all_v1/val (765 windows).
# wave  pass needs (gap >= .10 or frac >= .10): wave p5 gap=.25 -> 100% pass
# drink pass needs a hand within 0.45 torso of the head at some frame (drink p95 dmin=.37
#        -> 99% pass) and forbids the wave signature gap >= .25 (wave p5 gap=.25: kills
#        90% of wave-windows mis-filed as drink; costs ~1% of true drink, p95 gap=.17)
# chase pass needs joint speed >= .05/frame or joint-angle change >= 4 deg/frame:
#        chase p5 amp=.06 / ang=3.95 -> 99% pass
WAVE_GAP_MIN, WAVE_FRAC_MIN = 0.10, 0.10
DRINK_DMAX, DRINK_FRAC_MIN, WAVE_GAP_IRON = 0.45, 0.20, 0.25
CHASE_AMP_MIN, CHASE_ANG_MIN = 0.05, 4.0


def _angle_deg(a, b, c):
    """Angle at b (degrees) for coordinate rows (..., 2)."""
    u = a - b
    v = c - b
    nu = np.linalg.norm(u, axis=-1)
    nv = np.linalg.norm(v, axis=-1)
    cos = np.clip((u * v).sum(-1) / np.maximum(nu * nv, 1e-6), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def window_features(sk, conf_thr=CONF_THR):
    """sk (T, M, 17, 3) frame-normalized -> dict of window features (NaN where unevaluable).

    Main person = highest presence (max-over-frames mean keypoint score, the same person
    normalize_window weights most). All lengths are in torso units (mid-shoulder..mid-hip,
    window median) so features are scale- and resolution-invariant. NaN comparisons are
    False by design: any clause with missing evidence never vetoes.
    """
    T, M, V, _ = sk.shape
    pres = np.array([sk[:, m, :, 2].mean() for m in range(M)] if T else np.zeros(M))
    if not len(pres) or pres.max() <= 0:
        return {}
    m = int(pres.argmax())
    xy, cf = sk[:, m, :, :2], sk[:, m, :, 2]
    vis = cf > conf_thr

    def mid(i, j):                                          # per-frame midpoint or NaN
        both = vis[:, i] & vis[:, j]
        out = np.full((T, 2), np.nan, np.float32)
        out[both] = (xy[both, i] + xy[both, j]) / 2
        return out

    sh, hp = mid(L_SHO, R_SHO), mid(L_HIP, R_HIP)
    torso = np.linalg.norm(sh - hp, axis=1)
    scale = float(np.nanmedian(torso)) if np.isfinite(torso).any() else np.nan
    if not np.isfinite(scale) or scale < 1e-3:
        return {}
    f = {"scale": scale}

    # head reference per frame: mid-eye -> mid-ear -> nose
    head = mid(L_EYE, R_EYE)
    fb = mid(L_EAR, R_EAR)
    fill = np.isnan(head[:, 0]) & ~np.isnan(fb[:, 0])
    head[fill] = fb[fill]
    fill = np.isnan(head[:, 0]) & ~np.isnan(xy[:, NOSE, 0])
    head[fill] = xy[:, NOSE][fill]

    # --- wave: 手腕高于头（y 向下为正，所以 gap>0 = 手在头之上）---
    wr = np.stack([xy[:, L_WRI], xy[:, R_WRI]], 1)           # (T,2,2)
    wr_vis = vis[:, L_WRI] | vis[:, R_WRI]
    ev = wr_vis & ~np.isnan(head[:, 0])
    raised = np.zeros(T, bool)
    if ev.any():
        d = (head[ev, None, :] - wr[ev])                    # (n,2,2): head - wrist
        above = (d[..., 1] > 0) & (vis[ev][:, [L_WRI, R_WRI]])
        raised[ev] = above.any(1)
        f["wave_gap"] = float(np.max(d[..., 1])) / scale    # + = highest wrist above head
    f["wave_frac"] = float(raised[ev].mean()) if ev.sum() else np.nan

    # --- drink: 手到头部距离 ---
    if ev.any():
        dist = np.linalg.norm(wr[ev] - head[ev, None, :], axis=-1) / scale
        f["drink_dmin"] = float(dist.min())
        f["drink_frac_r0.5"] = float((dist < 0.5).any(1).mean())
    f["drink_n"] = int(ev.sum())

    # --- chase: 运动幅度（每帧、torso 单位）与肘/膝角度变化（度/帧）---
    spd = []
    for j in FAST_JOINTS:
        both = vis[1:, j] & vis[:-1, j]
        if both.any():
            spd.append(np.linalg.norm(xy[1:, j] - xy[:-1, j], axis=-1)[both] / scale)
    f["amp_joint"] = float(np.concatenate(spd).mean()) if spd else np.nan

    def ang_seq(a, b, c):
        ok = vis[:, a] & vis[:, b] & vis[:, c]
        seq = np.full(T, np.nan, np.float32)
        seq[ok] = _angle_deg(xy[ok, a], xy[ok, b], xy[ok, c])
        return seq

    d_ang = []
    for a, b, c in ((L_SHO, L_ELB, L_WRI), (R_SHO, R_ELB, R_WRI),
                    (L_HIP, L_KNE, L_ANK), (R_HIP, R_KNE, R_ANK)):
        s = ang_seq(a, b, c)
        mv = ~np.isnan(s[1:]) & ~np.isnan(s[:-1])
        if mv.any():
            d_ang.append(np.abs(s[1:] - s[:-1])[mv])
    f["ang_amp"] = float(np.concatenate(d_ang).mean()) if d_ang else np.nan
    return f


def _r_wave(f):
    """挥手: 手必须举过头顶（高度差 >= WAVE_GAP_MIN 或过顶帧占比 >= WAVE_FRAC_MIN）。"""
    return not (f.get("wave_gap", np.nan) < WAVE_GAP_MIN
                and f.get("wave_frac", np.nan) < WAVE_FRAC_MIN)


def _r_drink(f):
    """喝水: 手须到过头/颈附近（dmin <= DRINK_DMAX 或近距离帧占比 >= DRINK_FRAC_MIN），
    且从未出现挥手签名（wave_gap >= WAVE_GAP_IRON = wave 类 p5 分位，喝水类 p95 只有
    0.17）——这是单靠"手在头附近"分不开喝水/挥手时唯一的反向铁证。"""
    return not ((f.get("drink_dmin", np.nan) > DRINK_DMAX
                 and f.get("drink_frac_r0.5", np.nan) < DRINK_FRAC_MIN)
                or f.get("wave_gap", np.nan) >= WAVE_GAP_IRON)


def _r_chase(f):
    """追逐: 动作幅度须够大（关节速度 >= CHASE_AMP_MIN torso/帧 或 关节角度变化
    >= CHASE_ANG_MIN 度/帧；静止/小动作的窗两条件都满足不了才否决）。"""
    return not (f.get("amp_joint", np.nan) < CHASE_AMP_MIN
                and f.get("ang_amp", np.nan) < CHASE_ANG_MIN)


#: label -> (rule, 中文要求描述, 违规时给日志看的特征摘要)
RULES = {
    "wave": (_r_wave, "手腕须举过头顶",
             lambda f: f"wave_gap={f.get('wave_gap', float('nan')):.2f}"),
    "drink": (_r_drink, "手须到过脸/颈部附近",
              lambda f: f"dmin={f.get('drink_dmin', float('nan')):.2f}"),
    "chase": (_r_chase, "动作幅度须够大",
              lambda f: f"amp={f.get('amp_joint', float('nan')):.2f} ang={f.get('ang_amp', float('nan')):.1f}deg"),
}


def apply_prior(prob, sk, pres):
    """Post-fusion veto. prob (10,) fused distribution (skeleton_aux.BEHAVIORS order),
    sk (T, 2, 17, 3) raw frame-normalized pose of the window, pres presence score.

    Walks named candidates in descending probability; a candidate that violates its
    geometric requirement is vetoed; the first candidate that passes (or has no rule)
    wins. Every candidate vetoed -> `other`. The whole check abstains when presence <
    TAU (no person = no geometric evidence) or the window has no usable features.

    Returns (new_prob (10,) renormalized, vetoed list of (label, feature-summary str))
    . Renormalizing over the survivors keeps the vector a proper distribution for
    reject_probs downstream.
    """
    try:      # package context (the operator): fail loudly if the constants ever drift
        from .skeleton_aux import BEHAVIORS as _BA, TAU as _T
        assert _BA == BEHAVIORS and _T == TAU, "skel_prior constants drifted"
    except ImportError:   # standalone load (eval harness / NVIDIA-Box port): locals only
        pass
    if pres is None or pres < TAU or prob.max() <= 0:
        return prob, []
    f = window_features(sk)
    if not f:
        return prob, []
    other_i = BEHAVIORS.index("other")
    order = np.argsort(prob)[::-1]
    vetoed, winner = [], None
    for i in order:
        lb = BEHAVIORS[int(i)]
        if lb == "other":
            break
        rule = RULES.get(lb)
        if rule is None or rule[0](f):
            winner = lb
            break
        vetoed.append((lb, rule[2](f)))
    if not vetoed:
        return prob, []
    out = prob.copy()
    for lb, _ in vetoed:
        out[BEHAVIORS.index(lb)] = 0.0
    if winner is None:                       # every named candidate failed its prior
        out = np.zeros_like(prob)
        out[other_i] = 1.0
    else:
        out /= max(out.sum(), 1e-6)
    return out, vetoed
