"""max_prior decision-mode A/B vs the alpha-fusion baseline, on the three field suites
plus the platform round-trip clips. One pass computes BOTH modes:

  alpha mode   p = a*p_rgb + (1-a)*p_skel2 with the suite's hard no-person gate -- the same
               math as fusion_eval.py (so its historical numbers stay comparable). p_skel2 is
               the presence-weighted merge of the TOP-2 slots of the SAME 6-slot pose run
               analyze() does, which is exactly what the deployed probs() computes (its top-2
               fill IS slots 0/1 of the top-6 fill).
  max_prior    the deployed meeting three-step: SkeletonAux.analyze (up to 6 qualifying
               skeletons, best person) -> higher-confidence branch wins -> PriorGate on
               pick_up/wave/drink -> prior_reject -> reject_probs. Called through the
               deployed classes, not reimplemented.

Everything routes through the deployed package (darkpipe from platform/dark-demo-v2): the
RGB head is ckpts/videomamba_t_behavior_32f.pth, the skeleton branch and the prior gate are
the same objects serve constructs. Windows are cut from the RAW takes and enhanced with the
deployed chain (downscale 840 -> Retinexformer), suite A included -- its RGB cache cannot
provide keypoints or object detections.

Suites:
  A  val        177 windows / 33 takes, window-aligned (cache stems = <take>_<start>)
  B  fp65       64 no-action clips, sliding 1 s @ 0.5 s + whole   (real-actiondata-0909/other)
  C  positives  4 field positive takes, sliding windows
  P  platform   the 96 operator-saved false-positive clips (20261012 round-trip)

Usage: python -u prior_eval.py <gpu> [--suites ABCP] [--out runs.jsonl]
"""
import argparse
import csv
import glob
import json
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import torch

REPO = "/ssd1/wfs/project/lowlight/DarkScenePipeline"
PKG = os.path.join(REPO, "platform", "dark-demo-v2")
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from darkpipe.constants import IN_MEAN, IN_STD, RECO_CFG
from darkpipe.stages.downscale import DownscaleStage
from darkpipe.stages.enhance_retinexformer import RetinexformerStage
from darkpipe.stages.object_prior import PriorGate
from darkpipe.stages.recognize import (BehaviorRecognizer, frame_luma, preprocess_frame,
                                       prior_reject, reject_probs)
from darkpipe.stages.skeleton_aux import (MAX_PERSONS, TAU, SkeletonAux, merge_persons,
                                          normalize_window)
from darkpipe.vendor.videomamba import videomamba_tiny

LABELS = ["wave", "throw", "chase", "fall", "fight", "talk", "drink", "pick_up",
          "shake_hands", "other"]
OTHER = LABELS.index("other")
REJECT = 0.65
SRC = os.path.join(REPO, "real-actiondata-0909")
ODIR = os.path.join(SRC, "other")
PLAT = os.path.join(REPO, "online-negative-sample", "20261012_083305_78")
VAL_CACHE = "/ssd1/wfs/project/lowlight/compare/behavior/cache/realact_rf_v10_full"
TAKES = [("挥手22", "wave"), ("捡东西2", "pick_up"), ("追逐16", "chase"), ("喝水16", "drink")]
FOV_H, FOV_W = 224, 400


def read_video(path):
    cap = cv2.VideoCapture(path)
    frames, fps = [], cap.get(cv2.CAP_PROP_FPS) or 25
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps


def windows_of(n, fps):
    w = max(2, int(round(1.0 * fps)))
    wins = [("whole", 0, n)]
    i = 0
    while i + w <= n:
        wins.append((f"t{i / fps:.1f}", i, i + w))
        i += max(1, int(round(0.5 * fps)))
    return wins


def fired(p, conf=REJECT):
    return LABELS[int(p.argmax())] != "other" and float(p.max()) >= conf


class Stack:
    """The deployed recognizers, minus the ring buffer: rgb head + SkeletonAux + PriorGate
    are the serve objects; windows arrive pre-cut. Carries the attribute set the DEPLOYED
    BehaviorRecognizer._decide_max_prior expects (labels / lum_gate / act_tau / skel_tau /
    prior / t_parts), so run_window invokes the real method with this stack as `self` --
    the decision math cannot drift from what serve runs."""

    def __init__(self, dev, lum_gate=0.0, act_tau=0.65, skel_tau=0.65):
        self.dev = dev
        self.labels = LABELS
        self.lum_gate = float(lum_gate)
        self.act_tau = float(act_tau)
        self.skel_tau = float(skel_tau)
        ck = os.path.join(PKG, "ckpts")
        from darkpipe.constants import CKPT_FILES
        rgb_path = os.path.join(ck, CKPT_FILES["behavior"])
        meta = torch.load(rgb_path, map_location="cpu")
        h, w = (int(x) for x in meta["preprocess"][len("fullfov"):].split("x", 1))
        self.cfg = dict(fov=(h, w), mean=IN_MEAN, std=IN_STD)
        self.rgb = videomamba_tiny(num_classes=10, num_frames=32, img_size=(h, w))
        self.rgb.load_state_dict(meta["model"], strict=True)
        self.rgb = self.rgb.to(dev).eval()
        self.skel = SkeletonAux(ck, dev, alpha=0.0, no_person="soft")
        reason = self.skel.load()
        assert not reason, reason
        self.prior = PriorGate(ck, dev)
        reason = self.prior.load()
        assert not reason, reason
        self.t = defaultdict(float)
        self.t_parts = self.t        # the deployed method accumulates into t_parts["prior"]

    @torch.no_grad()
    def rgb_probs(self, win_bgr):
        t0 = time.perf_counter()
        idx = np.linspace(0, len(win_bgr) - 1, 32).round().astype(int)
        arr = np.stack([preprocess_frame(win_bgr[i], self.cfg) for i in idx])
        x = torch.from_numpy(arr).to(self.dev).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
        with torch.autocast("cuda", dtype=torch.float16):
            p = self.rgb(x).softmax(1)[0].float().cpu().numpy()
        self.t["rgb"] += time.perf_counter() - t0
        return p

    def skel_run(self, win_bgr):
        t0 = time.perf_counter()
        info = self.skel.analyze(win_bgr)
        self.t["skel"] += time.perf_counter() - t0
        # alpha-mode vector: the deployed probs() math on slots 0/1 of the same pose run
        t0 = time.perf_counter()
        x, pres = normalize_window(info["sk"])
        xt = torch.from_numpy(x[:2]).unsqueeze(0).to(self.dev)
        with torch.no_grad():
            lg = torch.stack([self.skel.net(xt[:, m]) for m in range(xt.shape[1])], 1)[0]
        ps2 = merge_persons(lg, torch.tensor(pres[:2]).unsqueeze(0).to(self.dev))
        self.t["skel2"] += time.perf_counter() - t0
        return info, ps2[0].softmax(0).cpu().numpy(), float(pres.max())

    def skel_pick(self, info):
        """Diagnostic-only re-derivation of the skeleton side's candidate (the deployed
        _decide_max_prior does the real selection; this just labels the audit columns)."""
        o = LABELS.index("other")
        if info["p"] is None:
            return None
        cand = [m for m in np.flatnonzero(info["valid"])
                if int(info["slot_p"][m].argmax()) != o
                and float(info["slot_p"][m].max()) > self.skel_tau]
        if not cand:
            return None
        m = cand[int(np.argmax(info["slot_p"][cand].max(1)))]
        return info["slot_p"][m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gpu")
    ap.add_argument("--suites", default="ABCP")
    ap.add_argument("--out", default="/tmp/prior_eval_runs.jsonl")
    ap.add_argument("--lum-gate", type=float, default=0.0,
                    help="deployed step-1 illuminance gate on RAW pre-enhance luma "
                         "(0 = off; per-window raw luma is always recorded, so gate "
                         "sweeps run offline on the JSONL)")
    ap.add_argument("--act-tau", type=float, default=0.65)
    ap.add_argument("--skel-tau", type=float, default=0.65)
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"
    alphas = [0.8, 0.7]

    ds = DownscaleStage(840)
    st = RetinexformerStage(os.path.join(PKG, "ckpts"))
    st.load(dev)
    S = Stack(dev, lum_gate=a.lum_gate, act_tau=a.act_tau, skel_tau=a.skel_tau)

    def enhance(frames):
        t0 = time.perf_counter()
        out = st(ds(frames))
        S.t["enhance"] += time.perf_counter() - t0
        return out

    rows = []

    def run_window(suite, name, expect, win, raw_win):
        p_rgb = S.rgb_probs(win)
        info, ps2, pres2 = S.skel_run(win)
        # alpha mode (suite convention: hard onehot on no-person, as fusion_eval)
        recs = {f"a{al:.2f}": (alpha_fuse(p_rgb, ps2, al, pres2)) for al in alphas}
        # max_prior mode: the DEPLOYED decision method, this stack as self
        sp = S.skel_pick(info)                       # audit columns only (see method doc)
        if sp is None or float(p_rgb.max()) >= float(sp.max()):
            branch, init_label = "rgb", LABELS[int(p_rgb.argmax())]
        else:
            branch, init_label = "skel", LABELS[int(sp.argmax())]
        # The gate's view of this window: the RAW (pre-enhance) frames, sampled exactly
        # like rgb_probs samples the enhanced ones. `lum` in the JSONL is therefore the
        # raw-scene luma the deployed gate compares against lum_gate -- NOT the enhanced
        # luma (whitened, can't see darkness).
        idx = np.linspace(0, len(raw_win) - 1, 32).round().astype(int)
        lums = [frame_luma(raw_win[i]) for i in idx]
        p_mp, note = BehaviorRecognizer._decide_max_prior(S, p_rgb, info, win, lums)
        p_mp = reject_probs(p_mp, LABELS, REJECT)
        row = dict(suite=suite, name=name, expect=expect,
                   rgb=[round(float(x), 4) for x in p_rgb],
                   ps2=[round(float(x), 4) for x in ps2], pres2=round(pres2, 3),
                   pres=round(info["pres"], 3), branch=branch, init=init_label,
                   lum=round(float(np.mean(lums)), 1),
                   nvalid=int(info["valid"].sum()),
                   # full precision: prior_reject's +1e-6 tie-break is rounded away at
                   # 4 decimals (0.8189 vs 0.8189 -> argmax flips back to the vetoed
                   # label), which miscounts vetoed windows as fired
                   note=note, mp=[float(x) for x in p_mp])
        for k, v in recs.items():
            row[k] = [round(float(x), 4) for x in v]
        rows.append(row)
        return row

    def alpha_fuse(pr, ps, al, pres):
        if pres >= TAU:
            return al * pr + (1 - al) * ps
        out = np.zeros(10, np.float32)
        out[OTHER] = 1.0
        return out

    # ---- A. val ---------------------------------------------------------------------
    if "A" in a.suites:
        t0 = time.perf_counter()
        vids = {}
        for p in sorted(glob.glob(os.path.join(VAL_CACHE, "val", "*", "*.npy"))):
            lab = os.path.basename(os.path.dirname(p))
            stem, s0 = os.path.basename(p)[:-4].rsplit("_", 1)
            if stem not in vids:
                src = SRC if os.path.exists(os.path.join(SRC, stem + ".mp4")) \
                    else os.path.join(SRC, "other")
                raw = read_video(os.path.join(src, stem + ".mp4"))[0]
                vids[stem] = (enhance(raw), raw)     # enhanced windows + raw luma source
            enh, rawf = vids[stem]
            i0 = int(s0)
            if i0 + 32 > len(enh):
                continue
            run_window("A", os.path.basename(p)[:-4], lab, enh[i0:i0 + 32], rawf[i0:i0 + 32])
        vids.clear()
        print(f"[A] done in {time.perf_counter() - t0:.0f}s")

    # ---- B. fp65 --------------------------------------------------------------------
    if "B" in a.suites:
        t0 = time.perf_counter()
        clips = sorted(f for f in os.listdir(ODIR)
                       if f.startswith("other") and f.endswith(".mp4"))
        for name in clips:
            raw, fps = read_video(os.path.join(ODIR, name))
            enh = enhance(raw)
            for wname, i0, i1 in windows_of(len(enh), fps):
                run_window("B", f"{name[:-4]}@{wname}", "other", enh[i0:i1], raw[i0:i1])
            del raw, enh
        print(f"[B] done in {time.perf_counter() - t0:.0f}s")

    # ---- C. positives ---------------------------------------------------------------
    if "C" in a.suites:
        t0 = time.perf_counter()
        for stem, expect in TAKES:
            raw, fps = read_video(os.path.join(SRC, stem + ".mp4"))
            enh = enhance(raw)
            for wname, i0, i1 in windows_of(len(enh), fps):
                run_window("C", f"{stem}@{wname}", expect, enh[i0:i1], raw[i0:i1])
            del raw, enh
        print(f"[C] done in {time.perf_counter() - t0:.0f}s")

    # ---- P. platform round-trip -----------------------------------------------------
    if "P" in a.suites:
        t0 = time.perf_counter()
        for path in sorted(glob.glob(os.path.join(PLAT, "*", "*.mp4"))):
            raw, fps = read_video(path)
            enh = enhance(raw)
            stem = os.path.basename(path)[:-4]
            for wname, i0, i1 in windows_of(len(enh), fps):
                run_window("P", f"{stem}@{wname}", "other", enh[i0:i1], raw[i0:i1])
            del raw, enh
        print(f"[P] done in {time.perf_counter() - t0:.0f}s")

    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[out] {len(rows)} windows -> {a.out}")
    print("[timing] " + " ".join(f"{k}={v:.0f}s" for k, v in sorted(S.t.items())))
    report(rows, alphas)


def report(rows, alphas):
    print(f"\n== A. val ({sum(1 for r in rows if r['suite'] == 'A')} windows) ==")
    for al in alphas:
        key = f"a{al:.2f}"
        sub = [r for r in rows if r["suite"] == "A"]
        cm = np.zeros((10, 10), np.int64)
        for r in sub:
            cm[LABELS.index(r["expect"]), int(np.argmax(r[key]))] += 1
        tp = np.diag(cm).astype(float)
        f1 = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1e-9)
        present = cm.sum(1) > 0
        print(f"  alpha={al:.2f}: top1={tp.sum() / cm.sum():.4f} "
              f"mF1={f1[present].mean():.4f}")
    sub = [r for r in rows if r["suite"] == "A"]
    cm = np.zeros((10, 10), np.int64)
    for r in sub:
        cm[LABELS.index(r["expect"]), int(np.argmax(r["mp"]))] += 1
    tp = np.diag(cm).astype(float)
    f1 = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1e-9)
    present = cm.sum(1) > 0
    print(f"  max_prior: top1={tp.sum() / cm.sum():.4f} mF1={f1[present].mean():.4f}")
    print("    " + "  ".join(f"{LABELS[i]}:f{f1[i]:.2f}(n{int(cm[i].sum())})"
                             for i in range(10) if cm[i].sum()))

    for suite, tag in (("B", "B. fp65"), ("P", "P. platform"), ("C", "C. positives")):
        sub = [r for r in rows if r["suite"] == suite]
        if not sub:
            continue
        print(f"\n== {tag} ({len(sub)} windows) ==")
        for al in alphas:
            key = f"a{al:.2f}"
            nf = sum(1 for r in sub if fired(np.array(r[key])))
            print(f"  alpha={al:.2f}: fired {nf}/{len(sub)}")
        nf = sum(1 for r in sub if fired(np.array(r["mp"])))
        print(f"  max_prior: fired {nf}/{len(sub)}")
        if suite == "C":
            for stem, expect in TAKES:
                take = [r for r in sub if r["name"].startswith(stem + "@")]
                hit_a = sum(1 for r in take if fired(np.array(r["a0.80"]))
                            and LABELS[int(np.argmax(r["a0.80"]))] == expect)
                hit_m = sum(1 for r in take if fired(np.array(r["mp"]))
                            and LABELS[int(np.argmax(r["mp"]))] == expect)
                print(f"    {stem} expect={expect}: alpha0.8 {hit_a}/{len(take)}"
                      f"  max_prior {hit_m}/{len(take)}")

    print("\n== max_prior 明细 ==")
    print("branch wins:", dict(defaultdict(int),
          **{k: sum(1 for r in rows if r["branch"] == k) for k in ("rgb", "skel")}))
    for r in rows:
        if r["note"]:
            head = r["note"].split(" ")[0].split(":")[0]
            tag = {"先验否决": "VETO", "先验通过": "pass", "照度": "lum",
                   "置信度": "act"}.get(head, head)
            top = LABELS[int(np.argmax(r["mp"]))]
            print(f"  [{tag}] {r['suite']} {r['name']} expect={r['expect']} "
                  f"init={r['init']} -> {top} | {r['note']}")


if __name__ == "__main__":
    main()
