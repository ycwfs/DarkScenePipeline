"""Debug video: skeleton overlay + per-stream probabilities + fusion decision, served-faithful.

Replays a video through the DEPLOYED operator stack (platform/dark-demo-v2): DownscaleStage(840)
-> Retinexformer -> the real BehaviorRecognizer (real window/trim/fire machinery via subclass,
real skeleton_aux fusion, real reject_probs). The only addition is capturing the pre-reject
per-stream probabilities at each firing window; nothing in the decision path is reimplemented.

Rendering per output frame:
  - pose of the CURRENT frame drawn on the enhanced pixels (drawing-only; p_skel still comes
    from the 32-frame window at fire time, exactly as serving does)
  - a header panel holding the last firing window's RGB / SKEL / FUSED top-1 + confidence
    (+ presence, gate branch, open-set decision), refreshed every `stride` pushes -- the same
    hold-between-fires behaviour as the served label bar; the firing frame gets an orange board.

Usage (container): python -u reco_video_vis.py in1.mp4 [in2.mp4 ...] [--out DIR]
Outputs <out>/<stem>.mp4 + <out>/<stem>.csv (per-frame: fire flag, per-stream top-1/conf,
presence, full 10-way vectors).
"""
import argparse
import csv
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

PKG = "/ssd1/wfs/project/lowlight/DarkScenePipeline/platform/dark-demo-v2"
sys.path.insert(0, PKG)

from darkpipe.constants import BEHAVIOR_ZH
from darkpipe.stages.downscale import DownscaleStage
from darkpipe.stages.enhance_retinexformer import RetinexformerStage
from darkpipe.stages.recognize import (BehaviorRecognizer, RecognitionEvent,
                                       VideoMambaRecognizer, reject_probs)
from darkpipe.stages.skeleton_aux import COCO_EDGES, NO_PERSON_ZH, SkeletonAux, TAU

FONT_PATH = os.path.join(PKG, "darkpipe", "assets", "NotoSansCJK-subset.ttf")
PANEL_H = 112
PANEL_W = 840


def score_color(s):
    if s >= 0.5:
        return (0, 200, 0)
    if s >= 0.3:
        return (0, 165, 255)
    return (0, 0, 255)


def draw_pose(img, kpts, scores):
    """COCO-17 overlay color-coded by keypoint score (green/orange/red)."""
    for i, j in COCO_EDGES:
        s = min(float(scores[i]), float(scores[j]))
        if s < 0.10:
            continue
        cv2.line(img, tuple(np.round(kpts[i]).astype(int)),
                 tuple(np.round(kpts[j]).astype(int)), score_color(s), 2, cv2.LINE_AA)
    for (x, y), s in zip(kpts, scores):
        if float(s) < 0.10:
            continue
        cv2.circle(img, (int(round(x)), int(round(y))), 3, score_color(float(s)), -1,
                   cv2.LINE_AA)


class DebugBehaviorRecognizer(BehaviorRecognizer):
    """push() is the parent's body verbatim plus a stash of the pre-reject per-stream
    probabilities -- anything less faithful would visualize a pipeline that is not deployed."""
    debug = None

    def load(self, device: str) -> None:
        # α≥0.999 时部署的 BehaviorRecognizer.load 会整体跳过骨架分支（纯 RGB，rawbuf 都不建）。
        # 可视化要求 α=1 的决策 + 骨架路照常展示，所以绕过该早退、强制装载骨架分支：
        # fuse() 在 α=1 时数值上恒等于 p_rgb（1·p_rgb + 0·p_skel），决策语义与纯 RGB 逐位一致。
        VideoMambaRecognizer.load(self, device)
        aux = SkeletonAux(os.path.dirname(self.ckpt), device, self.skel_alpha,
                          no_person=self.skel_no_person)
        reason = aux.load()
        if reason:
            print(f"[recognize] [warn] {reason}")
            return
        self.skel = aux
        print(f"[vis] 骨架分支已强制装载仅用于展示（α={self.skel_alpha:g}，决策=纯RGB，"
              f"无人窗策略={NO_PERSON_ZH.get(self.skel_no_person, self.skel_no_person)}，"
              f"姿态 {aux.pose_ms:.1f} ms/帧)")

    @torch.inference_mode()
    def push(self, frame_bgr, frame_index: int, timestamp: float):
        if self.skel is not None:
            self.rawbuf.append(frame_bgr)
        self.buf.append((self._preprocess(frame_bgr), timestamp))
        self._trim()
        if not self._ready():
            self.debug = None
            return None
        fire = self._filled_pushes % self.stride == 0
        self._filled_pushes += 1
        if not fire:
            return None            # debug 保持上一次触发的值：面板语义=服务的标签条保持
        arr = np.stack(self._window_frames())
        prob = self._infer(arr)
        ps, pres = None, float("nan")
        if self.skel is not None and len(self.rawbuf) == len(self.buf):
            ps, pres = self.skel.probs(self._raw_window())
            fused = self.skel.fuse(prob, ps, pres)
        else:
            fused = prob
        dec = reject_probs(fused, self.labels, self.reject_tau)
        self.debug = dict(rgb=prob.copy(), skel=None if ps is None else ps.copy(),
                          pres=float(pres), fused=fused.copy(), dec=dec.copy(),
                          alpha=float(self.skel_alpha))
        order = dec.argsort()[::-1][:min(3, len(self.labels))]
        return RecognitionEvent(
            frame_index=frame_index, timestamp=timestamp,
            label=self.display(self.labels[int(order[0])]),
            confidence=float(dec[order[0]]),
            topk=[(self.display(self.labels[int(i)]), float(dec[i])) for i in order],
            model=self.kind, window=self.window)


def zh(label):
    return BEHAVIOR_ZH.get(label, label)


def render_panel(dbg, fired, fire_no, holding):
    """PIL header: three stream columns (top-1 + conf + bar + one info line). Chinese only
    for behaviour names (the subset font covers exactly those); fixed words stay ASCII."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (PANEL_W, PANEL_H), (22, 22, 26))
    d = ImageDraw.Draw(img)
    f_big = ImageFont.truetype(FONT_PATH, 26)
    f_sm = ImageFont.truetype(FONT_PATH, 15)

    labels = ["wave", "throw", "chase", "fall", "fight", "talk", "drink", "pick_up",
              "shake_hands", "other"]
    if dbg is None:
        d.text((12, 8), "waiting for the first 1.0 s window ...", font=f_sm,
               fill=(200, 200, 200))
        return np.array(img)

    cols = [("RGB", dbg["rgb"], (110, 170, 255), (40, 70, 130)),
            ("SKEL", dbg["skel"], (140, 220, 140), (45, 95, 45)),
            ("FUSED", dbg["fused"], (215, 170, 235), (95, 60, 120))]
    for c, (head, vec, col, barcol) in enumerate(cols):
        x = 12 + c * 278
        d.text((x, 6), head, font=f_sm, fill=col)
        if vec is None:                            # skeleton abstained (fail-soft / branch off)
            d.text((x, 26), "abstain (no skel)", font=f_big, fill=(150, 150, 150))
            continue
        conf = float(np.max(vec))
        lab = labels[int(np.argmax(vec))]
        d.text((x, 24), f"{zh(lab)} {conf:.2f}", font=f_big, fill=(240, 240, 240))
        d.rectangle([x, 62, x + 250, 78], fill=(50, 50, 55))
        d.rectangle([x, 62, x + int(250 * conf), 78], fill=barcol)
        if head == "RGB":
            top3 = sorted(zip(labels, dbg["rgb"]), key=lambda t: -t[1])[:3]
            d.text((x, 84), "top3: " + " ".join(f"{zh(l)}:{v:.2f}" for l, v in top3),
                   font=f_sm, fill=(160, 170, 190))
        elif head == "SKEL":
            gate = "person" if dbg["pres"] >= TAU else "NO person"
            gcol = (140, 220, 140) if dbg["pres"] >= TAU else (255, 120, 90)
            if dbg.get("alpha", 0.7) >= 0.999:      # α=1: 骨架两路权重均为 0，仅展示
                note = "  (alpha=1, display only)"
            else:
                note = "" if dbg["pres"] >= TAU else "  (NOT USED, other-vote)"
            d.text((x, 84), f"pres {dbg['pres']:.2f}  gate: {gate}{note}",
                   font=f_sm, fill=gcol)

    # decision, in the FUSED column's info line
    dec_lab = labels[int(np.argmax(dbg["dec"]))]
    dec_conf = float(np.max(dbg["dec"]))
    fus_lab = labels[int(np.argmax(dbg["fused"]))]
    if dec_lab != "other":
        d.text((568, 84), f"DECISION: REPORT {zh(dec_lab)} ({dec_conf:.2f})",
               font=f_sm, fill=(120, 230, 120))
    elif fus_lab != "other":
        d.text((568, 84), f"DECISION: {zh(fus_lab)} {float(np.max(dbg['fused'])):.2f} "
                          f"-> other (open-set)", font=f_sm, fill=(255, 130, 90))
    else:
        d.text((568, 84), f"DECISION: other ({dec_conf:.2f})", font=f_sm,
               fill=(190, 190, 190))
    tag = f"FIRE #{fire_no}" if fired else ("hold" if holding else "")
    d.text((PANEL_W - 110, 6), tag, font=f_sm,
           fill=(255, 170, 60) if fired else (150, 150, 150))
    return np.array(img)


def process_video(path, rec, st, ds, out_dir, fire_counter):
    stem = os.path.basename(path).rsplit(".", 1)[0]
    out_mp4 = os.path.join(out_dir, stem + ".mp4")
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = None
    ff = None
    csv_f = open(os.path.join(out_dir, stem + ".csv"), "w", newline="", encoding="utf-8-sig")
    wr = csv.writer(csv_f)
    wr.writerow(["frame", "t_s", "fired", "gate_person", "presence",
                 "rgb_top", "rgb_p", "skel_top", "skel_p", "fused_top", "fused_p",
                 "dec_top", "dec_p",
                 "rgb_vec", "skel_vec", "fused_vec"])
    labels = rec.labels
    i = 0
    while True:
        ok, raw = cap.read()
        if not ok:
            break
        enh = st(ds([raw]))[0]
        ev = rec.push(enh, i, i / fps)
        fired = ev is not None
        if fired:
            fire_counter[0] += 1
        vis = enh.copy()
        if rec.skel is not None:                    # drawing-only: pose of THIS frame
            kpts, scores = rec.skel.pose(enh)
            for m in range(len(kpts)):
                draw_pose(vis, kpts[m], scores[m])
        panel = render_panel(rec.debug, fired, fire_counter[0], rec.debug is not None)
        if W is None:
            W, H = vis.shape[1], vis.shape[0] + PANEL_H
            ff = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{W}x{H}", "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264",
                 "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", out_mp4],
                stdin=subprocess.PIPE)
        frame = np.vstack([panel, vis])
        if fired:
            cv2.rectangle(frame, (0, 0), (W - 1, H - 1), (0, 140, 255), 3)
        ff.stdin.write(frame.tobytes())
        if rec.debug is not None:
            dbg = rec.debug
            vec = lambda p: " ".join(f"{x:.3f}" for x in p)  # noqa: E731
            wr.writerow([i, f"{i / fps:.2f}", fired, int(dbg["pres"] >= TAU),
                         f"{dbg['pres']:.3f}",
                         labels[int(np.argmax(dbg["rgb"]))], f"{float(np.max(dbg['rgb'])):.3f}",
                         "-" if dbg["skel"] is None else labels[int(np.argmax(dbg["skel"]))],
                         "-" if dbg["skel"] is None else f"{float(np.max(dbg['skel'])):.3f}",
                         labels[int(np.argmax(dbg["fused"]))],
                         f"{float(np.max(dbg['fused'])):.3f}",
                         labels[int(np.argmax(dbg["dec"]))],
                         f"{float(np.max(dbg['dec'])):.3f}",
                         vec(dbg["rgb"]),
                         "-" if dbg["skel"] is None else vec(dbg["skel"]),
                         vec(dbg["fused"])])
        i += 1
        if i % 100 == 0:
            print(f"  {stem}: {i} frames", flush=True)
    cap.release()
    if ff is not None:
        ff.stdin.close()
        ff.wait()
    csv_f.close()
    print(f"[done] {out_mp4} ({i} frames @ {fps:g} fps)", flush=True)
    rec.reset()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default="/ssd1/wfs/project/lowlight/DarkScenePipeline/reco_vis_video")
    ap.add_argument("--span", type=float, default=1.0)
    ap.add_argument("--min-conf", type=float, default=0.65, help="deployed suanzi.json default")
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--no-person", default="soft", choices=("soft", "other", "rgb"))
    ap.add_argument("--stride", type=int, default=0, help="0 = deployed default (window//2=16)")
    ap.add_argument("--limit-frames", type=int, default=0, help="debug: cap frames per video")
    ap.add_argument("--ckpt-dir", default=os.path.join(PKG, "ckpts"),
                    help="weights dir (NTIRE.pth / rtmo onnx / stgcn / videomamba head); "
                         "point it at a staging dir to A/B a candidate head without "
                         "touching the delivered package")
    a = ap.parse_args()

    ds = DownscaleStage(840)
    st = RetinexformerStage(a.ckpt_dir)
    st.load("cuda")
    rec = None
    for attempt in range(1, 6):
        rec = DebugBehaviorRecognizer(a.ckpt_dir, span_sec=a.span,
                                      reject_tau=a.min_conf, skel_fusion_alpha=a.alpha,
                                      skel_no_person=a.no_person)
        if a.stride:
            rec.stride = a.stride
        rec.load("cuda")
        if rec.skel is not None:
            break
        # 部署卡独占不会发生；开发共享卡上邻居进程会瞬时抬高姿态探针（实测 15->58ms）。
        # 不降 25ms 预算，只重试。
        print(f"[vis] 骨架分支未过延迟探针（第 {attempt}/5 次，共享卡争用？），5s 后重试",
              flush=True)
        rec.close()
        time.sleep(5)
    if rec is None or rec.skel is None:
        sys.exit("error: 骨架分支连续 5 次未通过延迟探针，换一张空卡再跑")
    rec.warmup()

    os.makedirs(a.out, exist_ok=True)
    fire_counter = [0]
    for v in a.videos:
        process_video(v, rec, st, ds, a.out, fire_counter)
    rec.close()


if __name__ == "__main__":
    main()
