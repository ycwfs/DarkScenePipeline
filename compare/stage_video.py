"""Four-stage visualization: one video per pipeline step, plus a 2x2 grid of all four.

    1. input                  the raw dark frame, as decoded
    2. + low-light enhancement
    3. + super-resolution     physically 2x panel 1
    4. + action recognition   panel 3 with the label bar the pipeline draws BELOW the frame

The panels are real stage outputs from the shipped classes, run in the shipped order and with
the shipped contract -- enhance -> recognizer tap (post-enhance, PRE-SR) -> SR -> label bar,
mirroring `darkpipe/pipeline.py:process_chunk`, including the detail that the bar shows the
newest event for a whole chunk. So the grid shows what `darkpipe` writes, not a lookalike.

Panel sizes are honest: after x2 SR a frame really is twice as large, so every grid cell is
sized for the largest panel and the smaller ones are centred at native pixel size inside it,
with the true size printed in the caption. Nothing is resized to make the grid tidy (the one
exception is `--scale`, which shrinks the FINAL canvas uniformly for viewing and is stated in
the footer when used).

Captions also carry the measured ms/frame for that stage on this clip, which is the point of
the exercise: stages 1-2-4 are ~free, and stage 3 is where the choice of `--sr` backend
decides whether the pipeline meets its 15 fps / 1 s spec.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DP = os.path.dirname(HERE)
sys.path.insert(0, DP)

FONT = cv2.FONT_HERSHEY_SIMPLEX
PAD, CAP = 12, 34
BG = 24


def _caption(w, text, sub="", scale=0.6):
    """One cell's title strip: bold-ish white title on the left, grey detail on the right."""
    strip = np.full((CAP, w, 3), BG, np.uint8)
    cv2.putText(strip, text, (PAD, CAP - 11), FONT, scale, (235, 235, 235), 1, cv2.LINE_AA)
    if sub:
        (sw, _), _ = cv2.getTextSize(sub, FONT, scale * 0.85, 1)
        cv2.putText(strip, sub, (w - sw - PAD, CAP - 11), FONT, scale * 0.85,
                    (150, 150, 150), 1, cv2.LINE_AA)
    return strip


def _cell(frame, cell_w, cell_h, title, sub):
    """Centre `frame` at NATIVE size on a cell-sized dark background, caption on top.

    Cell height is per ROW (see main), so the 240p row is not padded out to the height of the
    480p row; the width is shared, which is what keeps the 2x size step visible.
    """
    canvas = np.full((cell_h, cell_w, 3), BG, np.uint8)
    h, w = frame.shape[:2]
    y, x = (cell_h - h) // 2, (cell_w - w) // 2
    canvas[y:y + h, x:x + w] = frame
    return np.vstack([_caption(cell_w, title, sub), canvas])


def _even(n):
    return n if n % 2 == 0 else n + 1


def build_stages(enhance, sr, recognize, ckpt_dir, device, reco_ckpt=""):
    """The real factory, driven by a real PipelineConfig -- no hand-built stage lists."""
    from darkpipe.config import PipelineConfig, validate
    from darkpipe.stages import build_stages as factory
    cfg = validate(PipelineConfig(input="x", output="x.mp4", enhance=enhance, sr=sr,
                                  recognize=recognize, ckpt_dir=ckpt_dir, device=device,
                                  reco_ckpt=reco_ckpt))
    frame_stages, recognizer = factory(cfg)
    for s in frame_stages:
        s.load(device)
    if recognizer:
        recognizer.load(device)
    return frame_stages, recognizer


def run_clip(raw, frame_stages, recognizer, fps, chunk=16):
    """Run one clip through the shipped stage order, keeping every intermediate.

    enhance -> recognizer tap (post-enhance, PRE-SR) -> SR -> label bar, mirroring
    `darkpipe/pipeline.py:process_chunk` including the detail that the bar shows the newest
    event for a whole chunk. Returns (enhanced, upscaled, annotated, events, per_chunk) where
    per_chunk rows are (frames, t_enh, t_sr, t_reco, warm) -- see `steady_state` for `warm`.

    Callers rendering several clips through ONE loaded recognizer must `reset()` it between
    them; this function deliberately does not, so a caller can also stream a long video in
    pieces.
    """
    from darkpipe.render import append_label_bar
    enh_stages = [s for s in frame_stages if s.name.startswith("enhance")]
    sr_stages = [s for s in frame_stages if s.name.startswith("sr")]
    enhanced, upscaled, annotated, per_chunk, events = [], [], [], [], []
    current, frame_idx = None, 0
    for i in range(0, len(raw), chunk):
        c = raw[i:i + chunk]
        # Two cold starts to keep out of the average: the first chunk (cudnn autotune, lazy
        # CUDA init) and the recognizer's first forward, which only happens once its window
        # has filled -- several chunks in, and worth ~2 s of triton/mamba kernel compilation.
        warm = i > 0 and (recognizer is None or len(events) > 0)
        t = time.time()
        for s in enh_stages:
            c = s(c)
        t_enh = time.time() - t
        enhanced.extend(c)

        t = time.time()
        if recognizer:
            for f in c:
                ev = recognizer.push(f, frame_idx, frame_idx / fps)
                frame_idx += 1
                if ev:
                    current, _ = ev, events.append(ev)
        t_reco = time.time() - t

        t = time.time()
        for s in sr_stages:
            c = s(c)
        t_sr = time.time() - t
        upscaled.extend(c)
        annotated.extend(append_label_bar(f, current) for f in c)
        per_chunk.append((len(c), t_enh, t_sr, t_reco, warm))
    return enhanced, upscaled, annotated, events, per_chunk


def steady_state(per_chunk):
    """-> {'enh','sr','reco'} ms/frame over the warm chunks, and how many frames those were."""
    warm = [c for c in per_chunk if c[4]] or per_chunk[1:] or per_chunk
    tf = sum(c[0] for c in warm)
    return {k: sum(c[j] for c in warm) / tf * 1000
            for j, k in ((1, "enh"), (2, "sr"), (3, "reco"))}, tf


def read_frames(path, limit=None):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while limit is None or len(frames) < limit:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps


def write_video(path, seq, fps):
    """mp4 at the sequence's own size, padded to even dimensions (mp4v needs that)."""
    h, w = seq[0].shape[:2]
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (_even(w), _even(h)))
    for f in seq:
        wr.write(cv2.copyMakeBorder(f, 0, _even(h) - h, 0, _even(w) - w,
                                    cv2.BORDER_CONSTANT, value=(BG, BG, BG)))
    wr.release()
    return w, h


def grid_frame(four, titles, k):
    """One 2x2 frame: shared cell WIDTH (that is what makes the x2 step readable), per-ROW
    height, so the 240p row is not padded out to the height of the 480p row."""
    cell_w = max(s[0].shape[1] for _, s in four) + 2 * PAD
    row_h = [max(s[0].shape[0] for _, s in four[:2]) + 2 * PAD,
             max(s[0].shape[0] for _, s in four[2:]) + 2 * PAD]
    cells = [_cell(seq[k], cell_w, row_h[i // 2], t, s)
             for i, ((_, seq), (t, s)) in enumerate(zip(four, titles))]
    return np.vstack([np.hstack(cells[:2]), np.hstack(cells[2:])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(
        HERE, "arid", "test_clips", "Drink", "Drink_11_1.mp4"))
    ap.add_argument("--enhance", default="retinexformer")
    ap.add_argument("--sr", default="bicubic_x2")
    ap.add_argument("--recognize", default="behavior")
    ap.add_argument("--ckpt-dir", default=os.path.join(DP, "ckpts"))
    ap.add_argument("--reco-ckpt", default="", help="override the recognizer weights file")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="shrink the final grid canvas for viewing (panels stay relative)")
    ap.add_argument("--outdir", default=os.path.join(HERE, "results", "stage_video"))
    ap.add_argument("--prefix", default="")
    a = ap.parse_args()

    raw, fps = read_frames(a.input, a.max_frames)
    if not raw:
        sys.exit(f"no frames decoded from {a.input}")
    H, W = raw[0].shape[:2]

    frame_stages, recognizer = build_stages(a.enhance, a.sr, a.recognize, a.ckpt_dir, a.device,
                                            reco_ckpt=a.reco_ckpt)

    # ---- phase 1: run the pipeline, keeping every intermediate -------------------------
    enhanced, upscaled, annotated, events, per_chunk = run_clip(
        raw, frame_stages, recognizer, fps, a.chunk)
    for s in frame_stages:
        s.close()
    if recognizer:
        recognizer.close()

    ms, tf = steady_state(per_chunk)
    n = len(raw)
    SH, SW = upscaled[0].shape[:2]
    print(f"{n} frames {W}x{H} -> SR {SW}x{SH}; over {tf} steady-state frames: "
          f"enhance {ms['enh']:.1f} ms/frame, sr {ms['sr']:.1f} ms/frame, "
          f"recognize {ms['reco']:.1f} ms/frame ({len(events)} events, "
          f"{ms['enh'] + ms['sr'] + ms['reco']:.1f} ms/frame total = "
          f"{1000 / (ms['enh'] + ms['sr'] + ms['reco']):.1f} fps of stage work)")

    # ---- phase 2: write one video per stage + the 2x2 grid ------------------------------
    os.makedirs(a.outdir, exist_ok=True)
    p = a.prefix or f"{a.enhance}_{a.sr}_{a.recognize}_"
    four = [("1_input", raw), ("2_enhance", enhanced), ("3_sr", upscaled),
            ("4_recognition", annotated)]
    for tag, seq in four:
        path = os.path.join(a.outdir, f"{p}{tag}.mp4")
        w, h = write_video(path, seq, fps)
        print(f"  -> {path}  ({w}x{h}, {len(seq)} frames)")

    titles = [
        ("1. input (dark)", f"{W}x{H}"),
        (f"2. + enhancement: {a.enhance}", f"{W}x{H}   {ms['enh']:.1f} ms/frame"),
        (f"3. + super-resolution: {a.sr}", f"{SW}x{SH}   {ms['sr']:.1f} ms/frame"),
        (f"4. + recognition: {a.recognize}", f"{SW}x{SH}+bar   {ms['reco']:.1f} ms/frame"),
    ]
    g0 = grid_frame(four, titles, 0)
    grid_h, grid_w = _even(g0.shape[0]), _even(g0.shape[1])
    out = os.path.join(a.outdir, f"{p}grid.mp4")
    wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                         (int(grid_w * a.scale) // 2 * 2, int(grid_h * a.scale) // 2 * 2))
    for k in range(n):
        g = grid_frame(four, titles, k)
        g = cv2.copyMakeBorder(g, 0, grid_h - g.shape[0], 0, grid_w - g.shape[1],
                               cv2.BORDER_CONSTANT, value=(BG, BG, BG))
        if a.scale != 1.0:
            g = cv2.resize(g, (int(grid_w * a.scale) // 2 * 2, int(grid_h * a.scale) // 2 * 2),
                           interpolation=cv2.INTER_AREA)
        wr.write(g)
    wr.release()
    print(f"  -> {out}  ({grid_w}x{grid_h}"
          f"{f' scaled x{a.scale}' if a.scale != 1.0 else ''}, {n} frames)")


if __name__ == "__main__":
    main()
