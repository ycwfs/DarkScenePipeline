"""Multi-GPU offline processing: split the video by frame range, one GPU per segment.

The pipeline is embarrassingly parallel over time -- the only cross-frame state is the
recognizer's ring buffer -- so the cheapest way to use N GPUs is N independent single-GPU
runs over contiguous segments, concatenated afterwards. Each worker is the ordinary
`darkpipe --mode offline` entry point, so a sharded run and a single-GPU run execute exactly
the same code per frame; nothing here is a second implementation of the pipeline.

Two costs are real and are not hidden in the reported throughput, which is measured
end-to-end by the parent:

  seeking     workers reach their segment by decoding and discarding the frames before it,
              because `CAP_PROP_POS_FRAMES` seeks to a keyframe and cv2 will not tell you
              whether it landed where you asked. Exactness matters more than the wasted
              decode: an off-by-a-few seek would silently duplicate or drop frames at every
              segment boundary, which no amount of downstream checking would recover.
  concat      there is no ffmpeg on this box, so the parts are re-encoded into the final
              file with cv2. That is one extra decode+encode pass over the video (~3.5 ms
              per frame at 720p).

Recognition has a real boundary effect: every worker starts with an empty window, so each
segment emits its first event only after `window` frames instead of continuing across the
cut. A K-way split therefore loses up to K-1 events relative to a single-GPU run and the
events straddling a boundary describe a shorter span. Splitting a 300-frame clip 6 ways
would be mostly boundary, so `min_seg_frames` refuses to use more GPUs than the video is
long enough to justify.
"""
import json
import os
import subprocess
import sys
import time

import cv2


def _frame_count(path, max_frames=None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open input source: {path!r}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    if n <= 0:
        raise RuntimeError(f"{path}: frame count unavailable (a stream?) -- "
                           "sharded offline processing needs a seekable file")
    return (min(n, max_frames) if max_frames else n), fps


def _segments(n_frames, n_gpus, min_seg_frames):
    """Contiguous [start, length) ranges, at most one per GPU, none shorter than the floor."""
    k = max(1, min(n_gpus, n_frames // max(1, min_seg_frames)))
    base, extra = divmod(n_frames, k)
    segs, start = [], 0
    for i in range(k):
        ln = base + (1 if i < extra else 0)
        segs.append((start, ln))
        start += ln
    return segs


def _concat(parts, out, fps):
    from .media import VideoWriter
    wr = VideoWriter(out, fps=fps)
    for p in parts:
        cap = cv2.VideoCapture(p)
        while True:
            ok, f = cap.read()
            if not ok:
                break
            wr.write(f)
        cap.release()
    wr.close()
    return wr.count


def run_offline_sharded(cfg, gpus, min_seg_frames=120):
    n_frames, fps = _frame_count(cfg.input, cfg.max_frames)
    segs = _segments(n_frames, len(gpus), min_seg_frames)
    if len(segs) < len(gpus):
        print(f"[shard] {n_frames} frames is too short to split {len(gpus)} ways "
              f"(min {min_seg_frames}/segment) -- using {len(segs)} GPU(s)")
    if len(segs) == 1:
        from .pipeline import run_offline
        return run_offline(cfg)

    work = os.path.join(os.path.dirname(os.path.abspath(cfg.output)) or ".",
                        f".shard_{os.getpid()}")
    os.makedirs(work, exist_ok=True)
    base = [sys.executable, "-m", "darkpipe.cli", "--mode", "offline",
            "--input", cfg.input, "--enhance", cfg.enhance, "--sr", cfg.sr,
            "--recognize", cfg.recognize, "--ckpt-dir", cfg.ckpt_dir,
            "--enhance-chunk", str(cfg.enhance_chunk), "--sr-chunk", str(cfg.sr_chunk)]
    for flag, val in (("--sr-scale", cfg.sr_scale),
                      ("--reco-stride", cfg.reco_stride), ("--reco-span-sec", cfg.reco_span_sec),
                      ("--reco-ckpt", cfg.reco_ckpt), ("--labels", cfg.labels),
                      ("--xclip-model", cfg.xclip_model),
                      ("--xclip-reject-tau", cfg.xclip_reject_tau)):
        if val not in ("", None):
            base += [flag, str(val)]
    if cfg.sr_fp32:
        base.append("--sr-fp32")
    if cfg.no_label_bar:
        base.append("--no-label-bar")

    t0 = time.time()
    procs, parts, evs = [], [], []
    for i, (start, ln) in enumerate(segs):
        part = os.path.join(work, f"part{i:02d}.mp4")
        ev = os.path.join(work, f"part{i:02d}.json")
        parts.append(part)
        evs.append(ev)
        cmd = base + ["--output", part, "--device", f"cuda:{gpus[i]}",
                      "--start-frame", str(start), "--max-frames", str(ln)]
        if cfg.events_json:
            cmd += ["--events-json", ev]
        print(f"[shard] gpu {gpus[i]}: frames {start}..{start + ln - 1} -> {part}")
        procs.append(subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True))
    fail = []
    for i, p in enumerate(procs):
        out = p.communicate()[0]
        if p.returncode != 0:
            fail.append(f"segment {i} (gpu {gpus[i]}) exited {p.returncode}:\n"
                        f"{(out or '').strip()[-800:]}")
    if fail:
        raise RuntimeError("[shard] " + "\n".join(fail))

    n_written = _concat(parts, cfg.output, fps)
    dt = time.time() - t0
    print(f"[done] {n_frames} frames in {dt:.1f}s = {n_frames / max(dt, 1e-9):.1f} fps "
          f"-> {cfg.output} ({n_written} frames written, {len(segs)} GPUs)")

    events = []
    if cfg.events_json:
        for (start, _), ev in zip(segs, evs):
            if not os.path.exists(ev):
                continue
            for e in json.load(open(ev)):
                e["frame_index"] += start          # worker indices are segment-relative
                e["timestamp"] += start / fps
                events.append(e)
        events.sort(key=lambda e: e["frame_index"])
        with open(cfg.events_json, "w") as f:
            json.dump(events, f, indent=2)
        print(f"[done] {len(events)} recognition events -> {cfg.events_json}")
    for p in parts + evs:
        if os.path.exists(p):
            os.remove(p)
    os.rmdir(work)
    return events
