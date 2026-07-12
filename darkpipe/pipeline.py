"""Offline pipeline runner: decode -> enhance -> (recognizer tap) -> SR -> label bar -> encode.

The recognizer consumes POST-enhance, PRE-SR frames: both recognizer checkpoints were
trained on enhanced (not super-resolved) frames, recognition preprocessing downsamples
anyway, and SR was measured recognition-neutral — so SR stays out of the decision path.
"""
import json
import time

from .media import VideoReader, VideoWriter
from .render import append_label_bar
from .stages import build_stages


def run_offline(cfg):
    frame_stages, recognizer = build_stages(cfg)
    for s in frame_stages:
        print(f"[load] {s.name}")
        s.load(cfg.device)
    if recognizer:
        print(f"[load] recognize:{recognizer.name} (window={recognizer.window}, "
              f"stride={recognizer.stride})")
        recognizer.load(cfg.device)

    reader = VideoReader(cfg.input, chunk=cfg.enhance_chunk, max_frames=cfg.max_frames)
    writer = VideoWriter(cfg.output, fps=reader.fps)
    whole = [s for s in frame_stages if s.whole_video]
    streaming = [s for s in frame_stages if not s.whole_video]

    events, current = [], None
    t0 = time.time()
    n_in = 0
    frame_idx = 0

    def process_chunk(chunk):
        nonlocal current, frame_idx
        # enhance stages (streaming, non-whole-video) come before SR in build order;
        # the recognizer taps right after the LAST enhance stage / before SR.
        enh_stages = [s for s in streaming if s.name.startswith("enhance")]
        sr_stages = [s for s in streaming if s.name.startswith("sr")]
        for s in enh_stages:
            chunk = s(chunk)
        if recognizer:
            for f in chunk:
                ev = recognizer.push(f, frame_idx, frame_idx / reader.fps)
                frame_idx += 1
                if ev:
                    current = ev
                    events.append(ev)
        else:
            frame_idx += len(chunk)
        for s in sr_stages:
            chunk = s(chunk)
        for f in chunk:
            writer.write(append_label_bar(f, current) if
                         (recognizer and not cfg.no_label_bar) else f)

    if whole:  # RealRestorer: two-pass (restore entire video first)
        frames = reader.read_all()
        n_in = len(frames)
        for s in whole:
            frames = s(frames)
        for i in range(0, len(frames), cfg.enhance_chunk):
            process_chunk(frames[i:i + cfg.enhance_chunk])
    else:
        for chunk in reader.chunks():
            n_in += len(chunk)
            process_chunk(chunk)

    writer.close()
    for s in frame_stages:
        s.close()
    if recognizer:
        recognizer.close()

    dt = time.time() - t0
    print(f"[done] {n_in} frames in {dt:.1f}s = {n_in / max(dt, 1e-9):.1f} fps "
          f"-> {cfg.output} ({writer.count} frames written)")
    if recognizer:
        labels = [e.label for e in events]
        print(f"[done] {len(events)} recognition events; last: "
              f"{events[-1].label if events else 'none'}; "
              f"majority: {max(set(labels), key=labels.count) if labels else 'none'}")
    if cfg.events_json and recognizer:
        with open(cfg.events_json, "w") as f:
            json.dump([e.to_dict() for e in events], f, indent=2)
        print(f"[done] events -> {cfg.events_json}")
    return events
