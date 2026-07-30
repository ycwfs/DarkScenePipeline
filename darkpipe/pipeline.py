"""Offline pipeline runner: decode -> enhance -> (recognizer tap) -> SR -> label bar -> encode.

The recognizer consumes POST-enhance, PRE-SR frames: both recognizer checkpoints were
trained on enhanced (not super-resolved) frames, recognition preprocessing downsamples
anyway, and SR was measured recognition-neutral — so SR stays out of the decision path.

Decode and encode run on their own threads. They are cv2/ffmpeg work that releases the GIL,
and they are not small: at 640x480 the whole enhance+recognize path measured 40 ms/frame of
which only ~6 ms was the enhancer, so a serial loop spent most of its time not using the GPU.
The queues are bounded (backpressure, no unbounded RAM growth) and single-producer /
single-consumer, so frame order is preserved without any sequence bookkeeping.
"""
import json
import queue
import threading
import time

from .media import VideoReader, VideoWriter
from .render import append_label_bar
from .stages import build_stages

_STOP = object()          # sentinel: end of stream


def _decode_thread(reader, q, err):
    try:
        for chunk in reader.chunks():
            q.put(chunk)
    except BaseException as e:            # noqa: BLE001 - re-raised on the main thread
        err.append(e)
    finally:
        q.put(_STOP)


def _encode_thread(writer, q, err):
    try:
        while True:
            item = q.get()
            if item is _STOP:
                return
            for f in item:
                writer.write(f)
    except BaseException as e:            # noqa: BLE001
        err.append(e)
        # drain so a failed writer cannot deadlock the producer on a bounded queue
        while q.get() is not _STOP:
            pass


def run_offline(cfg):
    frame_stages, recognizer = build_stages(cfg)
    for s in frame_stages:
        print(f"[load] {s.name}")
        s.load(cfg.device)
    if recognizer:
        print(f"[load] recognize:{recognizer.name} (window={recognizer.window}, "
              f"stride={recognizer.stride})")
        recognizer.load(cfg.device)

    reader = VideoReader(cfg.input, chunk=cfg.enhance_chunk, max_frames=cfg.max_frames,
                         start_frame=cfg.start_frame)
    writer = VideoWriter(cfg.output, fps=reader.fps)
    whole = [s for s in frame_stages if s.whole_video]
    streaming = [s for s in frame_stages if not s.whole_video]

    events, current = [], None
    t0 = time.time()
    n_in = 0
    frame_idx = 0

    # enhance stages (streaming, non-whole-video) come before SR in build order;
    # the recognizer taps right after the LAST enhance stage / before SR.
    enh_stages = [s for s in streaming if s.name.startswith("enhance")]
    sr_stages = [s for s in streaming if s.name.startswith("sr")]

    def process_chunk(chunk):
        """GPU work for one chunk -> the frames to encode."""
        nonlocal current, frame_idx
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
        if recognizer and not cfg.no_label_bar:
            # NOTE: the bar shows the newest event for the whole chunk, matching the previous
            # serial behaviour (`current` was likewise only advanced between pushes).
            return [append_label_bar(f, current) for f in chunk]
        return chunk

    if whole:  # RealRestorer: two-pass (restore entire video first), no overlap to be had
        frames = reader.read_all()
        n_in = len(frames)
        for s in whole:
            frames = s(frames)
        for i in range(0, len(frames), cfg.enhance_chunk):
            for f in process_chunk(frames[i:i + cfg.enhance_chunk]):
                writer.write(f)
    else:
        in_q, out_q, err = queue.Queue(maxsize=2), queue.Queue(maxsize=2), []
        dec = threading.Thread(target=_decode_thread, args=(reader, in_q, err), daemon=True)
        enc = threading.Thread(target=_encode_thread, args=(writer, out_q, err), daemon=True)
        dec.start()
        enc.start()
        drained = False
        try:
            while True:
                chunk = in_q.get()
                if chunk is _STOP:
                    drained = True
                    break
                n_in += len(chunk)
                out_q.put(process_chunk(chunk))
        finally:
            out_q.put(_STOP)
            enc.join()
            # If the loop exited early -- process_chunk raised -- the decode thread is still
            # blocked on `in_q.put()` against a bounded queue with no consumer left, so
            # `dec.join()` would never return and a crashed run would present as a hung one.
            # (Measured: a CUDA OOM at --sr-chunk 8 held the process for 17 minutes at 0% GPU
            # before it was killed.) Drain until the decoder's own `finally` sentinel arrives.
            while not drained:
                drained = in_q.get() is _STOP
            dec.join()
        if err:
            raise err[0]

    writer.close()
    for s in frame_stages:
        s.close()
    if recognizer:
        recognizer.close()

    dt = time.time() - t0
    cfg.stats = dict(frames=n_in, seconds=round(dt, 3), fps=round(n_in / max(dt, 1e-9), 2),
                     frames_written=writer.count, events=len(events), gpus=1)
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
