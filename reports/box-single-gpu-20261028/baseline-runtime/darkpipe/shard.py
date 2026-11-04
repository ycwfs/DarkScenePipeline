"""Multi-GPU offline processing: split the video by frame range, one GPU per segment.

The pipeline is embarrassingly parallel over time -- the only cross-frame state is the
recognizer's ring buffer -- so the cheapest way to use N GPUs is N independent single-GPU
runs over contiguous segments, concatenated afterwards. Each worker is the ordinary
`darkpipe.shard_worker` entry point, so a sharded run and a single-GPU run execute exactly
the same code per frame; nothing here is a second implementation of the pipeline.

Two costs are real and are not hidden in the reported throughput, which is measured
end-to-end by the parent:

  seeking     workers reach their segment by decoding and discarding the frames before it,
              because `CAP_PROP_POS_FRAMES` seeks to a keyframe and cv2 will not tell you
              whether it landed where you asked. Exactness matters more than the wasted
              decode: an off-by-a-few seek would silently duplicate or drop frames at every
              segment boundary, which no amount of downstream checking would recover.
  concat      the parts are decoded with cv2 and re-encoded into the final file, i.e. one
              extra decode+encode pass over the video (~3.5 ms per frame at 720p). An
              ffmpeg concat-demuxer stream copy would avoid it; it is not done here because
              the parts would first have to be proven to share identical encoder settings.

Recognition has a real boundary effect: every worker starts with an empty window, so each
segment emits its first event only after `window` frames instead of continuing across the
cut. A K-way split therefore loses up to K-1 events relative to a single-GPU run and the
events straddling a boundary describe a shorter span. Splitting a 300-frame clip 6 ways
would be mostly boundary, so `min_seg_frames` refuses to use more GPUs than the video is
long enough to justify.
"""
from dataclasses import asdict
import json
import os
import subprocess
import sys
import shutil
import tempfile
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


def _concat(parts, out, fps, expected_counts):
    from .media import VideoWriter
    wr = VideoWriter(out, fps=fps)
    try:
        for path, expected in zip(parts, expected_counts):
            cap = cv2.VideoCapture(path)
            count = 0
            try:
                if not cap.isOpened():
                    raise RuntimeError(f"cannot open shard: {path}")
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    wr.write(frame)
                    count += 1
            finally:
                cap.release()
            if count != expected:
                raise RuntimeError(f"shard {path}: expected {expected} frames, decoded {count}")
    finally:
        wr.close()
    return wr.count


def worker_config(cfg, gpu, start, length, output, events):
    """Serialize every PipelineConfig field; no hand-maintained CLI flag subset."""
    data = asdict(cfg)
    for key in ("input", "ckpt_dir", "reco_ckpt", "xclip_model", "rr_bundle"):
        if data.get(key):
            data[key] = os.path.abspath(data[key])
    data.update(mode="offline", device=("cpu" if cfg.device == "cpu" else f"cuda:{gpu}"),
                gpus="", start_frame=start, max_frames=length, output=output,
                events_json=events, warnings=[], stats={})
    return data


def run_offline_sharded(cfg, gpus, min_seg_frames=120):
    if cfg.clip_dir:
        raise ValueError("clip recording requires a single continuous timeline; disable sharding")
    if not gpus:
        raise ValueError("sharding needs at least one GPU id")
    total, fps = _frame_count(cfg.input)
    n_frames = max(0, total - int(cfg.start_frame or 0))
    if cfg.max_frames:
        n_frames = min(n_frames, cfg.max_frames)
    if n_frames <= 0:
        raise ValueError("no frames in requested shard range")
    segs = _segments(n_frames, len(gpus), min_seg_frames)
    if len(segs) == 1:
        from .pipeline import run_offline
        return run_offline(cfg)

    parent = os.path.dirname(os.path.abspath(cfg.output))
    os.makedirs(parent, exist_ok=True)
    work = tempfile.mkdtemp(prefix=".shard_", dir=parent)
    env = dict(os.environ)
    pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [pkg_parent, env.get("PYTHONPATH")]))
    # The worker cwd is private, so an unrelated root-level darkpipe cannot shadow v2.
    t0 = time.monotonic()
    procs, logs, parts, evs = [], [], [], []
    try:
        for i, (relative_start, length) in enumerate(segs):
            part = os.path.join(work, f"part{i:02d}.mp4")
            ev = os.path.join(work, f"part{i:02d}.json") if cfg.events_json else ""
            config_path = os.path.join(work, f"config{i:02d}.json")
            data = worker_config(cfg, gpus[i], int(cfg.start_frame or 0) + relative_start,
                                 length, part, ev)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            parts.append(part)
            evs.append(ev)
            log_path = os.path.join(work, f"worker{i:02d}.log")
            logs.append(log_path)
            print(f"[shard] {data['device']}: frames {data['start_frame']}.."
                  f"{data['start_frame'] + length - 1}")
            # A file avoids PIPE backpressure when another worker is being waited on.
            with open(log_path, "w", encoding="utf-8") as log:
                procs.append(subprocess.Popen(
                    [sys.executable, "-m", "darkpipe.shard_worker", config_path],
                    stdout=log, stderr=subprocess.STDOUT, env=env, cwd=work))
        for i, proc in enumerate(procs):
            if proc.wait() != 0:
                with open(logs[i], encoding="utf-8", errors="replace") as log:
                    detail = log.read()[-2000:]
                raise RuntimeError(f"[shard] segment {i} exited {proc.returncode}:\n{detail}")

        events = []
        if cfg.events_json:
            for (start, _), ev in zip(segs, evs):
                with open(ev, encoding="utf-8") as f:
                    segment_events = json.load(f)
                for event in segment_events:
                    event["frame_index"] += start
                    event["timestamp"] += start / fps
                    events.append(event)
            events.sort(key=lambda event: event["frame_index"])
        merged = os.path.join(work, "merged.mp4")
        n_written = _concat(parts, merged, fps, [length for _, length in segs])
        os.replace(merged, cfg.output)
        if cfg.events_json:
            with open(cfg.events_json, "w", encoding="utf-8") as f:
                json.dump(events, f, ensure_ascii=False, indent=2)
        dt = time.monotonic() - t0
        cfg.stats = dict(frames=n_frames, seconds=round(dt, 3),
                         fps=round(n_frames / max(dt, 1e-9), 2), frames_written=n_written,
                         gpus=len(segs), events=len(events))
        print(f"[done] {n_written} frames, {len(segs)} shards -> {cfg.output}")
        return events
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        shutil.rmtree(work, ignore_errors=True)
