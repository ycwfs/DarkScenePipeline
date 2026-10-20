"""Measure the real clip writer on a recorded output frame, without GPU inference."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op-dir", required=True)
    parser.add_argument("--frame", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", default="quality", choices=["off", "fast", "quality"])
    args = parser.parse_args()
    sys.path.insert(0, args.op_dir)
    from darkpipe.clips import ClipRecorder
    from darkpipe.stages.base import RecognitionEvent

    # Isolate per-frame writer cost; this is not a full service throughput benchmark.
    cv2.setNumThreads(1)
    frame = cv2.imread(args.frame)
    if frame is None:
        raise ValueError(f"cannot read {args.frame}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = ClipRecorder(str(out), pre_sec=2, post_sec=2, denoise=args.mode,
                       session="clips", stage_dir=str(out / "stage"))
    clean_ms = []
    original = rec._clean

    def clean(image):
        t0 = time.perf_counter()
        result = original(image)
        clean_ms.append((time.perf_counter() - t0) * 1000)
        return result

    rec._clean = clean
    start, cpu_start, queue_peak = time.perf_counter(), time.process_time(), 0
    try:
        for i in range(60):
            time.sleep(max(0, start + i / 15 - time.perf_counter()))
            ev = RecognitionEvent(i, i / 15, "Waving", 0.9) if i % 15 == 0 else None
            rec.push(frame.copy(), ev, time.time())
            queue_peak = max(queue_peak, rec._q.qsize())
    finally:
        rec.close()
    elapsed, cpu = time.perf_counter() - start, time.process_time() - cpu_start
    alive_at_close, stats_at_close = rec._thread.is_alive(), rec.stats()
    videos = []
    for video in Path(rec.root).rglob("*.mp4"):
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                                "-of", "json", str(video)], check=True, capture_output=True,
                               text=True, timeout=20)
        decode = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video),
                                 "-f", "null", "-"], capture_output=True, text=True, timeout=30)
        videos.append({"path": str(video), "probe": json.loads(probe.stdout),
                       "decode_exit": decode.returncode, "decode_errors": decode.stderr})
    report = {"mode": args.mode, "frame_shape": list(frame.shape), "input_frames": 60,
              "input_fps": 15, "opencv_threads": cv2.getNumThreads(),
              "wall_seconds_to_close_return": elapsed, "process_cpu_seconds_to_close_return": cpu,
              "wall_seconds_after_probe_cleanup": time.perf_counter() - start,
              "clean_calls": len(clean_ms), "clean_median_ms": statistics.median(clean_ms),
              "queue_peak": queue_peak, "writer_alive_after_close": alive_at_close,
              "stats_at_close": stats_at_close, "stats_after_probe_cleanup": rec.stats(),
              "videos": videos}
    (out / "cost.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "videos"}, ensure_ascii=False))
    if alive_at_close or rec.saved != 1 or len(videos) != 1 or any(v["decode_exit"] for v in videos):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
