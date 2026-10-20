"""Run the actual v2 operator/Uvicorn entrypoint with deterministic CPU model substitutes.

The network, event bus, JPEG/FLV encoders, clip writer and session handling remain real.
No CUDA device or external camera is used. Called in a subprocess by lifecycle tests.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op-dir", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--fault", default="")
    args = parser.parse_args()
    sys.path.insert(0, args.op_dir)

    import main as operator
    from darkpipe import config, server
    from darkpipe.stages.base import RecognitionEvent

    work = Path(args.work_dir)
    trigger = work / "fail-now"

    class Stage:
        post_recognition = False

        def load(self, device):
            if args.fault == "load":
                raise RuntimeError("probe model load failure")

        def __call__(self, frames):
            if args.fault == "enhance" and trigger.exists():
                raise RuntimeError("probe enhancement failure")
            return frames

        def close(self):
            pass

    class Recognizer:
        def load(self, device):
            pass

        def warmup(self):
            pass

        def push(self, frame, idx, ts, raw_frame=None):
            if args.fault == "recognize" and trigger.exists():
                raise RuntimeError("probe recognition failure")
            return (RecognitionEvent(frame_index=idx, timestamp=ts, label="wave",
                                     confidence=0.9, topk=[("wave", 0.9)])
                    if idx % 5 == 0 else None)

        def timings(self):
            return {}

        def close(self):
            pass

    def capture(st):
        if args.fault == "waiting":
            st.stop.wait()
            return
        st.capture_alive = True
        st.src_fps = 25.0
        st.src_fps_known.set()
        try:
            if args.fault == "no_frames":
                st.stop.wait()
                return
            while not st.stop.is_set():
                frame = np.full((48, 64, 3), st.frames_in % 200, np.uint8)
                st.raw.put((frame, time.time()))
                st.frames_in += 1
                st.stop.wait(0.04)
        finally:
            st.capture_alive = False

    server.build_stages = lambda cfg: ([Stage()], Recognizer())
    server.bind_thread_device = lambda device: None
    server.capture_loop = capture
    server._encode_ceiling_probe = lambda st, cfg: None
    if args.fault == "backpressure":
        # Force a blocked ASGI send with a client that never reads its tiny receive buffer.
        server.cv2.imencode = lambda *a, **kw: (True, np.zeros(4 * 1024 * 1024, np.uint8))
    if args.fault == "clip_close":
        from darkpipe.clips import ClipRecorder, ClipWriteError
        original_close = ClipRecorder.close

        def bad_close(rec):
            original_close(rec)
            rec._write_failed("probe clip close failure")
            raise ClipWriteError(rec._writer_error)

        ClipRecorder.close = bad_close
    config.validate = lambda cfg: cfg
    operator.parse_gpu_ids = lambda value: ["0"]
    operator._print_gpu_inventory = lambda gpus: None
    operator.apply_extra_hosts = lambda spec: 0
    operator.resolve_input = lambda *a: ("stream", "http://probe/live")
    started = time.monotonic()
    try:
        operator.main([
            "--video_path", "http://probe/live", "--gpu_ids", "0",
            "--enhance", "off", "--sr", "off", "--denoise", "off",
            "--clip_denoise", "off", "--label_bar", "false",
            "--serve_port", str(args.port), "--run_seconds", str(args.seconds),
            "--stream_formats", "mjpeg" if args.fault == "backpressure" else "mjpeg,flv",
            "--extra_hosts", "",
            "--clip_dir", str(work / "clips"),
            "--session_json", str(work / "session.json"),
        ])
    finally:
        (work / "probe-timing.json").write_text(json.dumps({
            "operator_seconds": time.monotonic() - started,
        }))


if __name__ == "__main__":
    main()
