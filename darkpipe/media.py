"""Video I/O: reader for files/RTSP/HTTP/webcam, lazy-open writer with codec fallback."""
import os

import cv2


def open_capture(src):
    """cv2.VideoCapture from a file path, rtsp/http URL, or webcam index."""
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open input source: {src!r}")
    return cap


class VideoReader:
    def __init__(self, src, chunk=32, max_frames=None, start_frame=0):
        self.src = src
        self.chunk = chunk
        self.max_frames = max_frames
        self.cap = open_capture(src)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 0/neg for streams
        self.frames_read = 0
        self.recovered = 0
        # Decode-and-discard rather than CAP_PROP_POS_FRAMES: seeking lands on a keyframe and
        # cv2 reports the requested index either way, so a shard would silently start a few
        # frames off and duplicate or drop frames at the segment boundary.
        for _ in range(max(0, start_frame)):
            if not self.cap.read()[0]:
                break

    # A failed read is not proof of end-of-stream. Measured: the same 602-frame file stopped
    # at 536 on one run and read to the end on the next, same code, same container -- a
    # transient decoder miss that the old `if not ok: break` turned into a silently truncated
    # output video. Retry a bounded number of times; at a real EOF these fail immediately and
    # cost nothing.
    READ_RETRIES = 5

    def chunks(self):
        n = 0
        done = False
        while not done:
            batch = []
            while len(batch) < self.chunk:
                if self.max_frames and n >= self.max_frames:
                    done = True
                    break
                ok, f = self.cap.read()
                if not ok:
                    misses = 1
                    while misses <= self.READ_RETRIES:
                        ok, f = self.cap.read()
                        if ok:
                            break
                        misses += 1
                    if not ok:
                        done = True
                        break
                    self.recovered += 1
                batch.append(f); n += 1
            if batch:
                yield batch
        self.frames_read = n
        # Silent truncation is the failure worth shouting about: the run still "succeeds",
        # the summary reports its own (short) count, and nobody can tell footage went missing.
        if self.n_frames > 0 and not self.max_frames and n < self.n_frames:
            print(f"[media] WARNING: read {n} of {self.n_frames} frames declared by "
                  f"{self.src!r} -- output is short by {self.n_frames - n}")
        if self.recovered:
            print(f"[media] recovered from {self.recovered} transient read failure(s)")
        self.cap.release()

    def read_all(self):
        frames = [f for c in self.chunks() for f in c]
        if len(frames) > 2000:
            print(f"[media] WARNING: {len(frames)} frames held in RAM (whole-video stage)")
        return frames


class VideoWriter:
    """Lazy-open on first frame (output dims known only after SR + label bar)."""

    def __init__(self, path, fps):
        self.path = path
        self.fps = fps
        self.vw = None
        self.count = 0

    def write(self, frame):
        if self.vw is None:
            h, w = frame.shape[:2]
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            for codec in ("mp4v", "avc1"):
                self.vw = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*codec),
                                          self.fps, (w, h))
                if self.vw.isOpened():
                    break
            if not self.vw.isOpened():
                raise RuntimeError(f"cannot open VideoWriter for {self.path}")
        self.vw.write(frame)
        self.count += 1

    def close(self):
        if self.vw is not None:
            self.vw.release()
