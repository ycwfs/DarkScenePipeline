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
    def __init__(self, src, chunk=32, max_frames=None):
        self.src = src
        self.chunk = chunk
        self.max_frames = max_frames
        self.cap = open_capture(src)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 0/neg for streams

    def chunks(self):
        n = 0
        while True:
            batch = []
            while len(batch) < self.chunk:
                ok, f = self.cap.read()
                if not ok or (self.max_frames and n >= self.max_frames):
                    break
                batch.append(f); n += 1
            if batch:
                yield batch
            if len(batch) < self.chunk:
                break
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
