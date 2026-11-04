"""Bounded real-network capture with explicit timeouts and reconnect generations."""
import threading
import time

import cv2

from runtime import DownscaleStage, content_height, frame_luma


class NetworkCapture:
    def __init__(self, source, seconds=240, processed=False, fps=25., stop=None):
        self.source, self.seconds, self.processed = source, seconds, processed
        self.fps = fps
        self.origin = time.monotonic()
        self.duration_seconds = 0.
        self.count = self.total = self.dropped = 0
        self.stop = stop or threading.Event()
        self.done = threading.Event()
        self.condition = threading.Condition()
        self.slot = None
        self.errors = []
        self.connections = []
        self.last_frame_at = None
        self.epoch = -1
        self.thread = threading.Thread(target=self._run, daemon=True, name='person-network-capture')
        self.thread.start()

    def _run(self):
        cap = None
        downscale = DownscaleStage(720)
        backoff = .5
        deadline = self.origin+self.seconds if self.seconds else float('inf')
        try:
            while not self.stop.is_set() and time.monotonic() < deadline:
                try:
                    # Both open and read must be bounded: a peer may keep TCP connected
                    # without sending any bytes. A stop flag alone cannot interrupt C++ read.
                    cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, [
                        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000,
                        cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000])
                    if not cap.isOpened():
                        raise RuntimeError('stream open failed')
                    self.epoch += 1
                    self.connections.append(dict(epoch=self.epoch, opened=time.monotonic()-self.origin))
                    while not self.stop.is_set() and time.monotonic() < deadline:
                        ok, frame = cap.read()
                        if not ok:
                            raise RuntimeError('stream read failed or timed out')
                        stamp = time.monotonic()-self.origin
                        if self.connections[-1].get('first_frame') is None:
                            self.connections[-1]['first_frame'] = stamp
                        self.last_frame_at = stamp
                        shown = frame[:content_height(frame.shape[0])] if self.processed else None
                        raw = downscale([shown if self.processed else frame])[0]
                        luma = None if self.processed else frame_luma(raw)
                        with self.condition:
                            if self.slot is not None:
                                self.dropped += 1
                            self.slot = (self.total, raw, shown, luma, stamp, self.epoch)
                            self.total += 1
                            self.condition.notify_all()
                        backoff = .5
                except Exception as exc:
                    self.errors.append(dict(time=time.monotonic()-self.origin, error=str(exc)))
                    self.stop.wait(backoff)
                    backoff = min(4., backoff*2)
                finally:
                    if cap is not None:
                        cap.release()
                        cap = None
        finally:
            self.duration_seconds = time.monotonic()-self.origin
            self.done.set()
            with self.condition:
                self.condition.notify_all()

    def read(self):
        with self.condition:
            while self.slot is None and not self.done.is_set():
                self.condition.wait(.1)
            if self.slot is None:
                return None
            result, self.slot = self.slot, None
            self.count += 1
            return result

    def close(self):
        self.stop.set()
        self.thread.join(timeout=8)
        if self.thread.is_alive():
            raise RuntimeError('Network capture did not stop within its I/O deadline')

    def stats(self):
        return dict(received=self.total, consumed=self.count, dropped=self.dropped,
                    duration=self.duration_seconds, connections=self.connections,
                    errors=self.errors, last_frame_at=self.last_frame_at,
                    thread_alive=self.thread.is_alive())
