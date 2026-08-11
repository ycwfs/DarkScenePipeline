"""Live output in formats other than MJPEG, muxed by ffmpeg.

MJPEG needs no help — `/stream` just concatenates the JPEGs the process loop already makes.
The formats a surveillance platform actually speaks do need a muxer: HTTP-FLV (what the
GB28181 gateways on the *input* side serve, so downstream can consume our output exactly the
way it consumes a camera), HLS (browser-native, at the cost of segment latency), and a push
to someone else's RTMP/RTSP server.

Every one of them is fed from the JPEG slot rather than from raw frames. That is not a
detail: the frames are already JPEG-encoded once for `/stream`, so reusing those bytes avoids
a second encode per format and cuts what goes through the pipe by roughly 10x (a 640x528 BGR
frame is ~1 MB raw against ~30 KB compressed). ffmpeg reads it as `-f mjpeg -i -` and infers
the resolution itself, so nothing here has to know the frame size.

Failure policy matches the rest of the service: a dead or missing ffmpeg degrades that one
format and says so, it never takes the recognition loop or the other outputs down.
"""
import os
import shutil
import subprocess
import threading
import time
from collections import deque

# Encoder settings shared by the low-latency outputs. ultrafast/zerolatency because this is a
# live monitor -- a smaller file is worth nothing here and every extra frame of lookahead is
# latency against the <= 1 s budget.
#
# The scale filter is not cosmetic: H.264 with yuv420p refuses odd dimensions, and libx264
# fails at *encoder init*, so the whole output never starts -- "height not divisible by 2
# (3840x2333)" from a 1080p source upscaled x2 plus a label bar. The bar is now kept even
# (darkpipe/render.py), which covers that case at the source; this crops at most one row or
# column and covers everything else, including an odd-sized source with SR off.
LOW_LATENCY = ["-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2",
               "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
               "-pix_fmt", "yuv420p"]


def rate_limit(bitrate):
    """Cap the encoder. Uncapped x264 on this content is not a tuning detail.

    Enhanced low-light video is noisy, and noise is what H.264 spends bits on: measured
    44 Mbit/s at 1080p and 179 Mbit/s at 4K with no cap. A link that cannot absorb that
    does not degrade gracefully -- frames arrive truncated, the missing rows render as
    green (zero-filled yuv420p), and the server eventually drops the connection.
    """
    if not bitrate:
        return []
    return ["-b:v", bitrate, "-maxrate", bitrate, "-bufsize", f"{bitrate}"]


FORMATS = ("mjpeg", "flv", "hls")


def ffmpeg_path():
    return shutil.which("ffmpeg")


def parse_formats(value):
    """'mjpeg, flv' -> ['mjpeg', 'flv']. Unknown names are rejected loudly, not ignored."""
    want = [f.strip().lower() for f in str(value).split(",") if f.strip()]
    bad = [f for f in want if f not in FORMATS]
    if bad:
        raise ValueError(f"unknown stream format(s) {bad}; supported: {list(FORMATS)}")
    return want or ["mjpeg"]


class FFmpegOut:
    """An ffmpeg process fed from a latest-JPEG slot by its own thread.

    `slot` is anything with .get() -> (jpeg_bytes|None, seq); the server's LatestSlot.
    """

    def __init__(self, slot, out_args, fps=15.0, pipe_stdout=False, name="ffmpeg",
                 restart=None):
        exe = ffmpeg_path()
        if not exe:
            raise RuntimeError("容器内没有 ffmpeg，无法输出该格式")
        self.name = name
        self.fps = max(1.0, float(fps))
        self.stop = threading.Event()
        self.restarts = 0
        # A per-viewer FLV process cannot be restarted -- its stdout IS the HTTP response, so
        # when it dies that response is over. Shared outputs (HLS, push) must come back: a
        # media server dropping the connection once should not leave the operator silently
        # not-streaming for the rest of a shift.
        self._restart = (not pipe_stdout) if restart is None else restart
        self._cmd = [exe, "-hide_banner", "-loglevel", "error",
                     "-f", "mjpeg", "-framerate", f"{self.fps:g}", "-i", "pipe:0", *out_args]
        self._pipe_stdout = pipe_stdout
        self._err = deque(maxlen=40)
        self._spawn_t = 0.0
        self._spawn()
        self.thread = threading.Thread(target=self._feed, args=(slot,), daemon=True,
                                       name=f"feed-{name}")
        self.thread.start()

    def _spawn(self):
        self._spawn_t = time.time()
        self.proc = subprocess.Popen(
            self._cmd, stdin=subprocess.PIPE,
            stdout=(subprocess.PIPE if self._pipe_stdout else subprocess.DEVNULL),
            stderr=subprocess.PIPE)
        # stderr MUST be drained continuously. A piped stderr nobody reads fills its 64 KB
        # kernel buffer and then blocks ffmpeg forever -- the stream simply stops, with a
        # live process and no error anywhere. Keep only the tail; it is for diagnosis.
        threading.Thread(target=self._drain_err, args=(self.proc,), daemon=True,
                         name=f"err-{self.name}").start()

    def _drain_err(self, proc):
        for line in iter(proc.stderr.readline, b""):
            text = line.decode("utf-8", "replace").rstrip()
            if text:
                self._err.append(text)
                print(f"[{self.name}] {text}")

    def _feed(self, slot):
        """Resample the slot onto a fixed cadence: encoders want a constant rate, and the
        pipeline's is not (it sags under load). Repeats the last frame if nothing is new."""
        step = 1.0 / self.fps
        due = time.time()
        backoff = 1.0
        while not self.stop.is_set():
            if self.proc.poll() is not None:            # ffmpeg exited
                if not self._restart:
                    break
                self.restarts += 1
                # Backoff resets only after a run that actually lasted, not after a
                # successful write: ffmpeg accepts a JPEG into the pipe buffer while it is
                # already dying, so keying on writes made every retry look like a recovery
                # and pinned the delay at 1 s -- a server that is down then gets hammered
                # once a second indefinitely.
                if time.time() - self._spawn_t > 15.0:
                    backoff = 1.0
                print(f"[{self.name}] 进程已退出（{self.error_tail() or '无错误输出'}），"
                      f"{backoff:.0f}s 后重启（第 {self.restarts} 次）")
                if self.stop.wait(backoff):
                    break
                backoff = min(backoff * 2, 30.0)
                try:
                    self._spawn()
                except Exception as e:                  # noqa: BLE001
                    print(f"[{self.name}] 重启失败: {e}")
                    continue
                due = time.time()
            jpg, _ = slot.get()
            if jpg:
                try:
                    self.proc.stdin.write(jpg)
                    self.proc.stdin.flush()
                except (BrokenPipeError, ValueError, OSError):
                    continue                            # loop head handles the dead process
            due += step
            time.sleep(max(0.0, due - time.time()))
        try:
            self.proc.stdin.close()
        except Exception:                               # noqa: BLE001
            pass

    def alive(self):
        return self.proc.poll() is None

    def error_tail(self):
        return " | ".join(list(self._err)[-3:])

    def close(self):
        self.stop.set()
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:                       # noqa: BLE001
            try:
                self.proc.kill()
            except Exception:                   # noqa: BLE001
                pass


def flv_pipe(slot, fps, bitrate="4M"):
    """HTTP-FLV for one client: FLV bytes on stdout, to be streamed straight to the socket.

    One process per viewer. A shared encoder would be cheaper, but fanning FLV out to clients
    that join mid-stream means holding the header and replaying from the last keyframe, i.e.
    parsing FLV tags -- real complexity for a demo endpoint that serves a handful of viewers.
    `max_flv_clients` bounds the cost instead.
    """
    return FFmpegOut(slot, [*LOW_LATENCY, *rate_limit(bitrate), "-g", str(int(max(1, fps))),
                            "-an", "-f", "flv", "pipe:1"],
                     fps=fps, pipe_stdout=True, name="flv")


def hls_writer(slot, fps, out_dir, bitrate="4M"):
    """Shared HLS segmenter. All viewers read the same segments off disk.

    1-second segments with a 6-segment window: HLS latency is roughly segments x duration, so
    this is ~3-6 s -- far worse than FLV and unavoidable for the format. delete_segments keeps
    the directory from growing without bound over a long run.
    """
    os.makedirs(out_dir, exist_ok=True)
    return FFmpegOut(slot, [*LOW_LATENCY, *rate_limit(bitrate), "-g", str(int(max(1, fps))),
                            "-an", "-f", "hls", "-hls_time", "1", "-hls_list_size", "6",
                            "-hls_flags", "delete_segments+append_list+omit_endlist",
                            "-hls_segment_filename", os.path.join(out_dir, "seg_%05d.ts"),
                            os.path.join(out_dir, "index.m3u8")],
                     fps=fps, name="hls")


def rtmp_push(slot, fps, url, bitrate="4M"):
    """Push to an external media server (rtmp:// or rtsp://) that fans out to its own clients.

    RTSP publishing is pinned to TCP. ffmpeg's rtsp muxer defaults to UDP, which needs the
    RTP/RTCP port pair to survive whatever sits between the container and the media server —
    and in a data centre that is usually a NAT and a firewall, so the ANNOUNCE succeeds and
    the media silently goes nowhere. TCP carries everything over the one connection that was
    already accepted. Both were verified against a real server here; TCP is the safer default.
    """
    extra = ["-rtsp_transport", "tcp"] if url.startswith("rtsp://") else []
    fmt = "rtsp" if url.startswith("rtsp://") else "flv"
    return FFmpegOut(slot, [*LOW_LATENCY, *rate_limit(bitrate), "-g", str(int(max(1, fps))),
                            "-an", *extra, "-f", fmt, url],
                     fps=fps, name="push")
