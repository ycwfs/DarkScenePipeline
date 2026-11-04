"""Video I/O: reader for files/RTSP/HTTP/webcam, lazy-open writer encoding H.264 via ffmpeg."""
import os
import shutil
import subprocess
import threading
from collections import deque

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
    """Lazy-open on first frame (output dims known only after SR + label bar).

    Frames are piped raw into an ffmpeg subprocess instead of going through cv2.VideoWriter,
    because this OpenCV build has no H.264 encoder at all: its bundled FFmpeg carries only
    `h264_v4l2m2m` (a V4L2 hardware wrapper that exists on a Raspberry Pi, not on x86), so
    `avc1`/`H264`/`X264` all fail to open and every output silently came out as MPEG-4 Part 2.
    VLC and ffplay play that; Chrome, 微信, 钉钉 and Windows「电影和电视」refuse it -- which is
    exactly the "开发机上好好的，交付出去都打不开" shape of the clips bug. The ffmpeg CLI in the
    image does have libx264 (platform/Dockerfile verifies it at build time), so the encoder
    lives in a subprocess: one encode with no intermediate decode, no GIL contention with the
    recognition threads, and roughly a quarter of the bytes on disk.
    """

    # Left to itself x264 spawns a thread per core and starves the recognition threads that
    # have to hold >=15 fps. veryfast on 4 threads still encodes 720p far above real time.
    THREADS = 4
    # `+faststart` rewrites the file once at the end so it can be played before it is fully
    # downloaded (these files get pulled back out of HDFS and opened in a browser). On a long
    # offline video that is a whole-file copy, hence a generous ceiling rather than seconds.
    CLOSE_TIMEOUT = 300.0

    def __init__(self, path, fps):
        self.path = path
        fps = float(fps or 0)
        self.fps = max(1.0, fps) if fps > 0 else 25.0
        self.count = 0
        self.proc = None
        self.vw = None            # set only on the no-ffmpeg fallback path
        self._size = None         # (w, h) the encoder was actually opened with
        self._nbytes = 0
        self._err = deque(maxlen=40)
        self._err_thread = None
        self._warned_resize = False
        self._closed = False

    def _open(self, w, h):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._size, self._nbytes = (w, h), w * h * 3
        exe = shutil.which("ffmpeg")
        if not exe:
            return self._open_fallback(w, h)
        # yuv420p subsamples chroma in both axes and so cannot describe an odd dimension.
        # Pad rather than scale: one black row/column beats resampling every pixel.
        pw, ph = w + (w & 1), h + (h & 1)
        vf = [] if (pw, ph) == (w, h) else ["-vf", f"pad={pw}:{ph}:0:0"]
        cmd = [exe, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
               "-framerate", f"{self.fps:g}", "-i", "pipe:0", "-an", *vf,
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
               "-pix_fmt", "yuv420p", "-threads", str(self.THREADS),
               "-movflags", "+faststart", self.path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        # stderr MUST be drained. A piped stderr nobody reads fills its 64 KB kernel buffer
        # and then blocks ffmpeg forever -- which here means the next write() blocks and the
        # pipeline stops dead, with a live process and no error printed anywhere.
        self._err_thread = threading.Thread(target=self._drain_err, daemon=True,
                                            name=f"enc-{os.path.basename(self.path)}")
        self._err_thread.start()

    def _open_fallback(self, w, h):
        """No ffmpeg binary: keep writing something rather than failing the run, but say so."""
        print(f"[media] WARNING: 找不到 ffmpeg，改用 cv2 写 {self.path}"
              f"（编码为 MPEG-4 Part 2，浏览器和微信可能打不开）")
        for codec in ("avc1", "mp4v"):
            vw = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*codec), self.fps, (w, h))
            if vw.isOpened():
                self.vw = vw
                return
            vw.release()
        raise RuntimeError(f"cannot open VideoWriter for {self.path}")

    def _drain_err(self):
        for line in iter(self.proc.stderr.readline, b""):
            text = line.decode("utf-8", "replace").rstrip()
            if text:
                self._err.append(text)
                print(f"[media] ffmpeg({os.path.basename(self.path)}): {text}")

    def write(self, frame):
        if self._size is None:
            h, w = frame.shape[:2]
            self._open(w, h)
        elif tuple(frame.shape[1::-1]) != self._size:
            # rawvideo carries no framing: one wrong-sized frame shifts every byte after it
            # and the rest of the video decodes as garbage. (cv2.VideoWriter used to drop
            # such frames silently, which was its own quiet bug -- the count still went up.)
            if not self._warned_resize:
                self._warned_resize = True
                print(f"[media] WARNING: {self.path} 帧尺寸由 {self._size} 变为 "
                      f"{tuple(frame.shape[1::-1])}，已缩放回原尺寸")
            frame = cv2.resize(frame, self._size)
        if self.vw is not None:
            self.vw.write(frame)
            self.count += 1
            return
        buf = frame.tobytes()
        if len(buf) != self._nbytes:      # not 3-channel uint8: would desync the whole pipe
            raise RuntimeError(f"{self.path}: 期望每帧 {self._nbytes} 字节（bgr24），"
                               f"收到 {len(buf)}（shape={frame.shape} dtype={frame.dtype}）")
        try:
            self.proc.stdin.write(buf)
        except (BrokenPipeError, ValueError, OSError) as e:
            raise RuntimeError(f"ffmpeg 编码进程已退出（{self.path}）: "
                               f"{'; '.join(self._err) or e}") from None
        self.count += 1

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.vw is not None:
            self.vw.release()
            return
        if self.proc is None:
            return                        # nothing was ever written; no file to finalise
        try:
            self.proc.stdin.close()       # EOF is what makes ffmpeg write the moov atom
        except OSError:
            pass
        try:
            rc = self.proc.wait(timeout=self.CLOSE_TIMEOUT)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
            raise RuntimeError(f"ffmpeg 收尾超过 {self.CLOSE_TIMEOUT:.0f}s，已终止；"
                               f"{self.path} 不完整") from None
        if self._err_thread is not None:
            self._err_thread.join(timeout=2.0)
        try:
            self.proc.stderr.close()
        except OSError:
            pass
        if rc != 0:
            tail = "; ".join(self._err)
            raise RuntimeError(f"ffmpeg 编码失败（退出码 {rc}）: {self.path}"
                               + (f" -- {tail}" if tail else ""))
