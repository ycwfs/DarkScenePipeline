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
import queue
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
               "-pix_fmt", "yuv420p",
               # The mjpeg input decodes to yuvj420p and swscale converts it to yuv420p without
               # carrying any colour metadata (hence the "deprecated pixel format used" warning),
               # so the output came out tagged bt470bg -- PAL, wrong for a BT.709 1080p source,
               # and some players render it with a visible cast. Tagging the matrix fixes that
               # field and brings color_range=tv along with it; this writes VUI metadata only and
               # does not touch a single pixel. -color_primaries/-color_trc are deliberately not
               # set: measured on ffmpeg 8.1.2, the libx264 wrapper drops both (they stay
               # `unknown` in the VUI regardless of container), so they would only look useful.
               "-colorspace", "bt709"]

# Requested size of ffmpeg's stdin pipe. See FFmpegOut._widen_stdin for why the 64 KiB default
# is the wrong size when one frame is ~119 KB, and why 1 MiB is the largest useful value. The
# request is reported by _spawn as `stdin_pipe=` because it silently did nothing once already.
PIPE_BYTES = 1 << 20

# Frames the cadence thread may run ahead of the writer thread. See FFmpegOut._write_loop. Six
# frames is 400 ms at 15 fps, which covers the worst write measured in deployment (225 ms) with
# room to spare, and is the whole added latency in the bad case -- so it stays small on purpose.
RING_FRAMES = 6

# Socket I/O timeout handed to ffmpeg for the push, in microseconds, and the watchdog deadline
# that does not trust it. Both exist because of the same failure, reproduced locally against a
# peer that stops reading (probe in the plan notes, ffmpeg 8.1.2):
#
#   t+0s    peer stops reading. Everything stays green -- `-progress frame=` and `total_size`
#           keep advancing, writes stay at 0.1 ms -- for as long as the socket buffers absorb
#           (2.5-4 s at 4 Mbit/s with ~400 KiB buffers).
#   t+2.5s  ffmpeg's socket write blocks, so it stops draining stdin and stdin.write() blocks.
#           With no timeout it NEVER returns: the probe sat wedged for over five minutes.
#
# That is the deployment's freeze: `_write_loop` stuck in stdin.write(), `alive()` still true
# because it only asks poll(), so the existing restart path never fires and the picture stays
# frozen until the container is restarted.
#
# 5 s is ~10x the worst legitimate write measured on this link (peak_block 102 ms; the arrival
# study in rtmp_push saw gaps to 446 ms over 15 minutes). Lower risks restarting a merely
# jittery link, higher just lengthens the freeze.
#
# What does NOT work, so nobody adds it again: parsing `-progress`. total_size counts bytes
# handed to the muxer, not bytes the peer took, so it keeps advancing through exactly this
# failure -- measured, the peer received nothing for 7 s while progress went 100 -> 251 frames.
# There is no observation of our own socket that can see a peer which reads and then discards;
# that one needs the stream pulled back, which is deliberately not done here.
PUSH_TIMEOUT_US = 5_000_000
# How long a blocked stdin.write() is tolerated before the watchdog kills ffmpeg itself. Two
# values, because whether ffmpeg's own timeout above works at all depends on the muxer -- measured
# on the loopback sinks, same 840x472 4 Mbit/s stream, ffmpeg 8.1.2:
#
#   flv over tcp://   -rw_timeout honoured. ffmpeg exits by itself at a wedge age of 8.7-9.1s.
#   rtsp://           -timeout NOT honoured. ffmpeg sat blocked 15.6s and never gave up; only
#                     the watchdog ended it. This is the branch the deployment uses.
#
# Note the 5 s clock does not start when *our* write blocks -- it starts ~4 s later, when ffmpeg's
# own socket write does, with x264 and the AVIO buffer in between. Hence 8.7s, not 5s.
#
# So on flv the watchdog must stay clear of ffmpeg's deadline (at 8 s it won every race, giving
# kills=4/restarts=4, which reads exactly like a broken option), while on rtsp there is nothing to
# defer to and every extra second is pure freeze. Both recover; the wait is the only difference.
PUSH_STALL_KILL_SEC = 15.0      # default: ffmpeg may well beat us to it, leave it room
PUSH_STALL_KILL_RTSP = 8.0      # rtsp: nothing else will act, so act sooner. ~18x the worst
                                # legitimate write on this link (peak_block 102 ms, arrival gaps
                                # to 446 ms over 15 minutes), so a jittery link is still safe.


def rate_limit(bitrate):
    """Optional cap on the encoder, for a link that cannot carry the stream.

    Enhanced low-light video is noisy, and noise is what H.264 spends bits on: measured
    44 Mbit/s at 1080p and 179 Mbit/s at 4K uncapped. A link that cannot absorb that does
    not degrade gracefully -- frames arrive truncated, the missing rows render as green
    (zero-filled yuv420p), and the server eventually drops the connection.

    That said, capping is a mitigation, not a fix: the deployment those symptoms came from
    turned out to have a network fault, and the cap only softened the picture. So "no cap"
    has to stay reachable, and an empty string is not enough -- on the platform this field
    carries a default, which the spec makes mandatory and non-empty, so the UI cannot send
    one. `0` / `off` / `none` mean the same thing.

    Capped VBR (`-crf 23 -maxrate B -bufsize B`, dropping `-b:v`) was measured and rejected.
    The idea was that it would stop paying for duplicate frames: when the processing thread
    falls behind max_stream_fps the cadence thread re-sends the previous JPEG (FFmpegOut._feed,
    counted as `dup`), and a measured 21-28% of delivered frames carry no new picture. Feeding
    100 copies of one JPEG, that works exactly as hoped -- P frames drop from 13716 B to 1569 B,
    8.7x cheaper, because the reconstruction converges to a fixed point and everything becomes
    skip. On a real stream with duplicates interleaved it does not: each duplicate follows a
    *different* lossy reconstruction, so the residual it has to code is the previous frame's
    quantisation error rather than zero. Measured on 1200 frames of deployment output with 16.7%
    duplicates injected, 1440x874: duplicates cost 15.3 KB under CBR and 12.4 KB under capped
    VBR, against 16.0 / 13.6 KB for fresh frames -- i.e. never free either way. What was left was
    4.04 -> 3.49 Mbit/s (-13.5%) for 49.78 -> 49.27 dB PSNR and 0.99239 -> 0.99153 SSIM, which is
    just a slightly lower point on the same rate-distortion curve, not a saving. Not worth taking
    on a deployment whose open goal is to raise the working resolution.

    It would also have broken encode_ceiling(), which feeds 60 copies of one JPEG -- precisely
    the converging case above -- so its measured ceiling would have jumped without the encoder
    getting any faster.
    """
    if str(bitrate).strip().lower() in ("", "0", "off", "none", "no"):
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
    """An ffmpeg process fed from a latest-JPEG slot by two threads of its own.

    `slot` is anything with .get() -> (jpeg_bytes|None, seq); the server's LatestSlot.

    _feed holds the cadence and _write_loop does the writing, with a RING_FRAMES ring between
    them. They are split because a write to ffmpeg's stdin can take several cadence slots and
    used to be paid for out of the frame rate -- see _write_loop.
    """

    def __init__(self, slot, out_args, fps=15.0, pipe_stdout=False, name="ffmpeg",
                 restart=None, loglevel="error", gop=0, stall_kill=None):
        exe = ffmpeg_path()
        if not exe:
            raise RuntimeError("容器内没有 ffmpeg，无法输出该格式")
        self.name = name
        self.fps = max(1.0, float(fps))
        self.stop = threading.Event()
        self.restarts = 0
        # Keyframe period in written frames, for the IDR/P split in _feed. -tune zerolatency
        # sets scenecut=0, so IDRs are strictly periodic and the position is knowable here
        # without parsing the bitstream. 0 disables the split.
        self._gop = int(gop)
        # A per-viewer FLV process cannot be restarted -- its stdout IS the HTTP response, so
        # when it dies that response is over. Shared outputs (HLS, push) must come back: a
        # media server dropping the connection once should not leave the operator silently
        # not-streaming for the rest of a shift.
        self._restart = (not pipe_stdout) if restart is None else restart
        self._cmd = [exe, "-hide_banner", "-loglevel", loglevel,
                     "-f", "mjpeg", "-framerate", f"{self.fps:g}", "-i", "pipe:0", *out_args]
        self._pipe_stdout = pipe_stdout
        self._err = deque(maxlen=40)
        # Real stdin pipe size and why, filled by _widen_stdin on every _spawn. Initialised here
        # so a caller can report them even if _spawn raised before getting that far.
        self.pipe_bytes = 0
        self.pipe_note = "未尝试"
        self._spawn_t = 0.0
        # Push-side telemetry. `[process]` says nothing about any of this -- the JPEG slot is
        # latest-wins, so the process loop never blocks on an output and its numbers stay
        # perfect while a stream is dead. Same lock-free discipline as _Stat, with one addition
        # now that three threads live here: every key has exactly ONE writing thread. The
        # cadence thread (_feed) owns dup / missed / adv* / lag / skipped / ring_full; the
        # writer thread (_write_loop) owns everything about the write itself; the watchdog owns
        # `kills`. Readers snapshot and diff.
        #
        # What these counters do NOT prove, since this comment used to claim they were "the only
        # evidence of what the viewer actually got": every one of them is billed on the INPUT
        # side of ffmpeg. Measured against a peer that stops reading, all of them -- including
        # `stall`, which is time since the last successful write into stdin -- stay perfect for
        # as long as the socket buffers absorb, and `alive()` stays true forever after that. See
        # PUSH_TIMEOUT_US. They describe what ffmpeg was given, not what left the container.
        self._c = dict(frames=0, dup=0, bytes=0, blocked=0.0, worst_ms=0.0, lag=0.0,
                       skipped=0.0, idr_n=0, idr_blocked=0.0, idr_worst=0.0,
                       p_n=0, p_blocked=0.0, w_fast=0, w_mid=0, w_timer=0, w_over=0,
                       ring_full=0, missed=0, adv0=0, adv1=0, adv2=0, adv3p=0, kills=0)
        self._last_ok = 0.0
        # Wall time the in-flight stdin.write() started, 0.0 when no write is in flight. Owned by
        # the writer thread, read by the watchdog. This, and not `stall`, is what the watchdog
        # keys on: `stall` also grows when there is simply nothing to send (a dead capture, a
        # paused source), and killing a healthy ffmpeg for that would turn a source outage into
        # an output outage. A write that has not returned is unambiguous.
        self._write_t = 0.0
        # How long a blocked write is tolerated before this object kills the child itself. The
        # default leaves room for ffmpeg's own timeout to act first; a caller whose muxer is known
        # not to honour that option passes a shorter one (see rtmp_push's rtsp branch), because
        # there is then nothing to leave room for and the wait is pure freeze.
        self._stall_kill = PUSH_STALL_KILL_SEC if stall_kill is None else float(stall_kill)
        # The cadence and the write are separate threads with a small ring between them, so a
        # slow write cannot eat a cadence slot. See _write_loop for the measurement.
        self._ring = queue.Queue(maxsize=RING_FRAMES)
        self._spawn()
        self.writer = threading.Thread(target=self._write_loop, daemon=True,
                                       name=f"write-{name}")
        self.writer.start()
        self.thread = threading.Thread(target=self._feed, args=(slot,), daemon=True,
                                       name=f"feed-{name}")
        self.thread.start()
        # Only for outputs that can come back. A per-viewer FLV process is not restartable (its
        # stdout is the HTTP response), so killing it would end that response for good and buy
        # nothing -- the client can just reconnect.
        self.watchdog = None
        if self._restart:
            self.watchdog = threading.Thread(target=self._watch, daemon=True,
                                             name=f"wd-{name}")
            self.watchdog.start()

    def _watch(self):
        """Kill a child whose stdin.write has been blocked past the deadline, so that _feed's
        existing restart path can run.

        This is the half of the fix that does not trust ffmpeg, and on the deployment's own path it
        is the only half that works: `-timeout` is silently ignored by the rtsp muxer (measured --
        15.6 s blocked, no exit), while `-rw_timeout` on flv/tcp does fire. So for rtsp this thread
        is not a backstop, it is the mechanism, and `kills` climbing 1:1 with `restarts` there is
        the expected reading rather than a fault. An option that silently fails to take effect is
        the documented normal here (see _widen_stdin, which failed exactly that way for a whole
        deployment round), which is why this exists at all.

        Killing is enough because every downstream piece already exists: the blocked write raises
        OSError, which _write_loop swallows, and _feed's loop head sees poll() != None and
        respawns with backoff. Nothing else has to change.
        """
        while not self.stop.wait(1.0):
            started = self._write_t          # read once; the writer may clear it under us
            if not started:
                continue
            held = time.time() - started
            if held < self._stall_kill:
                continue
            self._c["kills"] += 1
            # Loud, and says which of the two mechanisms fired: reaching this line means
            # ffmpeg's own timeout did not, so the `timeout=` in the command line is either
            # absent or not reaching the socket. That distinction is the whole diagnostic.
            print(f"[{self.name}] 写入已阻塞 {held:.1f}s（超过 {self._stall_kill:.0f}s），"
                  f"对端疑似停止接收；杀掉 ffmpeg 重连（累计 {self._c['kills']} 次）")
            try:
                self.proc.kill()
            except Exception as e:                      # noqa: BLE001
                print(f"[{self.name}] kill 失败: {e}")
            # Do not re-arm until the writer reports a new in-flight write, otherwise the same
            # stuck timestamp fires again every second while the restart is still in progress.
            while not self.stop.wait(1.0):
                if self._write_t != started:
                    break

    def _spawn(self):
        self._spawn_t = time.time()
        self.proc = subprocess.Popen(
            self._cmd, stdin=subprocess.PIPE,
            stdout=(subprocess.PIPE if self._pipe_stdout else subprocess.DEVNULL),
            stderr=subprocess.PIPE)
        self.pipe_bytes = self._widen_stdin()
        # stderr MUST be drained continuously. A piped stderr nobody reads fills its 64 KB
        # kernel buffer and then blocks ffmpeg forever -- the stream simply stops, with a
        # live process and no error anywhere. Keep only the tail; it is for diagnosis.
        threading.Thread(target=self._drain_err, args=(self.proc,), daemon=True,
                         name=f"err-{self.name}").start()

    def _widen_stdin(self):
        """Grow ffmpeg's stdin pipe from the 64 KiB default to PIPE_BYTES; return the real size.

        A pipe smaller than one frame makes a write finish only after ffmpeg has decoded, encoded
        AND sent the *previous* frame. Measured in deployment: 39-42 ms of blocking per frame,
        43 ms on keyframes -- a 1.11x ratio, i.e. a cost that ignores frame size, so neither
        bandwidth nor the keyframe burst. Two rounds of shrinking bytes (JPEG q100->85, then VBV
        at 3M) confirmed that by moving blocked% only 63.4% -> 55.8%.

        This did NOT help, and the reason it was even arguable is the reason it now returns a
        value. Measured after deploying it: `blocked` 55.8% -> 57.0%, and of 2243 writes exactly
        ZERO completed in under 5 ms (80% landed in 25-60 ms). A 1 MiB pipe is ~9 frames of
        headroom against a consumer measured at 42 ms/frame and a cadence of 66.7 ms, so it
        should have left the pipe empty and made writes a memcpy. Getting no fast writes at all
        is what a 64 KiB pipe looks like -- but the call swallowed its exception and printed
        nothing, so "it did not work" and "it never ran" were indistinguishable for a whole
        deployment round. Hence the return value, printed by the caller as `stdin_pipe=`.

        Either way it is no longer load-bearing: _write_loop takes the write off the cadence
        thread, which is the same slack in a place that cannot silently fail to exist. That is
        confirmed -- with the writer thread in place, `dropped` is 0.0% in all 141 windows of a
        60 s run whose writes still cost ~35 ms each (`blocked` p50 55%, zero writes under 5 ms,
        i.e. still a 64 KiB pipe). So the pipe size now buys latency, not frames. Keep it as long
        as `stdin_pipe=` shows it working, since a kernel buffer is free; drop it if the log keeps
        showing 65536 with a note explaining why it cannot grow.

        Linux only: F_SETPIPE_SZ does not exist on macOS, and neither does the 64 KiB pipe that
        makes it necessary. Failure never stops the stream -- a tuning hint must not -- but it is
        no longer silent: the reason lands in `self.pipe_note`, printed next to the size. That
        distinction had to be built twice. The first version put the SET and the GET in one try
        block, so a raising SET skipped the GET and returned 0 -- which reads exactly like
        "F_SETPIPE_SZ does not exist here", the one thing the return value existed to rule out.
        Deployment then showed `stdin_pipe=0B` on Python 3.10.20, which *has* the constant.

        The likely reason for a raising SET, and why the request is now capped: F_SETPIPE_SZ
        raises EPERM above /proc/sys/fs/pipe-max-size unless the process holds CAP_SYS_RESOURCE,
        and a container usually does not.
        """
        self.pipe_note = ""
        try:
            import fcntl
        except Exception as e:                          # noqa: BLE001
            self.pipe_note = f"无 fcntl：{e}"
            return 0

        fd = self.proc.stdin.fileno()
        op = getattr(fcntl, "F_SETPIPE_SZ", None)
        if op is None:
            self.pipe_note = "本平台无 F_SETPIPE_SZ"
        else:
            want = PIPE_BYTES
            try:
                with open("/proc/sys/fs/pipe-max-size") as f:
                    cap = int(f.read().strip())
                if 0 < cap < want:
                    want = cap
            except Exception:                           # noqa: BLE001
                pass                                    # no procfs: just try and report
            # Descending, because a rejected size says nothing about a smaller one, and even
            # 256 KiB is two frames of headroom -- most of the benefit of the full request.
            for size in (want, 1 << 18, 1 << 17):
                if size <= 65536:
                    continue
                try:
                    fcntl.fcntl(fd, op, size)
                    break
                except Exception as e:                  # noqa: BLE001
                    self.pipe_note = f"F_SETPIPE_SZ({size}) 失败：{e}"

        # Its own try: the point of reading the size back is to report what the pipe ACTUALLY is,
        # which matters most in exactly the case where the write above failed.
        get = getattr(fcntl, "F_GETPIPE_SZ", None)
        if get is None:
            self.pipe_note = (self.pipe_note + "；" if self.pipe_note else "") + "无 F_GETPIPE_SZ"
            return 0
        try:
            return int(fcntl.fcntl(fd, get))
        except Exception as e:                          # noqa: BLE001
            self.pipe_note = (self.pipe_note + "；" if self.pipe_note else "") + f"读回失败：{e}"
            return 0

    # Lines that are always emitted and never explain anything, so they must not enter the ring
    # error_tail() reads: with them in, a child that was killed (SIGKILL, so no stderr at all)
    # reported "进程已退出（[swscaler ...] deprecated pixel format used）" -- a startup warning
    # from ten minutes earlier, presented as the cause of death. They are still printed; they are
    # just not allowed to be the explanation.
    _ERR_NOISE = ("deprecated pixel format used",        # swscaler, once per child on yuvj420p
                  "Failed to update header with correct")  # flv, on any close it cannot seek back

    def _drain_err(self, proc):
        # Consecutive duplicates are collapsed. ffmpeg happily emits the same warning once per
        # frame ("Non-monotonous DTS", RTSP retransmits), which at 15 fps buries every other
        # line in the operator log -- and this is the log the deployment reads. The count is
        # reported when the message finally changes, so nothing is lost, only repeated.
        prev, reps = None, 0
        for line in iter(proc.stderr.readline, b""):
            text = line.decode("utf-8", "replace").rstrip()
            if not text:
                continue
            if text == prev:
                reps += 1
                continue
            if reps:
                print(f"[{self.name}] 上一条重复 {reps} 次")
            prev, reps = text, 0
            if not any(n in text for n in self._ERR_NOISE):
                self._err.append(text)
            print(f"[{self.name}] {text}")

    def _feed(self, slot):
        """Resample the slot onto a fixed cadence: encoders want a constant rate, and the
        pipeline's is not (it sags under load). Repeats the last frame if nothing is new.

        This thread does not write. It hands the frame to _write_loop through a ring bounded at
        RING_FRAMES, so its own loop costs microseconds and the cadence is decided by the clock
        alone. Whether that ring ever fills is the honest answer to "is the peer too slow", and
        it is reported as ring_full.
        """
        step = 1.0 / self.fps
        due = time.time()
        backoff = 1.0
        last_seq = -1
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
                # Whatever is still in the ring belongs to the process that just died. Handing
                # it to the new one would open the stream with a burst of frames already older
                # than its first timestamp.
                while True:
                    try:
                        self._ring.get_nowait()
                    except queue.Empty:
                        break
                due = time.time()
            jpg, seq = slot.get()
            if jpg:
                # Content advance: how many frames the pipeline produced since the one we last
                # delivered. `seq` counts puts into the latest-wins slot, so anything past the
                # first was overwritten there and no output ever encoded it.
                #
                #   0  -- nothing new, the previous frame goes out again (`dup`)
                #   1  -- 1:1, every produced frame delivered
                #   n  -- n-1 frames overwritten in the slot, never sent
                #
                # Why the split is worth four counters: `dup` alone could not see the losing
                # half of this. Measured on a 25 fps source at a 20 fps cadence (87 windows,
                # 174 s): dup 8.0% of delivered frames, while 26% of PRODUCED frames were being
                # overwritten -- 3x more, with nothing in the log naming them. The ratio only
                # requires dropping 20%; the excess tracks the dup rate, i.e. every duplicate
                # also costs a real frame. Conservation check, useful when changing this:
                # missed + (adv1+adv2+adv3p) == total puts into the slot, exactly.
                #
                # What these counters DISPROVED, so nobody spends another round on it: the
                # deployment saw 15 and 25 fps stream cleanly while 20 fps stuttered, and the
                # suspicion was that the cadence margin (1/fps minus the source interval:
                # 26.7 ms at 15, 10.0 ms at 20, 0 at 25) falling under the pipeline's jitter
                # randomises which frame survives, making 20 a pessimum between "regular
                # decimation" and "1:1, nothing discarded". It is not, and neither is any other
                # delivery property. Five deployment runs on one 25.00 fps camera, everything
                # but max_stream_fps equal (an earlier three-run set on the same camera measured
                # 2-3x these dup rates but the same ordering):
                #
                #     out fps   r=src/out   adv 0/1/2/3+     dup%   missed%   blocked%   deploy
                #     12.5      2.000       0/108/587/90      0.0     49.4       39      fine
                #     15        1.667       5/358/564/14      0.5     38.7       48      fine
                #     17        1.471       2/519/440/1       0.2     31.5       52      STUTTERS
                #     20        1.250       41/1391/510/0     2.1     21.2       59      STUTTERS
                #     25        1.000       84/802/74/0       8.8      7.8       64      fine
                #
                # No threshold on any of those can separate {17,20} from {12.5,15,25} -- every
                # column interleaves them, as do write-block ms, pipe bitrate, latency p50 and
                # restarts (0 everywhere). Not "we failed to find one": they interleave, so none
                # exists. 12.5 discards the MOST frames of the five and is fine; 25 freezes most
                # and is fine; 17 has the most regular histogram of all five (3 frames of 962
                # outside {1,2}) and stutters, which reverse-refutes the irregularity story
                # these four counters were added to test.
                #
                # What does separate them is not about delivery at all. `adv` is an integer while
                # the correct advance is r = src_fps/out_fps, so every delivered frame carries
                # floor(r) or ceil(r) source frames of motion, never r, while -framerate stamps
                # them exactly 1/out_fps apart. Uniform pts, uneven content:
                #
                #     (ceil(r) - r) * 1000/src_fps  ms   == 0 exactly when r is an integer
                #
                # 0.0 / 13.3 / 21.2 / 30.0 / 0.0 ms for the rows above -- the only column that
                # splits fine from stuttering, and it explains why the bad set is not an
                # interval: the penalty rises with out_fps inside r in (1,2), peaks at
                # r = sqrt(2) (17.7 fps on a 25 fps source), then drops discontinuously to 0 at
                # r = 1. Perceptual threshold bracketed to 13.3-21.2 ms, about half a source
                # frame. The 15-fine/17-stutters pair also rules duration out as the confound:
                # 63 s and 57 s, adjacent cadences, opposite outcomes.
                #
                # The fix is upstream of this loop -- server.stream_fps_for_source snaps
                # max_stream_fps to src_fps/k so the term is identically zero. Two things that
                # do NOT fix it: making the extraction pattern regular (a Bresenham accumulator
                # over the slot) -- 17 fps is already regular and still stutters -- and reducing
                # the drop rate, see 12.5. Making `adv` fractional is impossible.
                #
                # That the CFR stamping is what discards the timing was measured, not inferred:
                # JPEGs written into an ffmpeg stdin 45/44/41/84 ms apart (the 25->20 pattern)
                # came back out with pts at exactly 50.0 ms throughout. Two ways out of it, both
                # measured on ffmpeg 8.1.2, the version the -colorspace note above was measured
                # on:
                #
                #   VFR timestamps -- `-use_wallclock_as_timestamps 1` on the input plus
                #     `-fps_mode passthrough` on the output does reproduce the real write timing,
                #     in FLV as well as Matroska (FLV matters: it is a default format here and
                #     the RTMP push). Removes the term at ANY cadence and would also retire the
                #     21-49% silent drop. Not done: it makes max_stream_fps a ceiling rather than
                #     a target, `-g <frames>` becomes meaningless (needs -force_key_frames, which
                #     breaks the _gop IDR/P split above), FLV/RTMP timestamps are millisecond
                #     resolution so two frames inside one ms collide -- one 0.0 ms delta appeared
                #     in 39 samples and would need a monotonicity guard -- and it removes the
                #     re-send-the-previous-JPEG behaviour that is still a suspect for the
                #     unexplained player reconnects below.
                #   Frame interpolation -- the only way to keep a strictly CFR non-integer
                #     cadence AND be smooth, since it synthesises the missing positions. Dead on
                #     cost: at 840x472 minterpolate runs 0.54x realtime (mci+obmc) and 0.29x
                #     (mci+aobmc+bidir) on 8 idle cores, against a container pinning x264 to 4
                #     threads. mi_mode=blend is free but trades judder for motion ghosting.
                #
                # Still open, and NOT explained by any of the above: the same deployment reported
                # the player reconnecting. Content judder cannot cause a reconnect. The suspect
                # is arrival starvation -- 118 KB JPEGs through the 8192 B stdin pipe this
                # container is granted (see _widen_stdin), with blocked% at 39-64% and w_fast at
                # 0 in all five runs.
                #
                # Accounting note: after an ffmpeg restart the first advance covers the whole
                # dead window, so those frames land in `missed` and adv3p. That is honest --
                # they were produced and never sent -- but it is a restart, not resampling, so
                # check `restarts` before reading a `missed` spike as a cadence problem.
                adv = seq - last_seq if last_seq >= 0 else 1  # first sample has no predecessor
                if adv <= 0:
                    self._c["dup"] += 1                 # pipeline slower than the cadence
                else:
                    self._c["missed"] += adv - 1
                self._c["adv0" if adv <= 0 else "adv1" if adv == 1 else
                        "adv2" if adv == 2 else "adv3p"] += 1
                last_seq = seq
                try:
                    self._ring.put_nowait(jpg)
                except queue.Full:
                    # RING_FRAMES behind and still not drained: the writer is not keeping up
                    # over a sustained stretch, not just in one spike. That is the "the peer is
                    # genuinely slower than max_stream_fps" signal, and until this counter
                    # existed there was nothing in the log that said it -- blocked% could not,
                    # because it was measuring the cadence thread paying for the writes itself.
                    self._c["ring_full"] += 1
            due += step
            behind = time.time() - due
            self._c["lag"] = max(self._c["lag"], behind)
            if behind > step:
                # Cannot hold the cadence. With the write on another thread this should now be
                # rare -- the loop above is a slot.get() and a put_nowait -- so if `dropped`
                # stays high, look at the scheduler or at ring_full, not at the network.
                #
                # The debt must be forgiven rather than repaid, because repaying it writes the
                # backlog back to back at sleep(0) while ffmpeg keeps stamping frames 1/fps
                # apart -- so the stream's timeline falls behind wall clock without bound.
                # Measured in deployment: 30 fps requested, 28.8 sustainable, and `lag` grew
                # monotonically to 30 s over ~11 min with no plateau. Dropping the backlog
                # instead means the stream runs at whatever rate is achievable and stays at
                # the live edge, which is what a live viewer needs. `skipped` is the evidence
                # that this is happening, and its rate is exactly the fps shortfall.
                self._c["skipped"] += behind
                due = time.time()
            time.sleep(max(0.0, due - time.time()))
        # Tell the writer there is nothing more coming. It owns stdin, so it also closes it --
        # closing it here would tear the fd out from under a write already in flight.
        self.stop.set()
        try:
            self._ring.put_nowait(None)
        except queue.Full:
            pass

    def _write_loop(self):
        """Own stdin.write, so that a slow write costs frames from the ring and never a cadence
        slot. All write-side telemetry is billed here.

        Why this thread exists, measured across two deployments: a single write reached 225 ms,
        which is more than three 66.7 ms cadence slots at 15 fps. With the write inline in
        _feed, every such write pushed the cadence past its next due time and the debt was
        forgiven -- i.e. dropped. The arithmetic came out exact: writes over 60 ms averaged 2.88
        per telemetry window and the slots implied by `dropped` averaged 2.89. So 9.6% of the
        stream was being discarded to pay for the tail of a link that has plenty of average
        throughput (the encoder's own ceiling measured 42.2 fps in-container against the 15 fps
        asked for, and the write cost is independent of frame size).

        Enlarging ffmpeg's stdin pipe was the same idea one layer down and could not be shown to
        work (see _widen_stdin). This can: the ring is in this process, its depth is a constant
        here, and ring_full says when it is not enough.

        The cost is honest and bounded: in a stall, frames wait in the ring, so up to
        RING_FRAMES/fps of extra latency -- 400 ms at 15 fps against the service's <=1 s budget.
        Shrink RING_FRAMES before anything else if latency becomes the binding constraint.

        Verified locally against a `-f null` sink at 15 fps (no network, so this checks the
        mechanism, not the deployment): 60 frames written in 4 s -- exactly the cadence -- with
        two writes over 60 ms (worst 108 ms, ffmpeg startup) and `skipped` still 0.0. Inline,
        those two writes would each have cost a cadence slot.
        """
        while True:
            try:
                jpg = self._ring.get(timeout=0.1)
            except queue.Empty:
                if self.stop.is_set():
                    break
                continue
            if jpg is None:
                break
            proc = self.proc          # read once: _feed may respawn between the write and the
            t_w = time.time()         # error handling below, and this must not straddle two
            try:
                # Published before the write, cleared after it, so the watchdog can tell "this
                # write has not come back" from "there is nothing to write". Ordering matters:
                # set it first, clear it in the finally, or a write that blocks forever would
                # never be visible to anyone.
                self._write_t = t_w
                proc.stdin.write(jpg)
                proc.stdin.flush()
            except (BrokenPipeError, ValueError, OSError):
                continue              # _feed's loop head owns the dead process and the restart
            finally:
                self._write_t = 0.0
            dt = time.time() - t_w
            self._c["blocked"] += dt
            self._c["worst_ms"] = max(self._c["worst_ms"], dt * 1000)
            # Split the write cost by keyframe position. Every frame written here becomes
            # exactly one encoded frame (duplicates included, skipped slots excluded), and
            # zerolatency's scenecut=0 makes IDRs strictly periodic -- so this frame is an
            # IDR exactly when its index is a multiple of -g. That makes the two buckets
            # directly comparable, and they answer a question `blocked%` alone cannot:
            #   idr_block >> p_block  => the keyframe burst is the wall. Cap the burst with
            #       VBV (stream_bitrate) rather than lengthening -g, which changes how often
            #       the spike happens but not how big it is.
            #   idr_block ~= p_block  => the wall is uniform and keyframes are innocent. This
            #       is what the deployment measured (43 vs 39 ms, 1.11x), which is what ruled
            #       out both bandwidth and the burst. Compare encode_ceiling() against
            #       max_stream_fps to tell a slow encoder from a slow link.
            if self._gop and self._c["frames"] % self._gop == 0:
                self._c["idr_n"] += 1
                self._c["idr_blocked"] += dt
                self._c["idr_worst"] = max(self._c["idr_worst"], dt * 1000)
            else:
                self._c["p_n"] += 1
                self._c["p_blocked"] += dt
            # Shape of the write cost, not just its mean -- the two are answered by different
            # fixes and a mean cannot tell them apart. A rate limit spreads writes smoothly in
            # proportion to bytes; a timer piles them onto one value. Measured: 0 of 2243 writes
            # under 5 ms, 80% in 25-60 ms, which says the cost is a fixed per-frame wait (delayed
            # ACK is 40 ms, which is why that band is drawn where it is) and no amount of
            # shrinking frames will move it. Now that the write is off the cadence thread these
            # numbers no longer bound the frame rate; they describe the link, and `ring_full`
            # is what says whether the link is too slow to keep up.
            ms = dt * 1000
            self._c["w_fast" if ms < 5 else "w_mid" if ms < 25 else
                    "w_timer" if ms < 60 else "w_over"] += 1
            self._c["frames"] += 1
            self._c["bytes"] += len(jpg)
            self._last_ok = time.time()
        try:
            self.proc.stdin.close()
        except Exception:                               # noqa: BLE001
            pass

    def alive(self):
        """Whether the child process exists. NOT whether the stream is reaching anyone -- a
        child whose socket writes block sits here as `True` indefinitely, which is what let a
        frozen output report healthy for a whole shift. `blocked_s` is the field that sees it.
        """
        return self.proc.poll() is None

    def blocked_s(self):
        """Seconds the in-flight stdin.write has been blocked, 0.0 when none is. The one field
        here that distinguishes a wedged output from an idle one; see _watch for why `stall`
        cannot. Reported to /health so the platform can act on it without reading the log.
        """
        started = self._write_t
        return (time.time() - started) if started else 0.0

    def snap(self):
        """Cumulative counters plus three instantaneous ones, for a periodic telemetry line.

        `stall` is time since the last successful write and is the reason a reader outside
        this object prints the line: while the writer thread is wedged inside stdin.write it
        emits nothing at all, so a self-printing output would go quiet exactly when it matters.
        """
        return dict(self._c, alive=self.alive(), restarts=self.restarts,
                    blocked_s=self.blocked_s(),
                    stall=(time.time() - self._last_ok) if self._last_ok else 0.0)

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


def _encoder_args(fps, bitrate):
    """Everything between the decoded frame and the H.264 bitstream, shared by all three
    outputs *and* by encode_ceiling() -- so the self-test measures the encoder the deployment
    actually runs, not an approximation of it. Keyframe interval is one second of stream time.
    """
    return [*LOW_LATENCY, *rate_limit(bitrate), "-g", str(int(max(1, fps)))]


def encode_ceiling(jpeg, fps, bitrate, frames=60, timeout=30.0):
    """Encode `frames` copies of one JPEG as fast as ffmpeg will take them, output discarded.

    Answers one question that no other number in the log can: when the feeder reports a high
    `blocked%`, is ffmpeg slow at *encoding*, or is it slow at *sending*? This run has the same
    decoder, filters and encoder as the live output but no muxer and no network, so its fps is
    the encode-side ceiling. Compare against max_stream_fps:

        ceiling >> max_stream_fps  => encoding is not the wall; blocked% is the output path
                                      (media server or link), i.e. outside this container
        ceiling ~ max_stream_fps   => encoding *is* the wall; look at CPU and frame size

    Costs one short burst of CPU, once, at startup. Returns 0.0 if it could not run at all --
    a diagnostic must never be the reason a stream fails to start.
    """
    exe = ffmpeg_path()
    if not exe or not jpeg:
        return 0.0
    cmd = [exe, "-hide_banner", "-loglevel", "error",
           "-f", "mjpeg", "-framerate", f"{float(fps):g}", "-i", "pipe:0",
           *_encoder_args(fps, bitrate), "-an", "-f", "null", "-"]
    proc = None
    try:
        # DEVNULL on both, deliberately: a piped stderr nobody drains wedges ffmpeg (see
        # _drain_err), and this helper has no drain thread.
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        t0 = time.time()
        for _ in range(frames):
            proc.stdin.write(jpeg)
        proc.stdin.close()
        proc.wait(timeout=timeout)
        dt = time.time() - t0
        return frames / dt if dt > 0 else 0.0
    except Exception:                               # noqa: BLE001
        return 0.0
    finally:
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except Exception:                       # noqa: BLE001
                pass


def flv_pipe(slot, fps, bitrate="4M"):
    """HTTP-FLV for one client: FLV bytes on stdout, to be streamed straight to the socket.

    One process per viewer. A shared encoder would be cheaper, but fanning FLV out to clients
    that join mid-stream means holding the header and replaying from the last keyframe, i.e.
    parsing FLV tags -- real complexity for a demo endpoint that serves a handful of viewers.
    `max_flv_clients` bounds the cost instead.
    """
    return FFmpegOut(slot, [*_encoder_args(fps, bitrate), "-an", "-f", "flv", "pipe:1"],
                     fps=fps, pipe_stdout=True, name="flv", gop=int(max(1, fps)))


def hls_writer(slot, fps, out_dir, bitrate="4M"):
    """Shared HLS segmenter. All viewers read the same segments off disk.

    1-second segments with a 6-segment window: HLS latency is roughly segments x duration, so
    this is ~3-6 s -- far worse than FLV and unavoidable for the format. delete_segments keeps
    the directory from growing without bound over a long run.
    """
    os.makedirs(out_dir, exist_ok=True)
    return FFmpegOut(slot, [*_encoder_args(fps, bitrate),
                            "-an", "-f", "hls", "-hls_time", "1", "-hls_list_size", "6",
                            "-hls_flags", "delete_segments+append_list+omit_endlist",
                            "-hls_segment_filename", os.path.join(out_dir, "seg_%05d.ts"),
                            os.path.join(out_dir, "index.m3u8")],
                     fps=fps, name="hls", gop=int(max(1, fps)))


def rtmp_push(slot, fps, url, bitrate="4M"):
    """Push to an external media server (rtmp:// or rtsp://) that fans out to its own clients.

    RTSP publishing is pinned to TCP. ffmpeg's rtsp muxer defaults to UDP, which needs the
    RTP/RTCP port pair to survive whatever sits between the container and the media server —
    and in a data centre that is usually a NAT and a firewall, so the ANNOUNCE succeeds and
    the media silently goes nowhere. TCP carries everything over the one connection that was
    already accepted. Both were verified against a real server here; TCP is the safer default.

    TCP then wants Nagle off, because the muxers here issue many sub-MSS writes per frame and
    that is the pattern Nagle withholds until the peer ACKs. ffmpeg leaves it ON
    (`-h protocol=tcp` shows `tcp_nodelay ... (default false)`). `-tcp_nodelay 1` reaches the
    socket for rtmp:// -- verified, the child logs `tcp://host:port?tcp_nodelay=1` -- but NOT
    for rtsp://, whose muxer builds its own tcp URL and forwards only `timeout` (verified the
    same way: `tcp://host:port?timeout=0`, no nodelay). So it is passed only where it takes
    effect; for RTSP the buffering in _spawn is what covers the same cost.

    What this actually costs a viewer, measured end-to-end against the deployment's server by
    timestamping FLV tag arrivals (20 fps, 1680x1020, 3 Mbps): the media clock is exact
    (media/wall 0.9999-1.0009 across four captures, and pts is CFR 50.0 ms +/-1 ms with zero
    discontinuities), so nothing is lost or drifting -- the whole defect is arrival JITTER, and
    it is what a low-latency player starves on. Pulling the RTSP relay and the HTTP-FLV
    endpoint SIMULTANEOUSLY gives statistically identical distributions (p50 49.4/49.3 ms,
    p90 64.2/63.9, p99 117.9/125.3, max 160/160, sigma 16.3/16.3, >95 ms starvations
    0.39/s both), so the media server is a pure relay and adds nothing measurable -- the
    jitter is already there when it arrives, i.e. it is ours or the network's.

    Beware of drawing conclusions from one short capture: the SAME stream at the SAME 20 fps,
    sampled over ~15 minutes, ranged sigma 16.3 -> 44.4 ms, worst gap 160 -> 446 ms, and
    starvation rate 0.39 -> 2.27/s. That spread is larger than any framerate effect measurable
    here, so it dwarfs (and explains) reports that one framerate streams cleanly while another
    stutters. Two mechanisms were tested against it and REFUTED: keyframe bursts (0/20 of the
    >100 ms gaps land within one frame of an IDR, 15% expected by chance; IDRs actually arrive
    FASTER, 38.3 vs 49.9 ms mean) and a ~1 s periodic stall from the recognition forward pass
    (looked periodic at 1.05 s over 40 s, but over 87 s the phase spread is 0.265 against 0.289
    for uniform -- i.e. not periodic at all).

    A socket I/O timeout is set here, and which option carries it depends on the scheme. Without
    one, a peer that stops reading wedges ffmpeg's write forever and the picture freezes until
    the container restarts -- the whole story is in PUSH_TIMEOUT_US. `-rw_timeout` is the generic
    AVIO option and is what the tcp/flv path takes; verified against a loopback sink that stops
    reading, where the child exits by itself with "Error writing trailer: Operation timed out".

    For rtsp it is `-timeout` instead, because that muxer builds its own tcp URL and forwards only
    that name -- the same forwarding quirk already measured above for `-tcp_nodelay`. But passing
    the right name is not enough, and this is the part worth knowing: against the same kind of
    sink (a fake RTSP server that completes ANNOUNCE/SETUP/RECORD and then stops reading), ffmpeg
    stayed blocked for 15.6 s and never timed out at all. So on the rtsp path -- the deployment's
    path -- this option buys nothing measurable, and _watch's kill is what actually recovers the
    stream. It is still passed: it costs nothing and may work on another build.

    Which is why the rtsp branch also shortens the watchdog (PUSH_STALL_KILL_RTSP): with no
    competing mechanism to defer to, waiting the default 15 s is 15 s of frozen picture.

    How to check whether ffmpeg's own timeout ever starts working here, since an option that
    silently does nothing is the norm for this path: `kills` on the `[push]` line. It climbing 1:1
    with `restarts` is today's expected reading for rtsp; `restarts` advancing while `kills` holds
    still would mean the child began honouring the option.
    """
    if url.startswith("rtsp://"):
        extra = ["-rtsp_transport", "tcp", "-timeout", str(PUSH_TIMEOUT_US)]
        stall_kill = PUSH_STALL_KILL_RTSP
    else:
        extra = ["-tcp_nodelay", "1", "-rw_timeout", str(PUSH_TIMEOUT_US)]
        stall_kill = PUSH_STALL_KILL_SEC
    fmt = "rtsp" if url.startswith("rtsp://") else "flv"
    # `warning`, unlike the other outputs: this is the one that leaves the container, so its
    # failures are the ones nobody can reproduce afterwards, and at `error` ffmpeg says
    # nothing about a connection that degrades without dying. _drain_err collapses repeats,
    # which is what made the louder level affordable.
    return FFmpegOut(slot, [*_encoder_args(fps, bitrate), "-an", *extra, "-f", fmt, url],
                     fps=fps, name="push", loglevel="warning", gop=int(max(1, fps)),
                     stall_kill=stall_kill)
