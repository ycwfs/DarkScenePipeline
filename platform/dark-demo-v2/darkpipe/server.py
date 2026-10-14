"""Online streaming-inference server (FastAPI).

CaptureThread: cv2.VideoCapture -> single-slot latest-frame buffer (newest overwrites =
implicit drop policy when the GPU is slower than the stream; bounded latency).
Then two concurrent threads, split by the resource they contend for, so throughput is their
max rather than their sum (see process_loop for the measurements):
  GpuThread     enhance -> recognizer.push (or -> reco_q, given a second device)
  OutputThread  SR -> label bar -> JPEG -> clip recorder
plus, when there is a spare device, a third that is deliberately NOT on the frame path:
  RecoThread    recognizer.push on a device of its own; events return via ev_q
asyncio endpoints only read the JPEG slot / subscribe to the event bus:
  GET /stream  multipart MJPEG   GET /events  SSE recognition JSON
  GET /health  live counters     GET /config  active configuration
  GET /live.flv  HTTP-FLV        GET /hls/index.m3u8  HLS  (both via ffmpeg, see streams.py)

Every video format is fed from the same JPEG slot, so adding one costs a mux, not a second
encode of the pipeline output.

With `cfg.clip_dir` set, the same processed frames are also fed to a ClipRecorder, which
writes an mp4 per non-`other` behaviour (darkpipe/clips.py). It runs on its own thread so
the GPU loop keeps its latency budget.
"""
import asyncio
import json
import os
import queue
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict

import cv2

from .media import open_capture
from .render import append_label_bar
from .stages import build_stages
from .utils import bind_thread_device


class LatestSlot:
    def __init__(self):
        self._lock = threading.Lock()
        self.item = None
        self.seq = 0

    def put(self, item):
        with self._lock:
            self.item = item
            self.seq += 1

    def get(self):
        with self._lock:
            return self.item, self.seq


class _Stat:
    """Cumulative time spent in one stage.

    Exactly one thread calls `add`; readers snapshot and diff against their own previous
    snapshot. That keeps it lock-free and, unlike a counter the reader resets, loses no
    sample to a race between the writer and the 2 s telemetry window.
    """
    __slots__ = ("sum", "n")

    def __init__(self):
        self.sum, self.n = 0.0, 0

    def add(self, dt):
        self.sum += dt
        self.n += 1

    def snap(self):
        return (self.sum, self.n)


def _mean_ms(now, before):
    """Mean ms per call between two `_Stat.snap()` results; 0 if nothing ran."""
    ds, dn = now[0] - before[0], now[1] - before[1]
    return ds / dn * 1000 if dn else 0.0


def _duty(now, before, dt_win):
    """Share of a telemetry window a stage was busy, as a percentage of wall clock.

    The companion to _mean_ms and the more useful of the two on a multi-device box: a mean says
    how long one call took, a duty says whether the card running those calls has anything left
    to give. `enhance=46ms` alone cannot distinguish a saturated card from an idle one.
    """
    return (now[0] - before[0]) / dt_win * 100 if dt_win > 0 else 0.0


def _sum_snaps(snaps):
    """Combine several `_Stat.snap()` results into one, for an all-devices aggregate."""
    return (sum(s[0] for s in snaps), sum(s[1] for s in snaps))


class EventBus:
    def __init__(self):
        self._lock = threading.Lock()
        self._subs = []

    def subscribe(self):
        q = queue.Queue(maxsize=64)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, ev):
        with self._lock:
            for q in self._subs:
                try:
                    q.put_nowait(ev)
                except queue.Full:
                    try:
                        q.get_nowait()
                        q.put_nowait(ev)
                    except queue.Empty:
                        pass


class _FlvViewer:
    """One HTTP-FLV connection, from the GET to the disconnect.

    Holds the previous telemetry snapshot alongside the output, because a viewer is not in
    `p_out`: that dict is keyed by a name that outlives the window ("push", "hls"), and reusing
    it here would make viewer #7's first window diff against viewer #3's last one and print
    garbage. Each viewer carries its own baseline instead.
    """

    __slots__ = ("seq", "ip", "out", "started", "prev", "sent", "prev_sent")

    def __init__(self, seq, ip, out):
        self.seq = seq
        self.ip = ip
        self.out = out
        self.started = time.time()
        self.prev = None                # previous out.snap(), set by the telemetry loop
        # Bytes handed to the HTTP response, i.e. H.264 leaving for this client -- the one
        # per-peer delivery counter in this server. The shared outputs cannot have it: their
        # bytes go into an ffmpeg that owns the socket, and `-progress total_size` counts what
        # was handed to the muxer rather than what the peer took (see the note above
        # streams.PIPE_BYTES), so it advances right through a stalled peer. Here the chunk is
        # yielded into the ASGI response, so a client that stops reading stops this counter via
        # TCP backpressure. Not a delivery receipt -- a few hundred KB may still be in flight --
        # but it is the closest thing available and it does go flat when the peer does.
        self.sent = 0
        self.prev_sent = 0

    def age(self):
        return time.time() - self.started


class ServerState:
    def __init__(self, cfg, on_clip=None):
        self.cfg = cfg
        self.raw = LatestSlot()
        self.jpeg = LatestSlot()
        self.bus = EventBus()
        self.stop = threading.Event()
        self.t_start = time.time()
        self.reconnects = 0
        self.frames_in = 0
        self.frames_proc = 0
        self.fps_in = 0.0
        self.fps_proc = 0.0
        self.latency_ms = 0.0
        self.capture_alive = False
        # Frame rate the source itself reports, 0.0 when it reports nothing. Published by
        # capture_loop because that is the only place the stream is open, and consumed by
        # stream_fps_for_source before any output ffmpeg spawns -- see the wait in lifespan.
        self.src_fps = 0.0
        self.src_fps_known = threading.Event()
        self.last_event = None
        self.events_total = 0
        self.on_clip = on_clip
        # Called exactly once, from output_loop, when the first frame has made it all the way
        # through. Startup watchers hang off this: it is the only signal that means "weights
        # loaded, capture open, pipeline producing" rather than merely "the process is alive".
        self.on_ready = None
        self.clipper = None
        self.eventlog = None
        self.formats = ["mjpeg"]
        self.hls = None                 # shared HLS segmenter
        self.hls_dir = ""
        self.push = None                # shared RTMP/RTSP push
        self.flv_clients = 0
        self.flv_lock = threading.Lock()
        # Per-viewer FLV bookkeeping. The shared outputs (push, hls) are single long-lived
        # objects the telemetry loop can diff against a fixed key; an FLV viewer is neither --
        # it appears on a GET and is gone on disconnect, so it needs its own registry.
        #
        # This exists because "how many times did the player reconnect?" was unanswerable from
        # the log: the [push]/[hls] telemetry loop never looked at flv at all, uvicorn's access
        # log is off (see the uvicorn.Config call), and /health carried a live count with no
        # history behind it -- a viewer could connect and drop fifty times and every field here
        # would read exactly as it does when nobody has ever connected. `flv_total` is the field
        # that answers it; `flv_clients` alone cannot, and reading it as if it could is the
        # mistake this pair is here to prevent.
        self.flv_total = 0              # connections accepted, ever
        self.flv_rejected = 0           # connections refused by max_flv_clients
        self.flv_seq = 0                # per-connection id, for correlating the two log lines
        self.flv_live = {}              # seq -> _FlvViewer, guarded by flv_lock


_FINITE_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm")


def _is_finite_source(src):
    """Is this a file that ends, as opposed to a live stream that does not?

    The old test was "no :// in the address", which classified an `http://.../x.mp4` as a live
    stream: EOF then raised instead of looping, and the reconnect backoff reprinted the same
    error every 8 s forever. The platform operator now downloads such addresses before getting
    here, so this is belt-and-braces -- but the CLI still accepts a URL directly.
    """
    s = str(src)
    if "://" not in s:
        return not s.isdigit()          # a bare number is a camera index
    from urllib.parse import urlsplit
    return urlsplit(s).path.lower().endswith(_FINITE_SUFFIXES)


def capture_loop(st: ServerState):
    backoff = 0.5
    while not st.stop.is_set():
        try:
            # Both of these lines exist because open_capture on an RTSP URL has no timeout and
            # can block indefinitely inside FFmpeg. Without them a source that never answers is
            # indistinguishable from a hung operator: nothing else on this path prints until a
            # frame arrives, so the log simply stopped. Print BEFORE the call, not after.
            print(f"[capture] 正在打开输入 {st.cfg.input}")
            cap = open_capture(st.cfg.input)
            st.capture_alive = True
            backoff = 0.5
            is_file = _is_finite_source(st.cfg.input)
            # Two variables on purpose. `reported_fps` is 0.0 when the source said nothing;
            # `src_fps` carries the 25.0 assumption used to pace file playback. Only the former
            # reaches stream_fps_for_source: calibrating the output cadence against a number we
            # invented ourselves would snap to a ratio the camera never had.
            reported_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            src_fps = reported_fps or 25.0
            st.src_fps = reported_fps
            st.src_fps_known.set()
            print(f"[capture] 已打开，源 fps={src_fps:.2f}"
                  f"{'（有限长文件，将循环播放）' if is_file else ''}")
            t_last = time.time()
            n, t_win = 0, time.time()
            while not st.stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    if is_file:  # loop files for demo purposes
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    raise RuntimeError("stream read failed")
                st.raw.put((frame, time.time()))
                st.frames_in += 1
                n += 1
                if time.time() - t_win >= 2.0:
                    st.fps_in = n / (time.time() - t_win)
                    n, t_win = 0, time.time()
                if is_file:  # pace file playback at source fps
                    dt = 1.0 / src_fps - (time.time() - t_last)
                    if dt > 0:
                        time.sleep(dt)
                    t_last = time.time()
            cap.release()
        except Exception as e:
            st.capture_alive = False
            if st.stop.is_set():
                break
            print(f"[capture] {e}; reconnecting in {backoff:.1f}s")
            st.stop.wait(backoff)
            st.reconnects += 1
            backoff = min(backoff * 2, 8.0)
    st.capture_alive = False


# Per-frame over-carried motion the eye starts to read as judder. Bracketed by deployment, not
# chosen: on one 25.00 fps camera, 15 fps out (13.3 ms) streamed cleanly and 17 fps out
# (21.2 ms) stuttered. 16.0 sits in that gap, nearer the value proven good.
JUDDER_BUDGET_MS = 16.0


def _judder_ms(out_fps, src_fps):
    """Motion a rounded-up output frame carries in excess of its display duration, in ms.

    The output cadence advances the *content* by an integer number of source frames, while
    `-framerate` stamps every output frame exactly 1/out_fps apart. So with r = src/out each
    delivered frame carries floor(r) or ceil(r) source frames of motion, never r, and the
    over-carry of a rounded-up one is (ceil(r) - r) source-frame intervals. Identically zero
    when r is an integer -- that is the whole lever this module pulls.

    Measured directly rather than derived: writing JPEGs into an ffmpeg stdin at 45/44/41/84 ms
    (the 25->20 fps pattern) came back out with pts at exactly 50.0/50.0/50.0/50.0 ms. The input
    timing is discarded wholesale, which is what makes the mismatch a property of the ratio and
    not of our scheduling.

    Blind to how OFTEN the rounded-up frame occurs, only to how much it over-carries: r = 1.00004
    scores the same ~40 ms as r = 1.5 even though it over-carries once in 25000 frames. Harmless
    in the only place it matters -- when frac -> 0 the nearest candidate in
    stream_fps_for_source IS the adjacent exact divisor, so the snap it triggers moves the
    cadence by <=0.1 fps. Adding the frequency term would change no outcome and would cost the
    one-line reading of this function.
    """
    r = float(src_fps) / max(1e-6, float(out_fps))
    frac = r - int(r)
    return 0.0 if frac < 1e-6 else (1.0 - frac) * 1000.0 / float(src_fps)


def stream_fps_for_source(want, src_fps, budget_ms=JUDDER_BUDGET_MS):
    """Pick an output cadence that does not judder against a `src_fps` source.

    -> (fps, note). `note` always explains the outcome and is meant for the operator's log;
    compare the returned fps with `want` to tell "corrected" from "kept as asked".

    Why this exists. Five deployment runs on one 25.00 fps camera, everything but
    max_stream_fps equal, gave 12.5 / 15 / 25 fps clean and 17 / 20 fps stuttering. Nothing
    about delivery separates those two sets -- dup%, missed%, blocked%, write-block, pipe
    bitrate, latency and restarts all interleave {17,20} inside {12.5,15,25}, and 12.5
    discards the most frames of the five while streaming fine. What separates them is
    _judder_ms: 0.0 / 13.3 / 21.2 / 30.0 / 0.0 ms, the one column that splits them. It also
    explains why the bad set is not an interval: the penalty rises with out_fps inside
    r in (1,2), peaks at r = sqrt(2) (17.7 fps on a 25 fps source), then drops
    discontinuously to zero at r = 1. Hence "snap to src_fps/k", not "cap the frame rate".

    Choosing among the candidates src/1, src/2, ... : nearest to `want` in value, ties to the
    higher rate. On a 25 fps source that maps 20 -> 25 and 17 -> 12.5, both of them cadences
    this camera was observed to stream cleanly at.

    Snapping UP is free here, measured on the deployment's own stream_bitrate=3M: output bitrate
    is 3.32 Mbps at 25 fps and 3.41 at 20 (102.7% -- *higher*, not lower), encode wall clock
    within 2%, because -maxrate is a hard ceiling the rate controller spends regardless of frame
    count, so fewer frames just means bigger ones. It also saves no JPEG work at all: the
    pipeline encodes every frame it produces (~24.7 fps) for /stream and the clips no matter
    what the output cadence is, so a lower cadence only discards more of them -- 21.2% at 20 fps
    against 7.8% at 25. Frame rate only buys bitrate with the cap off (12.5/15/20/25 fps ->
    63/75/89/100%), and there 20 saves 11% while judder-free 12.5 saves 37%; asking for 20 to
    economise is the worst of both.

    Snapping up is NOT measured on a pipeline slower than the source. A single-GPU box at
    proc_max_side 720 produces ~15.6 fps, so `want`=20 would snap to 25 and roughly 38% of
    delivered frames would be repeats. `dup` in the [push] line is the signal. Preferring the
    lower candidate instead is a one-line change (restrict the range to k >= src/want), held
    back deliberately: it should be the operator asking for 12.5, not us guessing they meant to
    economise.

    `src_fps` is what the camera *reports*, which is not what it delivers -- this one reports
    25.00 and puts ~24.7 fps into the slot, so r is computed against a slightly wrong
    denominator and k=1/k=2 keep a residual dup/missed instead of landing on exactly 1.0/2.0.
    Left alone: both snap targets on this camera (25 and 12.5) are precisely the two cadences
    measured clean *at* that 24.7 delivery rate, so the reported figure is the one that
    predicted the outcome correctly.

    Deliberately does nothing when the source reports no usable rate. A wrong r is worse than
    an uncalibrated one, and `cv2` hands back 0 on plenty of RTSP cameras and five-digit
    nonsense on a few.

    Not fixable by asking for exact-20-and-smooth: 20 does not divide 25, so a strictly CFR
    20 fps stream can only be made smooth by SYNTHESISING the missing frame positions, and that
    is measured out -- minterpolate at this frame geometry runs 0.54x realtime (mci+obmc) to
    0.29x (mci+aobmc+bidir) on 8 idle cores, against a container that pins x264 to 4 threads and
    owes a >=15 fps recognition pipeline. mi_mode=blend is free but trades judder for motion
    ghosting, which is worse on footage whose whole purpose is judging human motion. The general
    alternative is VFR output timestamps (verified workable on ffmpeg 8.1.2 for both FLV and
    Matroska), which removes the term at any cadence but yields ~24.7 fps VFR, not 20.
    """
    want = max(1.0, float(want))
    src = float(src_fps or 0.0)
    if not 5.0 <= src <= 120.0:
        return want, (f"源未报告可用帧率（{src:g}），出流节拍未校准；若画面有规律抽动，"
                      f"手动把 max_stream_fps 设成源帧率的整数分之一")
    if want > src * 1.001:
        return src, f"源只有 {src:.2f} fps，更高的节拍只能重发上一帧，无非多费一次编码"
    j = _judder_ms(want, src)
    if j <= budget_ms:
        return want, (f"对源 {src:.2f} fps 每帧最多多带 {j:.1f} ms 运动，"
                      f"在 {budget_ms:g} ms 判据内")
    k = min(range(1, int(src) + 1), key=lambda i: (abs(src / i - want), i))
    pick = "与源同帧率" if k == 1 else f"取源 {src:.2f} fps 的 1/{k}"
    fps = src / k
    # "You raised my frame rate, am I now paying for it in bandwidth" is certain to be the next
    # question, so the answer ships in the line itself instead of waiting to be looked up.
    cost = ("；带宽不变，-maxrate 是硬上限（参考实测：3M 上限下 20 与 25 fps 同为 3.3~3.4 Mbps）"
            if fps > want else "")
    return fps, (f"{pick}；原值 {want:g} fps 每帧多带 {j:.1f} ms 运动"
                 f"（超过 {budget_ms:g} ms 判据），会看成规律抽动{cost}")


def serve_devices(cfg):
    """The devices enhancement round-robins over, in dealing order.

    `--gpus` means two different things by mode and both are legitimate: offline splits a
    file into frame ranges (one segment per GPU, needs future frames), serve deals live
    frames round-robin (no future frames needed). Same flag, same "use these GPUs" intent,
    different mechanism -- see config.validate.

    A single-id `--gpus` falls back to `--device`, which is what offline already does
    (cli.py only shards when len(gpus) > 1). Diverging here would mean `--gpus 3` picked a
    different card in serve than in offline, so config.validate warns about that spelling
    in both modes instead.
    """
    if getattr(cfg, "gpus", ""):
        ids = [g.strip() for g in cfg.gpus.split(",") if g.strip()]
        if len(ids) > 1:
            return [f"cuda:{g}" for g in ids]
    return [cfg.device]


def process_loop(st: ServerState):
    """Load the models, then run the pipeline as two concurrent threads.

    The four stages used to run back to back on one thread, so throughput was their SUM.
    They split by the resource they contend for: enhance and recognition are GPU, SR and
    JPEG encoding are CPU. Both torch's CUDA calls and OpenCV release the GIL, so the two
    sides genuinely overlap and throughput becomes the max of them instead of the sum.
    Measured on 720x404 frames (200-frame runs, alternated 3x to cancel machine drift,
    medians below; this is a shared GPU, so single runs vary by ~1 fps and ~20 ms):

        serial     13.5 fps   latency p50 74 ms  p95 119 ms  max 176 ms
        this split 16.5 fps   latency p50 89 ms  p95 140 ms  max 198 ms

    So the split buys throughput and *costs* latency -- it is not free on both axes. Two
    effects, both real: a frame now waits to be handed over, and the CPU-side stages run
    about half as fast under contention (sr 2 -> 7 ms, encode 11 -> 22 ms; measured, and not
    OpenCV thread oversubscription -- sweeping cv2.setNumThreads(0/8/4/2) left encode at
    22-23 ms throughout). That extra cost is hidden from throughput because the CPU side
    (~29 ms) still finishes well inside the GPU side (~62 ms), but the frame carrying it
    pays in latency.

    The trade is worth it here only because of which budget is binding: the delivery spec is
    >=15 fps and <=1 s of delay. Serial misses the fps bar; this clears it. The ~20 ms of
    added latency is spent against a 1000 ms budget that no run comes close to using. On a
    latency-bound deployment the serial loop would be the better choice.

    Recognition stays on the GPU thread WHEN THERE IS ONLY ONE DEVICE, and that is deliberate:
    it was tried on a third thread of its own, to keep its bursty cost (~2 ms to buffer a frame,
    then a full window inference every `stride` frames) off the frame path entirely. It measured
    WORSE than this on both axes -- 15.2 fps, p95 159 ms, max 403 ms. Two Python threads issuing
    CUDA work time-slice against each other on one device: the inference inflated 73 ms -> 250 ms
    and dragged enhance from 55 ms to 62 ms. The stall was not removed, just smeared across every
    frame that overlapped it, which is what wrecked the tail. Keeping all GPU work on one thread
    serialises it by construction and costs nothing, because the GPU is the bottleneck either way
    (see wait_gpu/wait_cpu in the telemetry below).

    That argument is about sharing a DEVICE, not about the thread. Given a second card the
    conclusion inverts -- see the multi-GPU section, where recognition does get its own thread
    because it also gets its own device.

    Ordering: the single hop is single-producer/single-consumer, so the clip recorder still
    sees frames in capture order with no sequence bookkeeping -- the same argument the
    offline runner relies on (pipeline.py).

    MORE THAN ONE GPU (`--gpus 0,1`) gives the LAST device to recognition alone and deals frames
    round-robin across the rest, adding a recognition thread, and -- only from three devices up --
    a dealer thread plus one enhance thread per extra enhance device. It is a separate path,
    entered only when serve_devices returns more than one device, and the two-thread loop above is
    left untouched -- that path is what was measured and shipped, and it is still the production
    path wherever the platform grants one card.

    The reason enhance scaling has to be more cards rather than more threads: enhance at 720x404
    is 1.9 ms of upload, 35.8 ms of GPU compute, 3.7 ms of download, so 86% of it is the GPU
    itself and only the other 14% can overlap on one device. Measured, dealing to two
    instances (`/tmp/multigpu_bench.py`, 23.7 fps for one instance as the baseline):

        2 instances on one card, default stream       25.2 fps   1.06x
        2 instances on one card, separate streams     27.1 fps   1.14x  <- at the 1/0.86 cap
        2 instances on two cards                      42.3 fps   1.78x

    Thread layout, and why it is this and not something simpler:

        dealer          pure CPU: newest frame from st.raw -> in_q[k], strictly rotating
                        (only with >=2 enhance devices, i.e. >=3 devices in total)
        gpu_worker[k>0] enh_devices[k]: enhance only -> done_q[k]
        gpu_worker[0]   enh_devices[0]: enhance + emitting in order
        reco_loop       reco_dev: nothing but recognition, fed by reco_q
        output_loop     SR -> label bar -> JPEG -> clipper (unchanged, still one thread)

    Every device carries exactly one CUDA-submitting thread, which is the invariant the
    measurements above depend on. Worker 0 owns the ordering because that loop already walks the
    output sequence, so with strict rotation `next_out % n_gpu` names the device owning the next
    frame, and out-of-order completion is absorbed by a blocking get on done_q[k]. No reorder
    buffer, no sequence numbers.

    Recognition is the one stage NOT on the frame path, so its 242 ms bursts delay only the
    event -- by up to one stride -- and never the stream. See the comment above `reco_dev` for
    the deployment measurement that moved it there and what it costs.

    End to end through the round-robin, two RTX 3090s vs one, same frames, alternated 3x for
    medians (a source offered faster than capacity, so the queues stay full -- this is the
    saturated case, not an idle one). NOTE: these were taken when both cards enhanced and
    recognition ran inline on worker 0, so they now describe a >=3-device box, not the 2-card
    deployment -- there, one card enhances and the second recognises:

        proc_max_side=840   1 gpu  12.9 fps  p50 109 ms  p95 169 ms  max 286 ms
                            2 gpu  23.8 fps  p50 197 ms  p95 289 ms  max 336 ms   1.85x
        proc_max_side=960   1 gpu  10.1 fps  p50 138 ms  p95 192 ms  max 258 ms
                            2 gpu  18.0 fps  p50 260 ms  p95 350 ms  max 392 ms   1.79x

    Latency roughly DOUBLES, which is the opposite of free and worth being clear about. It
    is not that a frame got slower -- each one still runs end to end on a single card at
    unchanged speed. It is that the pipeline got deeper: one device holds about two frames
    (one enhancing, one in out_q), two devices hold about six (two enhancing, two prefetched
    in in_q, one in done_q, one in out_q). Frames that sit in those extra slots age. At 840
    that is ~4.7 frame periods in flight against ~1.4 -- which is exactly the ratio between
    the two p50s, so there is nothing else going on.

    So the round-robin spends latency to buy rate, the same trade as the two-thread split above
    and for the same reason: the binding budget is >=15 fps and <=1 s, and at 840 the result is
    23.8 fps against a 289 ms p95 -- 59% over the fps bar with 3.4x of headroom on delay.
    Tightening the prefetch would give some of that latency back at a cost in throughput; it
    is not worth doing while the delay budget is this slack.
    """
    cfg = st.cfg
    devices = serve_devices(cfg)
    # The LAST device is recognition's alone; enhancement gets the rest. With the two cards the
    # platform actually grants, that means one enhance device and no round-robin at all.
    #
    # Recognition used to be folded into enhance worker 0 and run inline, which cost the whole
    # output path 242 ms every time it fired. Measured in deployment (80 telemetry windows):
    # inference 242 ms at 1.16/s = 27.9% duty on cuda:0, against 23.4% of the window spent
    # waiting on a peer (the `peer` share now printed inside the per-card `gpuK=` group) -- the
    # other card idle in put_until(done_q) for almost exactly the time cuda:0 spent inferring. The
    # visible effect was a two-state stream: 48 windows at 74 ms latency and 33 at 327 ms,
    # alternating (65 direction changes in 79 comparisons), with 20% of pushed frames being
    # duplicates because out_q had nothing new during the stall.
    #
    # Giving recognition its own card keeps the invariant that produced the round-robin numbers
    # below -- exactly one CUDA-submitting thread per device. The two rejected alternatives both
    # break it: making recognition async but leaving it on cuda:0, or keeping both cards on
    # enhance and putting recognition on cuda:1, each puts a second submitting thread on some
    # card, which is the configuration measured at 73 ms -> 250 ms inference.
    #
    # It costs enhance throughput: one card instead of two, 42.3 fps -> ~21.5 fps at
    # proc_max_side=840 (46.5 ms/frame measured). The binding requirement is >=15 fps out, and
    # the source offers 25, so this is spending headroom that was not being converted into
    # smoothness anyway. It does not affect the 1080p output: that comes from sr=bicubic_x2 on
    # top of proc_max_side=840, so enhance cost does not move with output resolution -- only
    # raising proc_max_side would, and that is when this trade needs re-measuring.
    reco_dev = devices[-1] if len(devices) > 1 else devices[0]
    enh_devices = devices[:-1] if len(devices) > 1 else devices
    n_gpu = len(enh_devices)
    # Recognition off the enhance thread entirely, whenever it has a card to itself.
    reco_async = reco_dev not in enh_devices
    # One independent set of enhance stages per enhance device. Everything after recognition
    # stays single-instance on enh_devices[0]: the output thread is single and CPU-bound, so a
    # second copy would only cost memory. build_stages hands back a recognizer each time too,
    # but only one is ever load()ed -- the extras cost a constructor and no weights.
    enh_sets, srs, recognizer, frame_stages = [], [], None, []
    for i, dev in enumerate(enh_devices):
        stages, rec = build_stages(cfg)
        # Split on the stage's own declaration, not on its name. Recognition sees the
        # enhanced pre-SR frame; anything flagged post_recognition only changes the picture.
        enh_sets.append([s for s in stages if not getattr(s, "post_recognition", False)])
        for s in enh_sets[i]:
            s.load(dev)
        frame_stages.extend(enh_sets[i])
        if i == 0:
            srs = [s for s in stages if getattr(s, "post_recognition", False)]
            for s in srs:
                s.load(dev)
            frame_stages.extend(srs)
            recognizer = rec
    if recognizer:
        recognizer.load(reco_dev)
        t0 = time.time()
        recognizer.warmup()
        print(f"[serve] 识别模型在 {reco_dev} 上预热完成 "
              f"{(time.time() - t0) * 1000:.0f}ms")
    print(f"[serve] 增强 {', '.join(enh_devices)}"
          f"{'（按帧轮询）' if n_gpu > 1 else ''}；"
          f"识别 {reco_dev}{'（独立线程）' if reco_async else '（与增强同线程）'}")
    enh = enh_sets[0]

    # One slot, because the encoder is the only consumer and one frame in flight is all the
    # overlap the split needs: while the encoder works on frame N the GPU builds N+1. A
    # deeper queue would only buy the right to run further ahead, which costs latency and
    # GPU time the recognizer needs.
    out_q = queue.Queue(maxsize=1)
    # Set by the encoder whenever it takes a frame, waited on by the GPU thread. There is no
    # lost wakeup to worry about despite the clear-then-fill gap below: the encoder is the
    # only thread that sets it and the only one that frees a slot, so a set that lands while
    # the GPU is mid-enhance simply means the wait after the put returns immediately.
    space = threading.Event()
    space.set()                       # the queue starts empty
    s_rec, s_fire, s_sr, s_enc = _Stat(), _Stat(), _Stat(), _Stat()
    # Seconds each side spent waiting on the other. Together they say which side is the
    # bottleneck -- the one thing the per-stage means stop being able to tell you once the
    # stages overlap. See the telemetry line below for how to read them.
    wait = {"gpu": 0.0, "cpu": 0.0}
    # Enhance timing and peer waiting, one entry per enhance device. _Stat is single-writer by
    # construction (see its docstring) and so is a float, but with n_gpu > 1 the enhancing
    # happens on n_gpu different threads -- gpu_worker(k) for k>0 and the sequencer for 0 -- so
    # a shared accumulator loses samples to a non-atomic `+=`. Latent today, because the
    # two-card deployment gives enhance one device and leaves n_gpu at 1; live the moment
    # enhance is dealt across cards. Per device rather than summed for a second reason: which
    # card is starved is the answer you want, and a sum only says that someone was.
    s_gpu = [_Stat() for _ in enh_devices]
    w_peer = [0.0] * n_gpu
    # Enhanced pre-SR frames on their way to the recognition card, and the events on their way
    # back. Depth 8 is generous on purpose: recognition consumes at the frame rate but only
    # spends 242 ms per 16 frames (27.9% duty measured), so this should never fill. If it does,
    # `reco_dropped` says so rather than the window silently sampling coarser -- and coarser is
    # all it would be, because span mode resamples the window by timestamp (`_window_frames`),
    # so a missing frame degrades the sampling instead of corrupting the window.
    reco_q = queue.Queue(maxsize=8)
    ev_q = queue.Queue(maxsize=8)
    reco_dropped = [0]
    done = threading.Event()          # the pipeline is unwinding, for any reason

    def running():
        return not (st.stop.is_set() or done.is_set())

    def worker(fn):
        """Run a loop on its own thread; a crash takes the whole pipeline down with it.

        Splitting one thread into three would otherwise invent a partial-failure state that
        never existed: a dead output thread leaves capture and enhance running, /health
        reporting `ok`, and the stream frozen on its last JPEG. Failing together keeps the
        old all-or-nothing behaviour, and `done` unblocks the other two loops.
        """
        def run():
            try:
                fn()
            except Exception as e:                      # noqa: BLE001 - logged, then fatal
                import traceback
                print(f"[process] {fn.__name__} died: {e}")
                traceback.print_exc()
            finally:
                done.set()
        return threading.Thread(target=run, daemon=True, name=f"darkpipe-{fn.__name__}")

    rec_idx = [0]

    def recognize(frame, t_cap, raw_frame=None):
        """Buffer a frame; every `stride` frames this also fires the window inference.

        `raw_frame` is the same moment's pre-enhancement frame, for the recognizer's
        illuminance gate (enhanced luma is whitened -- see stages.recognize.frame_luma)."""
        t0 = time.time()
        # Capture time, not "now": the recognizer's window is defined in wall-clock seconds
        # (`span_sec`), so feeding it processing time would stretch the window by however far
        # behind the source this thread is running.
        ev = recognizer.push(frame, rec_idx[0], t_cap - st.t_start, raw_frame=raw_frame)
        dt = time.time() - t0
        s_rec.add(dt)
        rec_idx[0] += 1
        if ev:
            s_fire.add(dt)            # only the frames that actually ran the model
            st.last_event = ev
            st.events_total += 1
            st.bus.publish(ev)
            if st.eventlog is not None:
                st.eventlog.write(ev)
        return ev

    def hand_to_recognizer(frame, t_cap, raw_frame=None):
        """Recognize this frame, or queue it for the card that does. Returns the event or None.

        Returning None in the async case is not a loss of information: the label bar reads
        st.last_event (which `recognize` sets from whichever thread ran it) and the clipper is
        served by ev_q instead. Only the clipper needs each event exactly once, which is what
        the queue guarantees and a shared "latest event" variable would not.

        Sharing the frame with another thread is safe because nothing downstream mutates it in
        place: the recognizer preprocesses into a new array on entry (`_preprocess`), the SR
        stage returns a new array, and append_label_bar returns an np.vstack. That includes
        `raw_frame`, which only the recognizer reads (a reference, not a copy).
        """
        if not recognizer:
            return None
        if not reco_async:
            return recognize(frame, t_cap, raw_frame)
        try:
            reco_q.put_nowait((frame, t_cap, raw_frame))
        except queue.Full:
            reco_dropped[0] += 1
        return None

    def reco_loop():
        """The recognition device: nothing but recognition, on a thread of its own.

        This is the one loop in the service that is allowed to be slow. It is off the frame
        path entirely, so a 242 ms inference now delays only the event by up to one stride --
        it no longer stops frames from reaching the encoder. Events were always about a window
        already in the past (which is why ClipRecorder keeps a pre-roll ring), so arriving one
        inference late changes nothing a viewer or a clip can see.
        """
        bind_thread_device(reco_dev)
        while running():
            try:
                frame, t_cap, raw_frame = reco_q.get(timeout=0.1)
            except queue.Empty:
                continue
            ev = recognize(frame, t_cap, raw_frame)
            if ev:
                try:
                    ev_q.put_nowait(ev)
                except queue.Full:
                    # The output thread runs at frame rate and events arrive ~1/s, so a full
                    # queue means output is wedged; the clip would be wrong either way.
                    print("[warn] 事件队列已满，本次事件未送达片段录制")

    def output_loop():
        """CPU: SR -> label bar -> JPEG. Owns the clip recorder and the end-to-end counters."""
        bind_thread_device(enh_devices[0])
        recorder = None
        n, t_win = 0, time.time()
        p_rec, p_fire, p_sr, p_enc = (s.snap() for s in (s_rec, s_fire, s_sr, s_enc))
        p_gpu = [s.snap() for s in s_gpu]
        p_wait = dict(wait)
        p_peer = list(w_peer)
        p_parts = recognizer.timings() if recognizer else None
        p_out = {}
        # {} rather than None: with None the first comparison below is `0 != None`, which fires
        # a [clip] line on the first window of every healthy run.
        p_clip = {}
        t_idle = time.time()
        while running():
            t_wait = time.time()
            try:
                frame, t_cap, ev = out_q.get(timeout=0.1)
            except queue.Empty:
                wait["gpu"] += time.time() - t_wait
                # Say something while starved. The telemetry below is inside the success path,
                # so before the first frame this thread used to print NOTHING, for as long as it
                # took -- and a source that never delivers then looks exactly like a hung
                # operator. These counters name the layer that is stuck: frames_in=0 means the
                # capture side never got a frame (look at [capture]); frames_in climbing with
                # fps=0 means the enhance thread is stuck on the GPU.
                if time.time() - t_idle >= 5.0:
                    t_idle = time.time()
                    print(f"[process] 等待帧… 已处理={st.frames_proc} 采集={st.frames_in} "
                          f"采集存活={int(st.capture_alive)} 重连={st.reconnects}"
                          + (f" reco_q={reco_q.qsize()}" if reco_async else ""))
                continue
            t_idle = time.time()
            space.set()
            wait["gpu"] += time.time() - t_wait
            t0 = time.time()
            chunk = [frame]
            for s in srs:
                chunk = s(chunk)
            t1 = time.time()
            out = chunk[0]
            if recognizer and not cfg.no_label_bar:
                # The configured stream rate, not the measured processing rate. These are
                # different quantities and the burnt-in one should be the viewer's: the feeder
                # resamples the JPEG slot onto a fixed max_stream_fps cadence (repeating the
                # last frame when the pipeline is slower), so that is genuinely the rate the
                # stream is delivered at. The processing rate stays reported, but in /health
                # as fps_proc, where it is a diagnostic rather than a number on a wall.
                out = append_label_bar(out, st.last_event,
                                       extra=f"{cfg.max_stream_fps:g} fps")
            ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
            if ok:
                st.jpeg.put(buf.tobytes())
            t2 = time.time()
            s_sr.add(t1 - t0)
            s_enc.add(t2 - t1)
            if reco_async:
                # Recognition runs on its own device now, so the event cannot travel with the
                # frame that produced it; it arrives here up to one stride late. Taken exactly
                # once, because the recorder decides clip boundaries by counting consecutive
                # events (`switch_after`) and a repeated event would forge a confirmation.
                try:
                    ev = ev_q.get_nowait()
                except queue.Empty:
                    ev = None
            # In the single-device path `ev` travelled with the frame that produced it rather
            # than in a side channel, so the recorder gets the exact pairing the serial loop
            # gave it. The frame handed to the recorder is the one the demo stream shows --
            # label bar burned in -- because these clips are for people to watch, not to
            # re-analyse.
            if st.clipper is not None:
                st.clipper.push(out, ev, time.time())
            if cfg.record:
                if recorder is None:
                    from .media import VideoWriter
                    recorder = VideoWriter(cfg.record, fps=10.0)
                recorder.write(out)
            st.frames_proc += 1
            st.latency_ms = (time.time() - t_cap) * 1000
            if st.frames_proc == 1 and st.on_ready:
                # Diagnostics never affect the run: a bad callback must not kill the loop that
                # is, by having reached this line, demonstrably healthy.
                try:
                    st.on_ready("首帧已产出")
                except Exception as e:                  # noqa: BLE001
                    print(f"[warn] on_ready 回调失败（不影响运行）：{e}")
            n += 1
            dt_win = time.time() - t_win
            if dt_win >= 2.0:
                st.fps_proc = n / dt_win
                c_rec, c_fire, c_sr, c_enc = (s.snap() for s in (s_rec, s_fire, s_sr, s_enc))
                c_gpu = [s.snap() for s in s_gpu]
                c_enh, p_enh = _sum_snaps(c_gpu), _sum_snaps(p_gpu)
                fires = c_fire[1] - p_fire[1]
                # How to read wait_gpu/wait_cpu: they are the share of the window each side
                # spent waiting on the other. wait_cpu high => the GPU is blocked on JPEG
                # encoding, so cut sr/jpeg_quality. wait_gpu high WITH fps < fps_in => the GPU
                # is the bottleneck, so cut proc_max_side. wait_gpu high WITH fps ~= fps_in is
                # the healthy case: there is spare capacity and the source is the limit.
                #
                # The percentages in brackets after `enhance` and `recognize` are DUTY -- the
                # share of the window that card was busy. They are what says whether a box is
                # balanced, and the reason they exist: with recognition on a card of its own,
                # `enhance=46ms` and `infer=312ms x2` are both healthy-looking numbers that
                # together mean cuda:0 is pinned at ~98% while cuda:1 sits ~60% idle. Reading
                # that off the log used to require multiplying the infer mean by the fire count
                # by hand. Cross-check against `nvidia-smi dmon` before acting on it: these are
                # our own wall-clock counters, not SM occupancy.
                print(f"[process] fps_in={st.fps_in:.1f}  fps={st.fps_proc:.1f}  "
                      f"latency={st.latency_ms:.0f}ms  "
                      f"enhance={_mean_ms(c_enh, p_enh):.0f}ms"
                      f"({_duty(c_enh, p_enh, dt_win) / n_gpu:.0f}%)  "
                      f"recognize={_mean_ms(c_rec, p_rec):.0f}ms"
                      f"(infer {_mean_ms(c_fire, p_fire):.0f}ms x{fires})  "
                      # Occupancy of the recognition card: every push, not only the firing ones,
                      # so it includes the per-frame preprocess. Only printed when recognition
                      # has a card to itself, because that is the only case where it is a
                      # statement about a device rather than a share of the enhance thread.
                      + (f"reco_duty={_duty(c_rec, p_rec, dt_win):.0f}%  "
                         if reco_async else "")
                      + f"sr={_mean_ms(c_sr, p_sr):.0f}ms  "
                      f"encode={_mean_ms(c_enc, p_enc):.0f}ms  "
                      f"wait_gpu={(wait['gpu'] - p_wait['gpu']) / dt_win * 100:.0f}%  "
                      f"wait_cpu={(wait['cpu'] - p_wait['cpu']) / dt_win * 100:.0f}%  "
                      # Only when the enhance side is itself split across devices: per card, the
                      # mean enhance, that card's duty, and the share of the window it spent
                      # waiting on a peer. Per card rather than aggregated because a strict
                      # rotation is only as fast as its slowest device, and the aggregate hides
                      # which one that is.
                      + ("".join(
                          f"gpu{k}={_mean_ms(c_gpu[k], p_gpu[k]):.0f}ms/"
                          f"{_duty(c_gpu[k], p_gpu[k], dt_win):.0f}%/"
                          f"peer{(w_peer[k] - p_peer[k]) / dt_win * 100:.0f}%  "
                          for k in range(n_gpu)) if n_gpu > 1 else "")
                      # Only when recognition has a device of its own. reco_q is the depth of
                      # the recognition backlog and reco_drop counts frames the recognizer never
                      # saw. Both being ~0 with `infer` still in the hundreds of ms is the whole
                      # point of the split: recognition is slow and nobody waits for it. reco_q
                      # pinned at 8 with reco_drop climbing means inference has become slower
                      # than stride/fps (~1.1 s) and the window is being sampled sparsely --
                      # raise reco_stride rather than letting it drop frames silently.
                      + (f"reco_q={reco_q.qsize()}  reco_drop={reco_dropped[0]}  "
                         if reco_async else "")
                      + f"(avg over {n} frames)")
                # Where a firing window's time actually went, so "infer is 312 ms" can be acted
                # on. `window` is the np.stack of the 32-frame buffer (18.4 MB) and `marshal` the
                # copy onto the card: both are HOST work, and if they dominate then the fix is
                # the data path, not the model or the card. `forward` also absorbs the async
                # GPU-side transpose, since .cpu() is the only sync point -- see
                # recognize._WindowRecognizer._infer.
                c_parts = recognizer.timings() if recognizer else None
                if c_parts and p_parts and c_parts["n"] > p_parts["n"]:
                    dn = c_parts["n"] - p_parts["n"]
                    print("[reco] " + "  ".join(
                        f"{k}={(c_parts[k] - p_parts[k]) / dn * 1000:.0f}ms"
                        for k in ("window", "marshal", "forward"))
                        + f"  x{dn}  dev={reco_dev}")
                p_parts = c_parts
                # Push/HLS telemetry, printed from here rather than from the feeder threads:
                # a feeder blocked in stdin.write prints nothing, which is exactly the case
                # worth seeing. How to read it -- blocked% is time inside stdin.write, and with
                # a frame larger than the pipe that means "time until ffmpeg had decoded,
                # encoded and SENT enough of the previous frame", so it charges the whole
                # downstream path, not just the peer. Do not read it as "the media server is
                # not draining us": this comment used to say that, and it cost two rounds of
                # chasing bytes. Use the buckets instead -- idr_block vs p_block separates a
                # keyframe burst from a uniform cost, and w=<5ms/.../60+ separates a fixed
                # per-frame wait from a rate limit. restarts climbing => read the [push] ffmpeg
                # error lines; lag > 0 => the cadence fell behind and burst to catch up.
                for key, out in (("push", st.push), ("hls", st.hls)):
                    if out is None:
                        continue
                    c, p = out.snap(), p_out.get(key)
                    if p is not None:
                        # Mean ms inside stdin.write, keyframe writes vs the rest. See the
                        # split in streams.FFmpegOut._write_loop for how to read the comparison.
                        d_in, d_pn = c['idr_n'] - p['idr_n'], c['p_n'] - p['p_n']
                        idr_ms = (c['idr_blocked'] - p['idr_blocked']) / d_in * 1000 if d_in else 0.0
                        p_ms = (c['p_blocked'] - p['p_blocked']) / d_pn * 1000 if d_pn else 0.0
                        print(f"[{key}] alive={int(c['alive'])}  "
                              f"restarts={c['restarts']}  "
                              f"wrote={(c['frames'] - p['frames']) / dt_win:.1f}fps"
                              # `dup` and `missed` are the two halves of the same resampling
                              # bill and only mean something together: dup is a cadence slot
                              # that found no new frame (picture freezes), missed is a produced
                              # frame the latest-wins slot overwrote before anyone sent it. On a
                              # source faster than the cadence, missed is the larger of the two
                              # and used to be invisible. See FFmpegOut._feed for the split.
                              f"(dup {c['dup'] - p['dup']} "
                              f"missed {c['missed'] - p['missed']})  "
                              # MJPEG *into* ffmpeg, not the H.264 coming out -- it is a
                              # feed rate, not a bitrate, and reads ~40x stream_bitrate at
                              # 1080p q85. Named `pipe` so nobody mistakes it for the cap.
                              f"pipe={(c['bytes'] - p['bytes']) * 8 / dt_win / 1e6:.1f}Mbps  "
                              f"blocked={(c['blocked'] - p['blocked']) / dt_win * 100:.0f}%  "
                              # Share of the timeline dropped to stay at the live edge. Since
                              # the write moved to its own thread this is no longer the link's
                              # bill -- the cadence loop is a slot.get() and a put_nowait, so
                              # >0 here means the ring filled or the scheduler starved it.
                              # `ring_full` is the one that names a genuinely slow peer.
                              f"dropped={(c['skipped'] - p['skipped']) / dt_win * 100:.0f}%  "
                              f"ring_full={c['ring_full'] - p['ring_full']}  "
                              f"stall={c['stall']:.1f}s  "
                              # The two fields that see a wedged peer, which none of the ones
                              # above can -- everything else here is billed on ffmpeg's input
                              # side and stays perfect while the output goes nowhere. `wedged`
                              # is a write that has not returned yet, so >0 means the freeze is
                              # happening RIGHT NOW and it is the field to read first when the
                              # picture is frozen.
                              #
                              # Expected shape of a peer that stops reading: wedged climbs ~1s
                              # per second to the kill deadline, then restarts +1 and wedged
                              # returns to 0. On an rtsp push `kills` advances with it, because
                              # ffmpeg ignores its own -timeout there and the watchdog is doing
                              # the work -- that is normal, not a defect (see rtmp_push). On
                              # rtmp/flv the child usually times out first and `kills` stays put.
                              # Either way the push recovers by itself, measured 15s (flv) to
                              # 22s (rtsp) end to end, so a handful of these over a long run is
                              # a flaky peer. What would NOT be normal is wedged climbing with
                              # neither counter moving: that means no restart is happening.
                              f"wedged={c['blocked_s']:.1f}s  "
                              f"kills={c['kills']}  "
                              # Peaks since start, not windowed: they are running maxima, and
                              # diffing two maxima under-reports a window whose worst is below
                              # the all-time worst. A peak that stops growing is the signal.
                              f"peak_block={c['worst_ms']:.0f}ms  peak_lag={c['lag']:.2f}s  "
                              # The whole point of the split: one number per bucket, so a
                              # keyframe-burst wall and a uniform wall look different in the log.
                              f"idr_block={idr_ms:.0f}ms(peak {c['idr_worst']:.0f})  "
                              f"p_block={p_ms:.0f}ms  "
                              # Where the writes landed this window. See the bucketing in
                              # streams.FFmpegOut._write_loop; `timer` heavy => fixed per-frame
                              # wait, and it no longer costs frames now that the cadence thread
                              # is not the one paying it.
                              f"w=<5ms:{c['w_fast'] - p['w_fast']}"
                              f"/<25:{c['w_mid'] - p['w_mid']}"
                              f"/<60:{c['w_timer'] - p['w_timer']}"
                              f"/60+:{c['w_over'] - p['w_over']}"
                              # Source frames consumed per delivered frame: 0 = duplicate,
                              # 1 = 1:1, n = n-1 overwritten. The SHAPE is the point, not the
                              # total -- a tight histogram is regular decimation and reads as a
                              # lower frame rate; a spread one is irregular and reads as
                              # stutter, and both give the same `missed`. But the shape does NOT
                              # predict which cadences the deployment complains about: the
                              # tightest of five measured histograms was one of the two that
                              # stuttered. What predicts it is whether src_fps/out_fps is an
                              # integer -- see stream_fps_for_source and the five-row table in
                              # streams.py:_feed before theorising about this line.
                              f"  adv=0:{c['adv0'] - p['adv0']}"
                              f"/1:{c['adv1'] - p['adv1']}"
                              f"/2:{c['adv2'] - p['adv2']}"
                              f"/3+:{c['adv3p'] - p['adv3p']}")
                    p_out[key] = c
                # Live FLV viewers, one line each, and only while somebody is watching -- the
                # normal deployment fans out through the media server on the push, so this stays
                # silent there and costs nothing. Without it a viewer connected for ten minutes
                # emits exactly two lines (connect, disconnect) and nothing in between, so a
                # stall that resolved before they gave up leaves no trace at all.
                #
                # Shorter than the [push] line on purpose: the fields dropped here (the w=
                # buckets, the adv= histogram, the idr/p split) diagnose the shared encoder's
                # cadence, which is the same code for every output and is already being reported
                # once per window by push/hls. What is per-viewer is whether THIS socket is
                # draining, which is `blocked`/`wedged`/`stall`.
                with st.flv_lock:
                    viewers = list(st.flv_live.values())
                for v in viewers:
                    c, p = v.out.snap(), v.prev
                    sent, p_sent = v.sent, v.prev_sent
                    v.prev, v.prev_sent = c, sent
                    if p is None:               # first window for this viewer, no baseline yet
                        continue
                    print(f"[flv] #{v.seq} {v.ip}  alive={int(c['alive'])}  "
                          f"持续={v.age():.0f}s  "
                          f"wrote={(c['frames'] - p['frames']) / dt_win:.1f}fps"
                          f"(dup {c['dup'] - p['dup']} missed {c['missed'] - p['missed']})  "
                          # H.264 actually handed to this client's socket. Against the [push]
                          # line's `pipe=` (MJPEG in, ~40x larger) this is the other direction;
                          # it should sit near stream_bitrate, and it going flat while
                          # `wrote` keeps up is the signature of a peer that stopped reading.
                          f"发出={(sent - p_sent) * 8 / dt_win / 1e6:.1f}Mbps  "
                          f"blocked={(c['blocked'] - p['blocked']) / dt_win * 100:.0f}%  "
                          f"dropped={(c['skipped'] - p['skipped']) / dt_win * 100:.0f}%  "
                          f"ring_full={c['ring_full'] - p['ring_full']}  "
                          f"stall={c['stall']:.1f}s  wedged={c['blocked_s']:.1f}s  "
                          f"peak_block={c['worst_ms']:.0f}ms")
                # Clip losses, and only when something moved this window -- a healthy recorder
                # already narrates itself (开始/结束/已保存) and an unconditional line here
                # would just push those out of a scrollback. The counters that matter are the
                # ones that mean the writer thread went unresponsive: `forced` is an `end` that
                # never arrived, so that clip was dropped and its encoder force-released
                # (before that release existed, each one leaked an ffmpeg child, a thread and
                # two fds for the life of the process -- see clips.force_close). `abandoned`
                # climbing with `forced` flat is the harmless half, a clip given up on before
                # it ever opened a file.
                if st.clipper is not None:
                    cl = st.clipper.stats()
                    if any(cl[k] != p_clip.get(k, 0) for k in
                           ("clips_abandoned", "clips_forced", "frames_dropped",
                            "uploads_dropped")):
                        print(f"[clip] saved={cl['clips_saved']}  "
                              f"abandoned={cl['clips_abandoned']}  "
                              f"forced={cl['clips_forced']}  "
                              f"frames_dropped={cl['frames_dropped']}  "
                              f"uploads_dropped={cl['uploads_dropped']}")
                    p_clip = cl
                n, t_win = 0, time.time()
                p_rec, p_fire, p_sr, p_enc = c_rec, c_fire, c_sr, c_enc
                p_gpu = c_gpu
                p_wait = dict(wait)
                p_peer = list(w_peer)
        if recorder:
            recorder.close()

    def enhance_loop():
        """GPU: newest captured frame -> enhanced frame, fanned out to output and recognition."""
        bind_thread_device(enh_devices[0])
        last_seq = 0
        while running():        # `running`, not just st.stop: a dead worker stops this too
            # Wait for the encoder to have room BEFORE picking a frame. Enhancing first and
            # blocking on the put afterwards would pipeline just as well, but it would spend
            # the GPU on whichever frame was newest a whole output period ago. This thread is
            # out_q's only producer, so the room is still there when the put finally happens.
            t_wait = time.time()
            got_space = space.wait(0.1)
            wait["cpu"] += time.time() - t_wait
            if not got_space:
                continue                      # recheck running()
            item, seq = st.raw.get()
            if item is None or seq == last_seq:
                # Waiting on the source, not on the encoder -- so leave `space` alone and do
                # not bill this to wait_cpu. Clearing it here would stall the next iteration
                # on a queue that has room.
                time.sleep(0.002)
                continue
            space.clear()                     # committed to filling the slot
            last_seq = seq
            frame, t_cap = item
            t0 = time.time()
            chunk = [frame]
            for s in enh:
                chunk = s(chunk)
            s_gpu[0].add(time.time() - t0)
            # Recognition off this thread when it has a device of its own; when it does not,
            # inline and before the handoff so the event and the frame it was computed from stay
            # together. Either way it reads the enhanced pre-SR frame. The pre-enhancement
            # `frame` rides along for the illuminance gate.
            ev = hand_to_recognizer(chunk[0], t_cap, raw_frame=frame)
            out_q.put((chunk[0], t_cap, ev))

    # --- multi-GPU. Unused when the enhance side has one device, which keeps the single-GPU
    # path above byte-for-byte the one that was measured and shipped. Sized by enh_devices,
    # not devices: the recognition card does not take part in the rotation.
    in_q = [queue.Queue(maxsize=1) for _ in enh_devices]     # dealer -> worker k
    done_q = [queue.Queue(maxsize=1) for _ in enh_devices]   # worker k>0 -> worker 0
    # Room in in_q[k], acquired by the dealer and released by worker k when it takes the
    # frame. Same purpose as `space` above -- do not read st.raw until there is somewhere to
    # put the frame -- but one slot deeper in effect, because the release happens on the get
    # rather than after the enhance. That gives each worker a one-frame prefetch, which is
    # what stops the strict rotation from stalling a free worker while the dealer is blocked
    # on a busy one. It costs freshness: a frame can sit in in_q[k] for up to one enhance
    # period (~55 ms at 720) before anyone touches it. Cheap against a 1 s budget, and the
    # alternative -- releasing after the enhance -- idles a whole GPU to save it.
    slot = [threading.Semaphore(1) for _ in enh_devices]

    def put_until(q, item, bill=None):
        """Blocking put that still notices shutdown. False if the pipeline is going down.

        `bill` is a (container, key) pair -- the `wait` dict or the per-device `w_peer` list --
        so the wait is charged even when the put is abandoned at shutdown.
        """
        t_wait = time.time()
        try:
            while running():
                try:
                    q.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    pass
            return False
        finally:
            if bill:
                container, key = bill
                container[key] += time.time() - t_wait

    def enhance_on(k, frame):
        t0 = time.time()
        chunk = [frame]
        for s in enh_sets[k]:
            chunk = s(chunk)
        s_gpu[k].add(time.time() - t0)
        return chunk[0]

    def dealer():
        """Read the newest frame and hand it to the next device, strictly round-robin.

        Strict rotation is what lets worker 0 reorder with `next_out % n_gpu` instead of a
        reorder buffer keyed by sequence number. The cost is that one slow device throttles
        the rotation for everybody, which is fine while the devices are homogeneous -- and
        shows up as the `peer` share of the per-card `gpuK=` telemetry group if that ever stops
        being true.
        """
        last_seq, k = 0, 0
        while running():
            # Not billed to any wait counter: the dealer blocking here means every worker is
            # busy, which is the healthy saturated state, not a bottleneck. wait_cpu keeps
            # its single meaning -- GPU work blocked behind the encoder -- and is billed by
            # the sequencer, the only thread here that ever waits on out_q.
            if not slot[k].acquire(timeout=0.1):
                continue
            item, seq = st.raw.get()
            if item is None or seq == last_seq:
                slot[k].release()     # nothing new; do not burn worker k's turn on a stale frame
                time.sleep(0.002)
                continue
            last_seq = seq
            in_q[k].put(item)         # the semaphore already guaranteed the room
            k = (k + 1) % n_gpu

    def gpu_worker(k):
        """devices[k], k>0: enhance only, then hand off to worker 0 to be emitted in order."""
        def loop():
            bind_thread_device(enh_devices[k])
            while running():
                try:
                    frame, t_cap = in_q[k].get(timeout=0.1)
                except queue.Empty:
                    continue
                slot[k].release()
                # Billed to w_peer[k]: blocking here means devices[0] has not come round to
                # collect, so this card is idle waiting on another card. That is the same
                # imbalance the sequencer's done_q wait measures, seen from the other side --
                # which side shows it just says who is ahead. The raw pre-enhancement frame
                # rides along for the recognizer's illuminance gate.
                put_until(done_q[k], (enhance_on(k, frame), t_cap, frame), bill=(w_peer, k))
        loop.__name__ = f"gpu_worker{k}"
        return loop

    def sequencer():
        """enh_devices[0]: its share of the enhancing, the ordering, and recognition if it owns it.

        Ordering is free here: this loop already walks the output sequence, so `next_out % n_gpu`
        names the device that owns the next frame and no reorder buffer is needed.

        Recognition only runs inline when there is no device to spare for it (a single card).
        With a card of its own it goes to reco_loop -- see the multi-GPU section of the
        process_loop docstring for the measurement that moved it, and for why it cannot simply
        become a second thread on a card that is already enhancing.
        """
        bind_thread_device(enh_devices[0])
        next_out = 0
        while running():
            k = next_out % n_gpu
            if k == 0:
                try:
                    frame, t_cap = in_q[0].get(timeout=0.1)
                except queue.Empty:
                    continue
                slot[0].release()
                enhanced = enhance_on(0, frame)
                raw_frame = frame
            else:
                t_wait = time.time()
                try:
                    enhanced, t_cap, raw_frame = done_q[k].get(timeout=0.1)
                    w_peer[0] += time.time() - t_wait
                except queue.Empty:
                    w_peer[0] += time.time() - t_wait
                    continue
            ev = hand_to_recognizer(enhanced, t_cap, raw_frame=raw_frame)
            # wait_cpu, same meaning as in enhance_loop: GPU work blocked behind the encoder.
            if not put_until(out_q, (enhanced, t_cap, ev), bill=(wait, "cpu")):
                return
            next_out += 1

    if n_gpu == 1:
        workers, main_loop = [worker(output_loop)], enhance_loop
    else:
        workers = ([worker(output_loop), worker(dealer)]
                   + [worker(gpu_worker(k)) for k in range(1, n_gpu)])
        main_loop = sequencer
    if reco_async:
        workers.append(worker(reco_loop))
    for t in workers:
        t.start()
    try:
        main_loop()
    finally:
        # `done` rather than `st.stop`: the workers have to stop before the stages they are
        # calling into get closed underneath them, whether this is a real shutdown or the
        # enhance loop failing on its own.
        done.set()
        for t in workers:
            t.join(timeout=5)
        for s in frame_stages:
            s.close()
        if recognizer:
            recognizer.close()


def _encode_ceiling_probe(st: ServerState, cfg):
    """Print the encode-side fps ceiling once, a few seconds after the push starts.

    Exists because the one number that would settle the diagnosis -- is `blocked%` the encoder
    or the network? -- can only be measured from inside the container, and the container is not
    reachable by shell in the deployment this was written for. So it goes into the log.

    Read it against max_stream_fps on the same `[config]` line:

        ceiling >> max_stream_fps  => encoding has headroom; the write block is downstream
                                      (media server / link), and `idr_block` on the [push]
                                      line says whether it is the keyframe burst specifically
        ceiling ~ max_stream_fps   => the encoder itself cannot keep the cadence

    Waits for a real JPEG rather than synthesising one: frame size and detail are exactly what
    the measurement is sensitive to, and a synthetic frame would be neither. Everything is
    swallowed and printed -- a diagnostic must never take down a stream (same rule as
    probe_video / _print_gpu_inventory).
    """
    try:
        # Local, because build_app() imports streams locally too and this function is at module
        # scope -- the first deployment of this probe died on `name 'streams' is not defined`,
        # printed by its own except branch, which is why the ceiling is missing from that log.
        from . import streams
        # The first JPEGs come out while the GPU is still settling; sampling one of those
        # would measure a frame the stream has already moved past. 5 s is after warmup.
        deadline = time.time() + 20.0
        jpeg = None
        while time.time() < deadline and not st.stop.is_set():
            jpeg, _ = st.jpeg.get()
            if jpeg:
                break
            time.sleep(0.5)
        if not jpeg or st.stop.is_set():
            print("[selftest] 跳过编码上限自测：启动 20 秒内没有取到画面")
            return
        time.sleep(5.0)
        jpeg, _ = st.jpeg.get()
        fps = streams.encode_ceiling(jpeg, cfg.max_stream_fps, cfg.stream_bitrate)
        if fps <= 0:
            print("[selftest] 编码上限自测未能运行（ffmpeg 不可用或被超时杀掉）")
            return
        ratio = fps / max(1e-6, float(cfg.max_stream_fps))
        # Deliberately not "so the stutter is in the output path": headroom here only rules out
        # the encoder. A stutter with plenty of headroom is most often the resampling ratio, not
        # the link -- see stream_fps_for_source and the [push] adv/missed counters.
        verdict = ("编码有余量，卡顿另有原因（先看 [push] 的 adv/missed 与出流节拍）" if ratio >= 2.0 else
                   "编码勉强够，余量不足" if ratio >= 1.2 else "编码本身就是瓶颈")
        print(f"[selftest] 编码上限 {fps:.1f} fps（帧 {len(jpeg) / 1024:.0f}KB，"
              f"目标 {cfg.max_stream_fps:g} fps，{ratio:.1f}x）-> {verdict}")
    except Exception as e:                              # noqa: BLE001
        print(f"[selftest] 编码上限自测异常，已忽略: {e}")


def build_app(cfg, on_clip=None, on_ready=None):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    st = ServerState(cfg, on_clip=on_clip)
    st.on_ready = on_ready
    if cfg.clip_dir:
        from .clips import ClipRecorder, EventLog
        st.clipper = ClipRecorder(
            cfg.clip_dir, pre_sec=cfg.clip_pre_sec, post_sec=cfg.clip_post_sec,
            max_sec=cfg.clip_max_sec,
            skip_labels=[s.strip() for s in cfg.clip_skip_labels.split(",") if s.strip()],
            min_confidence=cfg.clip_min_conf, on_saved=on_clip,
            session=(cfg.clip_session or None), denoise=cfg.clip_denoise)
        print(f"[serve] 片段保存已开启 -> {st.clipper.root} "
              f"(跳过 {sorted(st.clipper.skip) or '无'})")
        # Every event, including the ones no clip is cut for -- this is the record that
        # survives when /events is not reachable from outside the container.
        st.eventlog = EventLog(os.path.join(st.clipper.root, "events.jsonl"))
        print(f"[serve] 事件日志 -> {st.eventlog.path}")

    from . import streams
    st.formats = streams.parse_formats(cfg.stream_formats)

    @asynccontextmanager
    async def lifespan(app):
        threads = [threading.Thread(target=capture_loop, args=(st,), daemon=True),
                   threading.Thread(target=process_loop, args=(st,), daemon=True)]
        for t in threads:
            t.start()
        # Started after the workers so the first JPEG is usually there by the time ffmpeg
        # asks for one; an empty slot only costs a repeated frame, not a failure.
        #
        # The cadence has to be settled before any of them spawns: ffmpeg's -framerate is fixed
        # at exec time, so correcting it later would mean restarting a live push. Source fps is
        # only known once capture_loop has the stream open, hence the bounded wait -- and the
        # wait is announced BEFORE it starts, because an RTSP source that never answers stalls
        # right here and that line is the difference between "hung" and "waiting for the camera".
        # Timing out is not fatal: the requested cadence is then used uncalibrated.
        if not st.src_fps_known.is_set():
            print("[serve] 等待读到源帧率以校准出流节拍（最多 8 秒）")
            st.src_fps_known.wait(8.0)
        fps, note = stream_fps_for_source(cfg.max_stream_fps, st.src_fps)
        if abs(fps - cfg.max_stream_fps) > 1e-9:
            print(f"[warn] 出流节拍 {cfg.max_stream_fps:g} -> {fps:g} fps：{note}")
            # Mutated once, here, so /stream, /live.flv, HLS, the push and the encode-ceiling
            # self-test all read the same corrected value from cfg rather than each growing its
            # own copy of this rule.
            cfg.max_stream_fps = fps
        else:
            print(f"[serve] 出流节拍 {fps:g} fps：{note}")
        if "hls" in st.formats:
            st.hls_dir = cfg.hls_dir or os.path.join(tempfile.gettempdir(), "darkpipe_hls")
            st.hls = streams.hls_writer(st.jpeg, cfg.max_stream_fps, st.hls_dir,
                                        cfg.stream_bitrate)
            print(f"[serve] HLS 分片目录 {st.hls_dir}")
        if cfg.rtmp_push_url:
            st.push = streams.rtmp_push(st.jpeg, cfg.max_stream_fps, cfg.rtmp_push_url,
                                        cfg.stream_bitrate)
            print(f"[serve] 推流到 {cfg.rtmp_push_url}")
            # The exact command, so the same push can be reproduced by hand from a shell on
            # the host when the platform side and this side disagree about who dropped it.
            # stdin_pipe is the size F_SETPIPE_SZ actually granted: 65536 means the request
            # silently failed, 0 means the platform has no F_GETPIPE_SZ at all (macOS), and in
            # both cases only the ring in _write_loop is providing slack. It is printed because
            # a whole deployment round could not tell "did not help" from "never ran" -- see
            # streams.FFmpegOut._widen_stdin.
            print(f"[push] {' '.join(st.push._cmd)}")
            print(f"[push] stdin_pipe={st.push.pipe_bytes}B"
                  + (f"（{st.push.pipe_note}）" if st.push.pipe_note else "")
                  + f"  ring={streams.RING_FRAMES}帧")
        # Unconditional: /stream and FLV go through the same encoder settings, so the ceiling
        # is worth having even with no push configured. One 60-frame burst, then the thread ends.
        threading.Thread(target=_encode_ceiling_probe, args=(st, cfg), daemon=True,
                         name="probe-encode").start()
        yield
        for out in (st.hls, st.push):
            if out is not None:
                out.close()
        st.stop.set()
        for t in threads:
            t.join(timeout=5)
        if st.clipper is not None:
            st.clipper.close()                 # flush the in-flight clip before exiting
            print(f"[serve] 片段统计 {st.clipper.stats()}")
        if st.eventlog is not None:
            st.eventlog.close()
            print(f"[serve] 事件日志统计 {st.eventlog.stats()}")

    app = FastAPI(title="darkpipe", lifespan=lifespan)
    app.state.dark = st

    @app.get("/health")
    def health():
        body = dict(status="ok" if st.capture_alive else "degraded",
                    uptime_s=round(time.time() - st.t_start, 1),
                    capture_alive=st.capture_alive, source=str(cfg.input),
                    reconnects=st.reconnects, fps_in=round(st.fps_in, 2),
                    fps_proc=round(st.fps_proc, 2),
                    frames_dropped=max(0, st.frames_in - st.frames_proc),
                    latency_ms_last=round(st.latency_ms, 1),
                    events_total=st.events_total,
                    last_label=(st.last_event.label if st.last_event else None))
        if st.clipper is not None:
            body["clips"] = st.clipper.stats()
            body["clip_dir"] = st.clipper.root
        if st.eventlog is not None:
            body["event_log"] = st.eventlog.stats()
        body["stream_formats"] = st.formats
        # flv_clients is the live count; flv_total/flv_rejected are the history behind it. Poll
        # flv_total to count reconnects -- the live count returns to its old value after every
        # drop-and-retry cycle and so reports nothing about them. flv_rejected climbing means
        # players are being turned away at max_flv_clients, which looks to them exactly like an
        # unstable stream.
        body["flv_clients"] = st.flv_clients
        body["flv_total"] = st.flv_total
        body["flv_rejected"] = st.flv_rejected
        with st.flv_lock:
            viewers = list(st.flv_live.values())
        body["flv_viewers"] = [
            dict(seq=v.seq, ip=v.ip, age_s=round(v.age(), 1), alive=v.out.alive(),
                 wedged_s=round(v.out.blocked_s(), 1),
                 mb_sent=round(v.sent / 1e6, 1))
            for v in viewers]
        body["stream_bitrate"] = cfg.stream_bitrate
        # An ffmpeg that died takes its format down silently otherwise -- the endpoint keeps
        # answering, it just never produces bytes. Report liveness per output, with ffmpeg's
        # own last words when it is gone.
        #
        # `{key}_alive` is necessary but NOT sufficient, and reading it as sufficient is what let
        # a frozen output answer 200/ok for a whole shift: it only asks poll(), so an ffmpeg
        # whose socket writes block sits there as alive=true forever. `{key}_wedged_s` is the
        # field that sees that -- seconds the current write into ffmpeg has been blocked, >0 only
        # when the output is frozen right now. See streams.PUSH_TIMEOUT_US for the measurement.
        wedged = []
        for key, out in (("hls", st.hls), ("push", st.push)):
            if out is not None:
                body[f"{key}_alive"] = out.alive()
                body[f"{key}_restarts"] = out.restarts
                blocked = out.blocked_s()
                body[f"{key}_wedged_s"] = round(blocked, 1)
                body[f"{key}_kills"] = out.snap()["kills"]
                if blocked > 0 or not out.alive():
                    wedged.append(key)
                err = out.error_tail()
                if err:
                    body[f"{key}_error"] = err
        # Degrade on a wedged or dead output. The status field is what an orchestrator polls, so
        # a stream nobody can watch must not read as ok here -- that gap is why the freeze was
        # only ever noticed by someone looking at the picture. The HTTP code stays tied to
        # capture_alive: losing one output format is a degradation, not a dead service, and
        # flipping to 503 would make a platform restart the container over a transient reconnect.
        if wedged and body["status"] == "ok":
            body["status"] = "degraded"
            body["degraded_reason"] = "出流阻塞或已退出：" + ",".join(wedged)
        return JSONResponse(body, status_code=200 if st.capture_alive else 503)

    @app.get("/config")
    def config():
        d = asdict(cfg)
        d.pop("warnings", None)
        return d

    @app.get("/stream")
    async def stream():
        async def gen():
            last = 0
            interval = 1.0 / cfg.max_stream_fps
            while True:
                jpg, seq = st.jpeg.get()
                if jpg is not None and seq != last:
                    last = seq
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
                           + jpg + b"\r\n")
                await asyncio.sleep(interval)
        return StreamingResponse(gen(),
                                 media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/live.flv")
    async def live_flv(request: Request):
        """HTTP-FLV — the same shape the GB28181 gateways serve on the input side.

        One ffmpeg per viewer (see streams.flv_pipe for why), so the client count is capped
        rather than left to take the box down under a crowd.

        Both ends of the connection are logged, paired by `#seq`, because a reconnecting player
        is only visible as the *sequence* of those lines -- one viewer that drops and returns
        six times looks identical, in every other field this server exposes, to one viewer that
        sat there quietly. See ServerState.flv_total.
        """
        from fastapi.responses import Response
        if "flv" not in st.formats:
            return Response("flv 未在 stream_formats 中启用", status_code=404,
                            media_type="text/plain; charset=utf-8")
        ip = request.client.host if request.client else "?"
        with st.flv_lock:
            if st.flv_clients >= cfg.max_flv_clients:
                st.flv_rejected += 1
                # Printed, not silent: a player that retries into a full server produces exactly
                # the reconnect symptom being chased, and a 503 that only the client sees puts
                # the evidence on the wrong side of the wire.
                print(f"[flv] 拒绝 {ip}：并发已满 {st.flv_clients}/{cfg.max_flv_clients}"
                      f"（累计拒绝 {st.flv_rejected}）")
                return Response(f"FLV 并发观看数已达上限 {cfg.max_flv_clients}", status_code=503,
                                media_type="text/plain; charset=utf-8")
            st.flv_clients += 1
        try:
            out = streams.flv_pipe(st.jpeg, cfg.max_stream_fps, cfg.stream_bitrate)
        except Exception as e:                                   # noqa: BLE001
            with st.flv_lock:
                st.flv_clients -= 1
            print(f"[flv] {ip} 启动编码失败：{e}")
            return Response(f"无法启动 FLV 编码: {e}", status_code=503,
                            media_type="text/plain; charset=utf-8")

        with st.flv_lock:
            st.flv_total += 1
            st.flv_seq += 1
            viewer = _FlvViewer(st.flv_seq, ip, out)
            st.flv_live[viewer.seq] = viewer
        print(f"[flv] #{viewer.seq} 连接 {ip}  "
              f"当前 {st.flv_clients}/{cfg.max_flv_clients}  累计 {st.flv_total}")

        async def gen():
            # Which of the two ends closed first. The distinction is the whole diagnostic value
            # of the disconnect line: "客户端断开" points at the player or the network, "编码器退出"
            # points at us, and they need opposite investigations. Default is the player,
            # because that is the path that does not go through the break below.
            reason = "客户端断开"
            try:
                while True:
                    # read1, not read: read(n) waits for the full n bytes, which holds finished
                    # frames back until the buffer fills. read1 forwards whatever ffmpeg has
                    # already produced -- the difference is latency, which is the whole point
                    # of choosing FLV over HLS.
                    chunk = await asyncio.to_thread(out.proc.stdout.read1, 65536)
                    if not chunk:
                        # EOF on ffmpeg's stdout. flv_pipe is the one output that does NOT
                        # restart itself (pipe_stdout => _restart False, no watchdog), because
                        # its stdout IS this HTTP response -- a replacement child would be
                        # writing into a pipe nobody is reading. So this is terminal.
                        reason = "编码器退出"
                        break
                    viewer.sent += len(chunk)
                    yield chunk
            finally:
                c = out.snap()
                out.close()
                with st.flv_lock:
                    st.flv_clients -= 1
                    st.flv_live.pop(viewer.seq, None)
                age = viewer.age()
                # Same fields as the [push] line, in the same order, so the two can be read
                # side by side: everything before `wedged` is billed on ffmpeg's input side and
                # stays perfect while the output goes nowhere -- `wedged` and `stall` are the
                # two that see a peer which stopped reading. A short `持续` with a nonzero
                # `wedged` is a viewer we starved; a short one with both at zero is a viewer
                # that left.
                print(f"[flv] #{viewer.seq} 断开 {ip}  原因={reason}  "
                      f"持续={age:.0f}s  "
                      # 发出 is H.264 to this client; 喂入 is MJPEG into ffmpeg and reads ~40x
                      # larger. Keeping both makes the direction unmistakable -- see the `pipe=`
                      # comment on the [push] line for the confusion this naming prevents.
                      f"发出={viewer.sent / 1e6:.1f}MB"
                      f"({viewer.sent * 8 / age / 1e6 if age > 0 else 0:.1f}Mbps)  "
                      f"喂入={c['bytes'] / 1e6:.1f}MB  "
                      f"帧={c['frames']}({c['frames'] / age if age > 0 else 0:.1f}fps)  "
                      f"dup={c['dup']} missed={c['missed']}  "
                      f"dropped={c['skipped']}  ring_full={c['ring_full']}  "
                      f"wedged={c['blocked_s']:.1f}s  stall={c['stall']:.1f}s  "
                      f"peak_block={c['worst_ms']:.0f}ms"
                      + (f"  err={out.error_tail()}" if reason == "编码器退出" else ""))

        return StreamingResponse(gen(), media_type="video/x-flv")

    @app.get("/hls/{name}")
    def hls_file(name: str):
        """Playlist and segments. Served from the segmenter's own directory."""
        from fastapi.responses import FileResponse, Response
        if st.hls is None:
            return Response("hls 未在 stream_formats 中启用", status_code=404,
                            media_type="text/plain; charset=utf-8")
        # The path comes from a URL; keep it to a bare filename so it cannot walk out of the
        # segment directory.
        if name != os.path.basename(name) or not name:
            return Response("bad name", status_code=400, media_type="text/plain")
        path = os.path.join(st.hls_dir, name)
        if not os.path.exists(path):
            return Response("尚未生成（分片需要几秒）", status_code=404,
                            media_type="text/plain; charset=utf-8")
        mime = ("application/vnd.apple.mpegurl" if name.endswith(".m3u8")
                else "video/mp2t")
        return FileResponse(path, media_type=mime,
                            headers={"Cache-Control": "no-cache"})

    @app.get("/events")
    async def events():
        async def gen():
            q = st.bus.subscribe()
            try:
                while True:
                    try:
                        ev = await asyncio.to_thread(q.get, True, 15.0)
                        yield f"event: recognition\ndata: {json.dumps(ev.to_dict())}\n\n"
                    except queue.Empty:
                        yield ": ping\n\n"
            finally:
                st.bus.unsubscribe(q)
        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def run_server(cfg, on_clip=None, stop_after=0.0, on_ready=None):
    """Blocks until stopped. stop_after > 0 bounds the run (0 = until killed).

    The bound exists because a persistent service is otherwise untestable end-to-end and
    unschedulable as a finite job: with it, the same code path a camera runs forever can be
    run for 60 seconds in CI or by an orchestrator that needs the container to terminate.
    """
    import uvicorn
    # on_ready fires from the processing loop, not from here: uvicorn is listening long before
    # the weights are loaded and the camera is open, so "the port answers" is not "it works".
    app = build_app(cfg, on_clip=on_clip, on_ready=on_ready)
    print(f"[serve] http://{cfg.host}:{cfg.port}  endpoints: /stream /events /health /config")
    server = uvicorn.Server(uvicorn.Config(app, host=cfg.host, port=cfg.port,
                                           log_level="warning"))
    if stop_after and stop_after > 0:
        print(f"[serve] run_seconds={stop_after:g}，到期后自动退出")

        def _bell():
            print(f"[serve] 运行时长已达 {stop_after:g}s，开始收尾退出")
            server.should_exit = True

        t = threading.Timer(stop_after, _bell)
        t.daemon = True
        t.start()
    server.run()
    return app.state.dark
