"""Online streaming-inference server (FastAPI).

CaptureThread: cv2.VideoCapture -> single-slot latest-frame buffer (newest overwrites =
implicit drop policy when the GPU is slower than the stream; bounded latency).
Then two concurrent threads, split by the resource they contend for, so throughput is their
max rather than their sum (see process_loop for the measurements):
  GpuThread     enhance -> recognizer.push
  OutputThread  SR -> label bar -> JPEG -> clip recorder
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
        self.last_event = None
        self.events_total = 0
        self.on_clip = on_clip
        self.clipper = None
        self.eventlog = None
        self.formats = ["mjpeg"]
        self.hls = None                 # shared HLS segmenter
        self.hls_dir = ""
        self.push = None                # shared RTMP/RTSP push
        self.flv_clients = 0
        self.flv_lock = threading.Lock()


def capture_loop(st: ServerState):
    backoff = 0.5
    while not st.stop.is_set():
        try:
            cap = open_capture(st.cfg.input)
            st.capture_alive = True
            backoff = 0.5
            is_file = str(st.cfg.input).find("://") < 0 and not str(st.cfg.input).isdigit()
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
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

    Recognition stays on the GPU thread, and that is deliberate: it was tried on a third
    thread of its own, to keep its bursty cost (~2 ms to buffer a frame, then a full window
    inference every `stride` frames) off the frame path entirely. It measured WORSE than this
    on both axes -- 15.2 fps, p95 159 ms, max 403 ms. Two Python threads issuing CUDA work
    time-slice against each other on one device: the inference inflated 73 ms -> 250 ms and
    dragged enhance from 55 ms to 62 ms. The stall was not removed, just smeared across every
    frame that overlapped it, which is what wrecked the tail. Keeping all GPU work on one
    thread serialises it by construction and costs nothing, because the GPU is the bottleneck
    either way (see wait_gpu/wait_cpu in the telemetry below).

    Ordering: the single hop is single-producer/single-consumer, so the clip recorder still
    sees frames in capture order with no sequence bookkeeping -- the same argument the
    offline runner relies on (pipeline.py).

    MORE THAN ONE GPU (`--gpus 0,1`) deals frames round-robin across the devices, adding a
    dealer thread and one enhance thread per extra device. It is a separate path, entered
    only when serve_devices returns more than one device, and the two-thread loop above is
    left untouched -- that path is what was measured and shipped, and it is still the
    production path wherever the platform grants one card.

    The reason it has to be more cards rather than more threads: enhance at 720x404 is
    1.9 ms of upload, 35.8 ms of GPU compute, 3.7 ms of download, so 86% of it is the GPU
    itself and only the other 14% can overlap on one device. Measured, dealing to two
    instances (`/tmp/multigpu_bench.py`, 23.7 fps for one instance as the baseline):

        2 instances on one card, default stream       25.2 fps   1.06x
        2 instances on one card, separate streams     27.1 fps   1.14x  <- at the 1/0.86 cap
        2 instances on two cards                      42.3 fps   1.78x

    Thread layout, and why it is this and not something simpler:

        dealer          pure CPU: newest frame from st.raw -> in_q[k], strictly rotating
        gpu_worker[k>0] devices[k]: enhance only -> done_q[k]
        gpu_worker[0]   devices[0]: enhance + ALL recognition + emitting in order
        output_loop     SR -> label bar -> JPEG -> clipper (unchanged, still one thread)

    Recognition is folded into worker 0 rather than given a thread, for the reason measured
    two paragraphs up: a second thread issuing CUDA work on a card that already has one
    makes both slower. Every device here carries exactly one GPU-submitting thread. Putting
    it in worker 0 specifically is also what makes ordering free -- that loop already walks
    the output sequence, so with strict rotation `next_out % n_gpu` names the device owning
    the next frame, and out-of-order completion is absorbed by a blocking get on done_q[k].
    No reorder buffer, no sequence numbers.

    End to end through this loop, two RTX 3090s vs one, same frames, alternated 3x for
    medians (a source offered faster than capacity, so the queues stay full -- this is the
    saturated case, not an idle one):

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

    So this spends latency to buy rate, the same trade as the two-thread split above and for
    the same reason: the binding budget is >=15 fps and <=1 s, and at 840 the result is
    23.8 fps against a 289 ms p95 -- 59% over the fps bar with 3.4x of headroom on delay.
    Tightening the prefetch would give some of that latency back at a cost in throughput; it
    is not worth doing while the delay budget is this slack.
    """
    cfg = st.cfg
    devices = serve_devices(cfg)
    n_gpu = len(devices)
    # One independent set of enhance stages per device. Everything after recognition stays
    # single-instance on devices[0]: the output thread is single and CPU-bound, so a second
    # copy would only cost memory. build_stages hands back a recognizer each time too, but
    # only the first is ever load()ed -- the extras cost a constructor and no weights.
    enh_sets, srs, recognizer, frame_stages = [], [], None, []
    for i, dev in enumerate(devices):
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
                recognizer.load(dev)
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
    s_enh, s_rec, s_fire, s_sr, s_enc = _Stat(), _Stat(), _Stat(), _Stat(), _Stat()
    # Seconds each side spent waiting on the other. Together they say which side is the
    # bottleneck -- the one thing the per-stage means stop being able to tell you once the
    # stages overlap. See the telemetry line below for how to read them.
    wait = {"gpu": 0.0, "cpu": 0.0, "peer": 0.0}
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

    def recognize(frame, t_cap):
        """Buffer a frame; every `stride` frames this also fires the window inference."""
        t0 = time.time()
        # Capture time, not "now": the recognizer's window is defined in wall-clock seconds
        # (`span_sec`), so feeding it processing time would stretch the window by however far
        # behind the source this thread is running.
        ev = recognizer.push(frame, rec_idx[0], t_cap - st.t_start)
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

    def output_loop():
        """CPU: SR -> label bar -> JPEG. Owns the clip recorder and the end-to-end counters."""
        recorder = None
        n, t_win = 0, time.time()
        p_enh, p_rec, p_fire, p_sr, p_enc = (s.snap() for s in
                                             (s_enh, s_rec, s_fire, s_sr, s_enc))
        p_wait = dict(wait)
        p_out = {}
        while running():
            t_wait = time.time()
            try:
                frame, t_cap, ev = out_q.get(timeout=0.1)
            except queue.Empty:
                wait["gpu"] += time.time() - t_wait
                continue
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
            # `ev` travelled with the frame that produced it rather than in a side channel,
            # so the recorder still gets the exact pairing the serial loop gave it.
            # The frame handed to the recorder is the one the demo stream shows -- label bar
            # burned in -- because these clips are for people to watch, not to re-analyse.
            if st.clipper is not None:
                st.clipper.push(out, ev, time.time())
            if cfg.record:
                if recorder is None:
                    from .media import VideoWriter
                    recorder = VideoWriter(cfg.record, fps=10.0)
                recorder.write(out)
            st.frames_proc += 1
            st.latency_ms = (time.time() - t_cap) * 1000
            n += 1
            dt_win = time.time() - t_win
            if dt_win >= 2.0:
                st.fps_proc = n / dt_win
                c_enh, c_rec, c_fire, c_sr, c_enc = (s.snap() for s in
                                                     (s_enh, s_rec, s_fire, s_sr, s_enc))
                fires = c_fire[1] - p_fire[1]
                # How to read wait_gpu/wait_cpu: they are the share of the window each side
                # spent waiting on the other. wait_cpu high => the GPU is blocked on JPEG
                # encoding, so cut sr/jpeg_quality. wait_gpu high WITH fps < fps_in => the GPU
                # is the bottleneck, so cut proc_max_side. wait_gpu high WITH fps ~= fps_in is
                # the healthy case: there is spare capacity and the source is the limit.
                print(f"[process] fps_in={st.fps_in:.1f}  fps={st.fps_proc:.1f}  "
                      f"latency={st.latency_ms:.0f}ms  "
                      f"enhance={_mean_ms(c_enh, p_enh):.0f}ms  "
                      f"recognize={_mean_ms(c_rec, p_rec):.0f}ms"
                      f"(infer {_mean_ms(c_fire, p_fire):.0f}ms x{fires})  "
                      f"sr={_mean_ms(c_sr, p_sr):.0f}ms  "
                      f"encode={_mean_ms(c_enc, p_enc):.0f}ms  "
                      f"wait_gpu={(wait['gpu'] - p_wait['gpu']) / dt_win * 100:.0f}%  "
                      f"wait_cpu={(wait['cpu'] - p_wait['cpu']) / dt_win * 100:.0f}%  "
                      # Only with more than one device: the share of the window devices[0]
                      # spent waiting on its peers. High => the round-robin is unbalanced,
                      # which it mildly is by construction (devices[0] also recognises).
                      + (f"wait_peer={(wait['peer'] - p_wait['peer']) / dt_win * 100:.0f}%  "
                         if n_gpu > 1 else "")
                      + f"(avg over {n} frames)")
                # Push/HLS telemetry, printed from here rather than from the feeder threads:
                # a feeder blocked in stdin.write prints nothing, which is exactly the case
                # worth seeing. How to read it -- blocked% high => the media server is not
                # draining us, so the fault is downstream of the container; blocked% ~0 with
                # wrote ~= max_stream_fps and restarts flat => our side is delivering and a
                # viewer-side outage is not ours; restarts climbing => read the [push] ffmpeg
                # error lines; lag > 0 => the cadence fell behind and burst to catch up.
                for key, out in (("push", st.push), ("hls", st.hls)):
                    if out is None:
                        continue
                    c, p = out.snap(), p_out.get(key)
                    if p is not None:
                        print(f"[{key}] alive={int(c['alive'])}  "
                              f"restarts={c['restarts']}  "
                              f"wrote={(c['frames'] - p['frames']) / dt_win:.1f}fps"
                              f"(dup {c['dup'] - p['dup']})  "
                              f"rate={(c['bytes'] - p['bytes']) * 8 / dt_win / 1e6:.1f}Mbps  "
                              f"blocked={(c['blocked'] - p['blocked']) / dt_win * 100:.0f}%  "
                              f"stall={c['stall']:.1f}s  "
                              # Peaks since start, not windowed: they are running maxima, and
                              # diffing two maxima under-reports a window whose worst is below
                              # the all-time worst. A peak that stops growing is the signal.
                              f"peak_block={c['worst_ms']:.0f}ms  peak_lag={c['lag']:.2f}s")
                    p_out[key] = c
                n, t_win = 0, time.time()
                p_enh, p_rec, p_fire, p_sr, p_enc = c_enh, c_rec, c_fire, c_sr, c_enc
                p_wait = dict(wait)
        if recorder:
            recorder.close()

    def enhance_loop():
        """GPU: newest captured frame -> enhanced frame, fanned out to output and recognition."""
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
            s_enh.add(time.time() - t0)
            # Recognition before the handoff, so the event and the frame it was computed
            # from stay together; it reads the enhanced pre-SR frame either way.
            ev = recognize(chunk[0], t_cap) if recognizer else None
            out_q.put((chunk[0], t_cap, ev))

    # --- multi-GPU. Unused when there is one device, which keeps the single-GPU path above
    # byte-for-byte the one that was measured and shipped.
    in_q = [queue.Queue(maxsize=1) for _ in devices]     # dealer -> worker k
    done_q = [queue.Queue(maxsize=1) for _ in devices]   # worker k>0 -> worker 0
    # Room in in_q[k], acquired by the dealer and released by worker k when it takes the
    # frame. Same purpose as `space` above -- do not read st.raw until there is somewhere to
    # put the frame -- but one slot deeper in effect, because the release happens on the get
    # rather than after the enhance. That gives each worker a one-frame prefetch, which is
    # what stops the strict rotation from stalling a free worker while the dealer is blocked
    # on a busy one. It costs freshness: a frame can sit in in_q[k] for up to one enhance
    # period (~55 ms at 720) before anyone touches it. Cheap against a 1 s budget, and the
    # alternative -- releasing after the enhance -- idles a whole GPU to save it.
    slot = [threading.Semaphore(1) for _ in devices]

    def put_until(q, item, bill=None):
        """Blocking put that still notices shutdown. False if the pipeline is going down."""
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
                wait[bill] += time.time() - t_wait

    def enhance_on(k, frame):
        t0 = time.time()
        chunk = [frame]
        for s in enh_sets[k]:
            chunk = s(chunk)
        s_enh.add(time.time() - t0)
        return chunk[0]

    def dealer():
        """Read the newest frame and hand it to the next device, strictly round-robin.

        Strict rotation is what lets worker 0 reorder with `next_out % n_gpu` instead of a
        reorder buffer keyed by sequence number. The cost is that one slow device throttles
        the rotation for everybody, which is fine while the devices are homogeneous -- and
        shows up as wait_peer in the telemetry if that ever stops being true.
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
            while running():
                try:
                    frame, t_cap = in_q[k].get(timeout=0.1)
                except queue.Empty:
                    continue
                slot[k].release()
                # Billed to wait_peer: blocking here means devices[0] has not come round to
                # collect, so this card is idle waiting on another card. That is the same
                # imbalance the sequencer's done_q wait measures, seen from the other side --
                # which side shows it just says who is ahead.
                put_until(done_q[k], (enhance_on(k, frame), t_cap), bill="peer")
        loop.__name__ = f"gpu_worker{k}"
        return loop

    def sequencer():
        """devices[0]: its share of the enhancing, all of the recognition, and the ordering.

        Recognition sits here rather than on a thread of its own for the reason measured
        above -- a second thread issuing CUDA work on a device that already has one makes
        both slower. Folding it into THIS worker rather than a peer is also what makes
        ordering free: this loop already walks the output sequence, so `next_out % n_gpu`
        names the device that owns the next frame and no reorder buffer is needed.
        """
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
            else:
                t_wait = time.time()
                try:
                    enhanced, t_cap = done_q[k].get(timeout=0.1)
                    wait["peer"] += time.time() - t_wait
                except queue.Empty:
                    wait["peer"] += time.time() - t_wait
                    continue
            ev = recognize(enhanced, t_cap) if recognizer else None
            # wait_cpu, same meaning as in enhance_loop: GPU work blocked behind the encoder.
            if not put_until(out_q, (enhanced, t_cap, ev), bill="cpu"):
                return
            next_out += 1

    if n_gpu == 1:
        workers, main_loop = [worker(output_loop)], enhance_loop
    else:
        workers = ([worker(output_loop), worker(dealer)]
                   + [worker(gpu_worker(k)) for k in range(1, n_gpu)])
        main_loop = sequencer
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


def build_app(cfg, on_clip=None):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, StreamingResponse

    st = ServerState(cfg, on_clip=on_clip)
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
            print(f"[push] {' '.join(st.push._cmd)}")
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
        body["flv_clients"] = st.flv_clients
        body["stream_bitrate"] = cfg.stream_bitrate
        # An ffmpeg that died takes its format down silently otherwise -- the endpoint keeps
        # answering, it just never produces bytes. Report liveness per output, with ffmpeg's
        # own last words when it is gone.
        for key, out in (("hls", st.hls), ("push", st.push)):
            if out is not None:
                body[f"{key}_alive"] = out.alive()
                body[f"{key}_restarts"] = out.restarts
                err = out.error_tail()
                if err:
                    body[f"{key}_error"] = err
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
    async def live_flv():
        """HTTP-FLV — the same shape the GB28181 gateways serve on the input side.

        One ffmpeg per viewer (see streams.flv_pipe for why), so the client count is capped
        rather than left to take the box down under a crowd.
        """
        from fastapi.responses import Response
        if "flv" not in st.formats:
            return Response("flv 未在 stream_formats 中启用", status_code=404,
                            media_type="text/plain; charset=utf-8")
        with st.flv_lock:
            if st.flv_clients >= cfg.max_flv_clients:
                return Response(f"FLV 并发观看数已达上限 {cfg.max_flv_clients}", status_code=503,
                                media_type="text/plain; charset=utf-8")
            st.flv_clients += 1
        try:
            out = streams.flv_pipe(st.jpeg, cfg.max_stream_fps, cfg.stream_bitrate)
        except Exception as e:                                   # noqa: BLE001
            with st.flv_lock:
                st.flv_clients -= 1
            return Response(f"无法启动 FLV 编码: {e}", status_code=503,
                            media_type="text/plain; charset=utf-8")

        async def gen():
            try:
                while True:
                    # read1, not read: read(n) waits for the full n bytes, which holds finished
                    # frames back until the buffer fills. read1 forwards whatever ffmpeg has
                    # already produced -- the difference is latency, which is the whole point
                    # of choosing FLV over HLS.
                    chunk = await asyncio.to_thread(out.proc.stdout.read1, 65536)
                    if not chunk:
                        break
                    yield chunk
            finally:
                out.close()
                with st.flv_lock:
                    st.flv_clients -= 1

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


def run_server(cfg, on_clip=None, stop_after=0.0):
    """Blocks until stopped. stop_after > 0 bounds the run (0 = until killed).

    The bound exists because a persistent service is otherwise untestable end-to-end and
    unschedulable as a finite job: with it, the same code path a camera runs forever can be
    run for 60 seconds in CI or by an orchestrator that needs the container to terminate.
    """
    import uvicorn
    app = build_app(cfg, on_clip=on_clip)
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
