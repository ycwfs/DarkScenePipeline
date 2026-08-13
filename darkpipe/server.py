"""Online streaming-inference server (FastAPI).

CaptureThread: cv2.VideoCapture -> single-slot latest-frame buffer (newest overwrites =
implicit drop policy when the GPU is slower than the stream; bounded latency).
ProcessThread: owns the GPU; enhance -> recognizer.push -> SR -> label bar -> JPEG.
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


def process_loop(st: ServerState):
    cfg = st.cfg
    frame_stages, recognizer = build_stages(cfg)
    for s in frame_stages:
        s.load(cfg.device)
    if recognizer:
        recognizer.load(cfg.device)
    # Split on the stage's own declaration, not on its name. Recognition sees the enhanced
    # pre-SR frame; anything flagged post_recognition only changes the picture.
    srs = [s for s in frame_stages if getattr(s, "post_recognition", False)]
    enh = [s for s in frame_stages if not getattr(s, "post_recognition", False)]
    recorder = None
    last_seq = 0
    n, t_win = 0, time.time()
    idx = 0
    while not st.stop.is_set():
        item, seq = st.raw.get()
        if item is None or seq == last_seq:
            time.sleep(0.002)
            continue
        last_seq = seq
        frame, t_cap = item
        chunk = [frame]
        for s in enh:
            chunk = s(chunk)
        ev = None
        if recognizer:
            ev = recognizer.push(chunk[0], idx, time.time() - st.t_start)
            if ev:
                st.last_event = ev
                st.events_total += 1
                st.bus.publish(ev)
                if st.eventlog is not None:
                    st.eventlog.write(ev)
        for s in srs:
            chunk = s(chunk)
        out = chunk[0]
        if recognizer and not cfg.no_label_bar:
            # The configured stream rate, not the measured processing rate. These are
            # different quantities and the burnt-in one should be the viewer's: the feeder
            # resamples the JPEG slot onto a fixed max_stream_fps cadence (repeating the last
            # frame when the pipeline is slower), so that is genuinely the rate the stream is
            # delivered at. The processing rate stays reported, but in /health as fps_proc,
            # where it is a diagnostic rather than a number on a monitoring wall.
            out = append_label_bar(out, st.last_event,
                                   extra=f"{cfg.max_stream_fps:g} fps")
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
        if ok:
            st.jpeg.put(buf.tobytes())
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
        idx += 1
        n += 1
        if time.time() - t_win >= 2.0:
            st.fps_proc = n / (time.time() - t_win)
            n, t_win = 0, time.time()
    if recorder:
        recorder.close()
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
