"""Online streaming-inference server (FastAPI).

CaptureThread: cv2.VideoCapture -> single-slot latest-frame buffer (newest overwrites =
implicit drop policy when the GPU is slower than the stream; bounded latency).
ProcessThread: owns the GPU; enhance -> recognizer.push -> SR -> label bar -> JPEG.
asyncio endpoints only read the JPEG slot / subscribe to the event bus:
  GET /stream  multipart MJPEG   GET /events  SSE recognition JSON
  GET /health  live counters     GET /config  active configuration
"""
import asyncio
import json
import queue
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
    def __init__(self, cfg):
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
    enh = [s for s in frame_stages if s.name.startswith("enhance")]
    srs = [s for s in frame_stages if s.name.startswith("sr")]
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
        if recognizer:
            ev = recognizer.push(chunk[0], idx, time.time() - st.t_start)
            if ev:
                st.last_event = ev
                st.bus.publish(ev)
        for s in srs:
            chunk = s(chunk)
        out = chunk[0]
        if recognizer and not cfg.no_label_bar:
            out = append_label_bar(out, st.last_event, extra=f"{st.fps_proc:.1f} fps")
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
        if ok:
            st.jpeg.put(buf.tobytes())
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


def build_app(cfg):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, StreamingResponse

    st = ServerState(cfg)

    @asynccontextmanager
    async def lifespan(app):
        threads = [threading.Thread(target=capture_loop, args=(st,), daemon=True),
                   threading.Thread(target=process_loop, args=(st,), daemon=True)]
        for t in threads:
            t.start()
        yield
        st.stop.set()
        for t in threads:
            t.join(timeout=5)

    app = FastAPI(title="darkpipe", lifespan=lifespan)

    @app.get("/health")
    def health():
        body = dict(status="ok" if st.capture_alive else "degraded",
                    uptime_s=round(time.time() - st.t_start, 1),
                    capture_alive=st.capture_alive, source=str(cfg.input),
                    reconnects=st.reconnects, fps_in=round(st.fps_in, 2),
                    fps_proc=round(st.fps_proc, 2),
                    frames_dropped=max(0, st.frames_in - st.frames_proc),
                    latency_ms_last=round(st.latency_ms, 1))
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


def run_server(cfg):
    import uvicorn
    app = build_app(cfg)
    print(f"[serve] http://{cfg.host}:{cfg.port}  endpoints: /stream /events /health /config")
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="warning")
