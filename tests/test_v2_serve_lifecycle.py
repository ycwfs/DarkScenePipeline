"""Regression coverage for the deployed v2 package, isolated from root darkpipe imports."""
import asyncio
import importlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
OP = Path(os.environ.get("DARKPIPE_V2_OPERATOR_DIR", ROOT / "platform" / "dark-demo-v2"))


@pytest.fixture(scope="module")
def v2():
    name = "_darkpipe_v2_lifecycle"
    spec = importlib.util.spec_from_file_location(
        name, OP / "darkpipe" / "__init__.py",
        submodule_search_locations=[str(OP / "darkpipe")])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return importlib.import_module(f"{name}.server")


def config(v2, **kw):
    module = importlib.import_module(f"{v2.__package__}.config")
    return module.PipelineConfig(mode="serve", input="http://probe/live", device="cpu",
                                 enhance="off", sr="off", recognize="off", clip_dir="", **kw)


def health(app):
    route = next(r for r in app.routes if getattr(r, "path", None) == "/health")
    response = route.endpoint()
    return response.status_code, json.loads(response.body)


def endpoint(app, path):
    return next(r.endpoint for r in app.routes if getattr(r, "path", None) == path)


@pytest.mark.parametrize("fault,gpus", [
    ("load", ""), ("warmup", ""), ("enhance", ""), ("output", ""),
    ("recognize", "0,1"), ("secondary", "0,1,2"), ("system_exit", ""),
])
def test_processing_failure_degrades_health_and_closes_models(v2, monkeypatch, fault, gpus):
    made = []

    class Stage:
        def __init__(self, post=False, secondary=False):
            self.post_recognition, self.secondary = post, secondary
            self.closed = False
            self.loaded = False
            made.append(self)

        def load(self, device):
            self.loaded = True
            if fault == "load" and not self.post_recognition:
                raise RuntimeError("injected load failure")

        def __call__(self, frames):
            if fault == "enhance" and not self.post_recognition:
                raise RuntimeError("injected enhancement failure")
            if fault == "output" and self.post_recognition:
                raise RuntimeError("injected output failure")
            if fault == "secondary" and self.secondary:
                raise RuntimeError("injected secondary failure")
            if fault == "system_exit" and self.post_recognition:
                raise SystemExit("injected thread exit")
            return frames

        def close(self):
            self.closed = True

    class Recognizer:
        def __init__(self):
            self.loaded, self.closed = False, False
            made.append(self)

        def load(self, device):
            self.loaded = True

        def warmup(self):
            if fault == "warmup":
                raise RuntimeError("injected warmup failure")

        def push(self, *a, **kw):
            if fault == "recognize":
                raise RuntimeError("injected recognition failure")

        def timings(self):
            return {}

        def close(self):
            self.closed = True

    builds = []

    def stages(cfg):
        secondary = bool(builds)
        builds.append(1)
        return [Stage(secondary=secondary), Stage(post=True)], Recognizer()

    monkeypatch.setattr(v2, "build_stages", stages)
    monkeypatch.setattr(v2, "bind_thread_device", lambda dev: None)
    app = v2.build_app(config(v2, gpus=gpus))
    st = app.state.dark
    st.capture_alive = True
    process = threading.Thread(target=v2.process_loop, args=(st,), daemon=True)
    process.start()
    deadline = time.monotonic() + 3.0
    while process.is_alive() and time.monotonic() < deadline:
        st.raw.put((np.zeros((48, 64, 3), np.uint8), time.time()))
        time.sleep(0.01)
    process.join(timeout=0.5)
    assert not process.is_alive(), "processing failure left the pipeline running"
    assert st.stop.is_set()
    assert st.processing_error and not st.processing_alive
    code, body = health(app)
    assert code == 503 and body["status"] == "degraded"
    assert body["processing_status"] == "failed"
    assert body["processing_error"] == st.processing_error
    assert all(obj.closed for obj in made if isinstance(obj, Stage) and obj.loaded)
    assert all(obj.closed for obj in made if isinstance(obj, Recognizer) and obj.loaded)


def test_health_requires_ready_processing_and_recent_output(v2):
    app = v2.build_app(config(v2))
    st = app.state.dark
    st.capture_alive = True
    assert health(app)[0] == 503
    st.processing_alive = st.processing_ready = True
    assert health(app)[0] == 503
    st.last_frame_at = time.time()
    assert health(app)[0] == 200
    st.last_frame_at -= v2.PROCESS_STALE_SECONDS + 1
    assert health(app)[0] == 503
    st.last_frame_at = time.time()
    st.request_stop()
    code, body = health(app)
    assert code == 503 and body["processing_status"] == "stopping"


def test_event_bus_stop_wakes_empty_and_full_subscribers(v2):
    bus = v2.EventBus()
    empty = bus.subscribe()
    full = bus.subscribe()
    for _ in range(full.maxsize):
        full.put_nowait(object())
    bus.close()
    assert empty.get(timeout=0.1) is v2._BUS_CLOSED
    assert v2._BUS_CLOSED in [full.get_nowait() for _ in range(full.qsize())]
    assert bus.subscribe().get_nowait() is v2._BUS_CLOSED
    bus.publish(object())
    bus.close()


@pytest.mark.parametrize("cancel", [False, True])
def test_idle_sse_wait_exits_promptly_and_unsubscribes(v2, cancel):
    app = v2.build_app(config(v2))
    st = app.state.dark

    async def run():
        response = await endpoint(app, "/events")()
        pending = asyncio.create_task(anext(response.body_iterator))
        for _ in range(100):
            if st.bus._subs:
                break
            await asyncio.sleep(0.005)
        assert st.bus._subs and not pending.done()
        if cancel:
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        else:
            st.request_stop()
            with pytest.raises(StopAsyncIteration):
                await asyncio.wait_for(pending, 0.5)
        assert not st.bus._subs

    start = time.monotonic()
    asyncio.run(run())
    assert time.monotonic() - start < 1.0, "SSE left a blocking q.get in the executor"


def test_mjpeg_generator_ends_on_stop(v2):
    app = v2.build_app(config(v2))
    app.state.dark.jpeg.put(b"test-jpeg")

    async def run():
        response = await endpoint(app, "/stream")()
        assert b"test-jpeg" in await anext(response.body_iterator)
        app.state.dark.request_stop()
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(response.body_iterator), 0.5)

    asyncio.run(run())


def test_transient_output_failure_only_degrades_that_output(v2):
    app = v2.build_app(config(v2))
    st = app.state.dark
    st.capture_alive = st.processing_alive = st.processing_ready = True
    st.last_frame_at = time.time()

    class Output:
        restarts = 1

        def alive(self):
            return False

        def blocked_s(self):
            return 0.0

        def snap(self):
            return {"kills": 0}

        def error_tail(self):
            return "temporary reconnect"

    st.push = Output()
    code, body = health(app)
    assert code == 200 and body["status"] == "degraded"
    assert not st.stop.is_set() and st.processing_error is None


def _port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _launch(tmp_path, *, seconds=3, fault=""):
    port = _port()
    log = (tmp_path / "operator.log").open("wb")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
    process = subprocess.Popen([
        sys.executable, str(ROOT / "tests" / "v2_serve_probe.py"),
        "--op-dir", str(OP), "--port", str(port), "--work-dir", str(tmp_path),
        "--seconds", str(seconds), "--fault", fault,
    ], stdout=log, stderr=subprocess.STDOUT, env=env)
    return process, log, f"http://127.0.0.1:{port}"


def _ready(process, base, tmp_path):
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and process.poll() is None:
        try:
            with urlopen(base + "/health", timeout=0.2) as response:
                body = json.load(response)
            if body["status"] == "ok":
                return
        except (HTTPError, URLError, TimeoutError):
            pass
        time.sleep(0.02)
    pytest.fail("operator did not become healthy:\n" + (tmp_path / "operator.log").read_text())


def _finish(process, log):
    if process.poll() is None:
        process.kill()
        process.wait(timeout=5)
    log.close()


@pytest.mark.parametrize("fault", ["", "enhance", "recognize"])
def test_real_http_connections_close_and_session_finishes(tmp_path, fault):
    process, log, base = _launch(tmp_path, fault=fault, seconds=3 if not fault else 20)
    connections = []
    try:
        _ready(process, base, tmp_path)
        for path in ("/events", "/stream", "/live.flv"):
            connections.append(urlopen(base + path, timeout=5))
        assert connections[1].read(16).startswith(b"--frame")
        assert connections[2].read(3) == b"FLV"
        if fault:
            (tmp_path / "fail-now").touch()
        process.wait(timeout=8)
        payload = json.loads((tmp_path / "session.json").read_text())
        assert process.returncode == (1 if fault else 0)
        assert payload["status"] == ("failed" if fault else "stopped")
        assert payload["events_total"] > 0
        assert "clips" in payload and "event_log_stats" in payload
        local = Path(payload["clip_dir"]) / "session.json"
        assert json.loads(local.read_text())["status"] == payload["status"]
        if fault:
            assert "probe" in payload["error"]
        else:
            assert "error" not in payload
        # Open clients remain owned here until AFTER the process has stopped.
        assert "Cancel " not in (tmp_path / "operator.log").read_text()
    finally:
        for connection in connections:
            connection.close()
        _finish(process, log)


@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGINT])
def test_signal_ends_sse_and_writes_stopped_session(tmp_path, stop_signal):
    process, log, base = _launch(tmp_path, seconds=0)
    connection = None
    try:
        _ready(process, base, tmp_path)
        connection = urlopen(base + "/events", timeout=5)
        process.send_signal(stop_signal)
        process.wait(timeout=8)
        assert process.returncode == 0
        assert json.loads((tmp_path / "session.json").read_text())["status"] == "stopped"
    finally:
        if connection:
            connection.close()
        _finish(process, log)


@pytest.mark.parametrize("fault,status,code", [("load", "failed", 1), ("waiting", "stopped", 0)])
def test_early_failure_or_timer_interrupts_startup_wait(tmp_path, fault, status, code):
    process, log, _ = _launch(tmp_path, seconds=0.5, fault=fault)
    try:
        process.wait(timeout=10)
        timing = json.loads((tmp_path / "probe-timing.json").read_text())
        assert timing["operator_seconds"] < 3.0
        assert process.returncode == code
        payload = json.loads((tmp_path / "session.json").read_text())
        assert payload["status"] == status
    finally:
        _finish(process, log)


def test_connections_waiting_for_first_frame_exit_on_timer(tmp_path):
    process, log, base = _launch(tmp_path, seconds=3, fault="no_frames")
    connections = []
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                urlopen(base + "/health", timeout=0.2).close()
            except HTTPError as exc:
                body = json.load(exc)
                exc.close()
                assert body["capture_alive"] and body["frame_age_s"] is None
                break
            except (URLError, TimeoutError):
                time.sleep(0.02)
        else:
            pytest.fail("waiting operator never listened")
        for path in ("/events", "/stream", "/live.flv"):
            connections.append(urlopen(base + path, timeout=5))
        process.wait(timeout=8)
        assert process.returncode == 0
        assert json.loads((tmp_path / "session.json").read_text())["status"] == "stopped"
        assert "Cancel " not in (tmp_path / "operator.log").read_text()
    finally:
        for connection in connections:
            connection.close()
        _finish(process, log)


def test_nonreading_client_has_bounded_graceful_shutdown(tmp_path):
    process, log, base = _launch(tmp_path, seconds=2, fault="backpressure")
    client = socket.socket()
    try:
        _ready(process, base, tmp_path)
        client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
        client.connect(("127.0.0.1", int(base.rsplit(":", 1)[1])))
        client.sendall(b"GET /stream HTTP/1.1\r\nHost: localhost\r\n\r\n")
        # Keep the socket open without reading, even after the timer expires.
        process.wait(timeout=10)
        assert process.returncode == 0
        payload = json.loads((tmp_path / "session.json").read_text())
        assert payload["status"] == "stopped"
        assert "Cancel " in (tmp_path / "operator.log").read_text()
        timing = json.loads((tmp_path / "probe-timing.json").read_text())
        assert timing["operator_seconds"] < 9.0
    finally:
        client.close()
        _finish(process, log)


@pytest.mark.parametrize('stalled', [False, True])
def test_shutdown_allows_slow_model_cleanup_but_bounds_a_stall(v2, monkeypatch, stalled):
    release, exited = threading.Event(), threading.Event()
    monkeypatch.setattr(v2, 'PROCESS_STOP_TIMEOUT', .05 if stalled else 6.5, raising=False)

    def capture(st):
        st.src_fps_known.set()
        st.stop.wait()

    def process(st):
        try:
            st.stop.wait()
            if stalled:
                release.wait(10)
            else:
                # Model cleanup plus worker joins can legitimately exceed the old outer 5 s.
                time.sleep(5.2)
        finally:
            exited.set()

    monkeypatch.setattr(v2, 'capture_loop', capture)
    monkeypatch.setattr(v2, 'process_loop', process)
    monkeypatch.setattr(v2, '_encode_ceiling_probe', lambda *args: None)
    app = v2.build_app(config(v2, stream_formats='mjpeg'))
    state = app.state.dark

    async def run():
        async with app.router.lifespan_context(app):
            assert not state.processing_error

    try:
        asyncio.run(run())
        if stalled:
            assert state.processing_error and 'shutdown' in state.processing_error
            assert not exited.is_set()
        else:
            assert exited.is_set()
            assert state.processing_error is None
    finally:
        release.set()
        assert exited.wait(7)
