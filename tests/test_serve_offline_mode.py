"""The serve operator picks its own mode; these tests pin the decision and its consequences.

One operator now covers two shapes of work: a resident stream service, and a one-shot offline
run over a finite file (which is what an HDFS address resolves to once fetched). The whole
decision rests on a single line in `run()`, so it is worth stating in tests what each input
combination is supposed to mean -- especially the two that are easy to get backwards: a local
file WITH a push URL is still the live demo path, and an `http://…mp4` address is a file, not
a stream.
"""
import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLATFORM = os.path.join(ROOT, "platform")
OP = os.path.join(PLATFORM, "op_dark_behavior_serve")


@pytest.fixture(scope="module")
def op():
    spec = importlib.util.spec_from_file_location("_op_serve", os.path.join(OP, "main.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, PLATFORM)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(PLATFORM)
    return mod


@pytest.fixture
def spy(op, monkeypatch):
    """Replace both mode bodies with recorders: mode choice is what is under test, not GPU work."""
    calls = {}

    def offline(args, src, session, workdir):
        calls.update(mode="offline", src=src, session=session, workdir=workdir)

    def serve(args, src, session):
        calls.update(mode="serve", src=src, session=session)

    monkeypatch.setattr(op, "run_offline_mode", offline)
    monkeypatch.setattr(op, "run_serve_mode", serve)
    monkeypatch.setattr(op, "apply_extra_hosts", lambda spec: 0)
    return calls


def args_for(op, video_path, push_url="", tmp=None, **kw):
    argv = ["--video_path", video_path, "--rtmp_push_url", push_url,
            "--session_json", str((tmp or "/tmp") + "/s.json")]
    for k, v in kw.items():
        argv += [f"--{k}", str(v)]
    return op.build_parser().parse_args(argv)


def test_a_local_file_with_no_push_url_runs_offline(op, spy, tmp_path):
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"x")
    op.run(args_for(op, str(f), "", str(tmp_path)))
    assert spy["mode"] == "offline"
    assert spy["src"] == str(f)


def test_a_local_file_with_a_push_url_still_runs_the_live_service(op, spy, tmp_path):
    """The loop-a-file-and-push demo predates this change and must survive it."""
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"x")
    op.run(args_for(op, str(f), "rtmp://10.0.0.9:1935/live/k", str(tmp_path)))
    assert spy["mode"] == "serve"


@pytest.mark.parametrize("src", ["rtsp://10.0.0.1:554/live",
                                 "http://10.0.0.1:8080/hls/x.m3u8",
                                 "http://10.0.0.1:8080/live"])
def test_a_live_stream_runs_the_service_even_with_no_push_url(op, spy, tmp_path, src):
    op.run(args_for(op, src, "", str(tmp_path)))
    assert spy["mode"] == "serve"
    assert spy["src"] == src, "a stream address must reach the pipeline unchanged"


def test_a_downloaded_hdfs_input_counts_as_a_file(op, spy, monkeypatch, tmp_path):
    """`http://<namenode>:9870/…mp4` is a file: fetch_input lands it, so offline mode wins."""
    landed = tmp_path / "input.mp4"

    def fake_fetch(src, workdir, default_name="input.bin"):
        landed.write_bytes(b"video")
        return str(landed)

    monkeypatch.setattr(op, "fetch_input", fake_fetch)
    op.run(args_for(op, "http://10.46.79.133:9870/behavor/x.mp4", "", str(tmp_path)))
    assert spy["mode"] == "offline"
    assert spy["src"] == str(landed)


def test_the_service_is_fed_the_fetched_path_not_the_original_address(op, spy, monkeypatch,
                                                                     tmp_path):
    """A downloaded file + a push URL: the pipeline must open the local copy, not the URL."""
    landed = tmp_path / "input.mp4"
    landed.write_bytes(b"video")
    monkeypatch.setattr(op, "fetch_input", lambda *a, **kw: str(landed))
    op.run(args_for(op, "hdfs://u@10.0.0.1:8020/a/x.mp4", "rtmp://h/live/k", str(tmp_path)))
    assert spy["mode"] == "serve"
    assert spy["src"] == str(landed)


def test_the_download_directory_is_removed_when_the_run_ends(op, spy, tmp_path):
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"x")
    op.run(args_for(op, str(f), "", str(tmp_path)))
    assert not os.path.exists(spy["workdir"]), "the temporary download dir outlived the run"


def test_the_download_directory_is_removed_even_when_the_run_fails(op, monkeypatch, tmp_path):
    """A failed run must not leak a multi-GB download into the container's filesystem."""
    seen = {}

    def boom(args, src, session, workdir):
        seen["workdir"] = workdir
        raise RuntimeError("pipeline exploded")

    monkeypatch.setattr(op, "run_offline_mode", boom)
    monkeypatch.setattr(op, "apply_extra_hosts", lambda spec: 0)
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"x")
    with pytest.raises(RuntimeError):
        op.run(args_for(op, str(f), "", str(tmp_path)))
    assert not os.path.exists(seen["workdir"])


def test_hosts_are_applied_before_anything_is_fetched(op, monkeypatch, tmp_path):
    """The DataNode hostname has to resolve by the time the first WebHDFS hop redirects."""
    order = []
    monkeypatch.setattr(op, "apply_extra_hosts", lambda spec: order.append(("hosts", spec)))
    monkeypatch.setattr(op, "fetch_input",
                        lambda src, *a, **kw: order.append(("fetch",)) or src)
    monkeypatch.setattr(op, "run_serve_mode", lambda *a: None)
    op.run(args_for(op, "rtsp://x/live", "", str(tmp_path)))
    assert [k for k, *_ in order] == ["hosts", "fetch"]
    assert order[0][1] == "10.46.79.133 hdfs-datanode", \
        "the deployed default must reach the call without the platform filling anything in"


def test_neither_mode_mints_a_second_session_name(op):
    """One name keys clip_dir, the HDFS tree and session_json alike -- a second call to
    run_dir_name() inside a mode body would silently split them a second apart."""
    import inspect
    for fn in (op.run_offline_mode, op.run_serve_mode):
        assert "run_dir_name" not in inspect.getsource(fn), \
            f"{fn.__name__} generates its own session instead of using the one passed in"
    assert inspect.getsource(op.run).count("run_dir_name()") == 1
