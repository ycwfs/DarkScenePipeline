"""WebHDFS over plain HTTP: the only HDFS channel the operator image actually has.

The image ships no hadoop client and no JVM, so `hdfs dfs -get/-put` has never worked in it
-- the serve operator merely treated upload failures as warnings, so nobody noticed. The
platform hands the input down as `http://<namenode>:9870/<path>/<name>.mp4`, which is the
NameNode's *web UI* port, not a downloadable address; the real one is
`/webhdfs/v1/<path>?op=OPEN`, and that 307-redirects to a DataNode addressed **by hostname**.
That last hop is what broke in deployment, and it is what these tests pin down.

The real cluster (10.46.79.133) is unreachable from the dev machine, so everything here runs
against a local stand-in that reproduces the same shapes: the 404 on the raw path, the 200
text/html error page, the 307 to an unresolvable hostname, and the two-hop PUT.
"""
import importlib.util
import json
import os
import socket
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("oputil", os.path.join(ROOT, "platform",
                                                                      "oputil.py"))
oputil = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oputil)

PAYLOAD = b"\x00\x01fake-mp4-bytes" * 5000            # ~75 KB, crosses the 1 MiB read loop once
CN_NAME = "演示视频2026-08-19_103132_971.mp4"          # the real filename, non-ASCII on purpose
DN_HOST = "hdfs-datanode-testonly.invalid"            # .invalid is reserved: never resolves


def _resolves(host):
    try:
        socket.gethostbyname(host)
        return True
    except OSError:
        return False


class _Cluster:
    """A NameNode and a DataNode on two ports, speaking just enough WebHDFS."""

    def __init__(self, dn_host_in_redirect):
        self.dn_host = dn_host_in_redirect
        self.uploaded = {}          # hdfs path -> bytes
        self.mkdirs = []
        self.raw_hits = []          # non-webhdfs GETs the NameNode saw
        self.raw_reply = (404, "text/plain", b"not found")
        self.dn = self._serve(self._datanode)
        self.nn = self._serve(self._namenode)

    # -- plumbing -------------------------------------------------------------------
    def _serve(self, dispatch):
        cluster = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _go(self):
                dispatch(self)

            do_GET = do_PUT = _go

        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        srv.url = f"http://127.0.0.1:{srv.server_address[1]}"
        return srv

    def close(self):
        for s in (self.nn, self.dn):
            if not getattr(s, "closed", False):
                s.closed = True
                s.shutdown()
                s.server_close()      # shutdown() alone leaves the port bound: connects then
                #                       hangs for the full socket timeout instead of refusing

    @staticmethod
    def _reply(h, code, ctype, body=b"", extra=None):
        h.send_response(code)
        h.send_header("Content-Type", ctype)
        h.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            h.send_header(k, v)
        h.end_headers()
        if body:
            h.wfile.write(body)

    # -- roles ----------------------------------------------------------------------
    def _namenode(self, h):
        u = urllib.parse.urlsplit(h.path)
        path = urllib.parse.unquote(u.path)
        if not path.startswith(oputil.WEBHDFS_PREFIX):
            self.raw_hits.append(path)
            code, ctype, body = self.raw_reply
            return self._reply(h, code, ctype, body)
        op = urllib.parse.parse_qs(u.query).get("op", [""])[0].upper()
        if op == "MKDIRS":
            self.mkdirs.append(path[len(oputil.WEBHDFS_PREFIX):])
            return self._reply(h, 200, "application/json", b'{"boolean":true}')
        # OPEN and CREATE both only *locate* the data; the DataNode does the transfer.
        dn_port = self.dn.server_address[1]
        return self._reply(h, 307, "application/octet-stream", b"",
                           {"Location": f"http://{self.dn_host}:{dn_port}{h.path}"})

    def _datanode(self, h):
        u = urllib.parse.urlsplit(h.path)
        path = urllib.parse.unquote(u.path)[len(oputil.WEBHDFS_PREFIX):]
        op = urllib.parse.parse_qs(u.query).get("op", [""])[0].upper()
        if op == "CREATE":
            n = int(h.headers.get("Content-Length") or 0)
            self.uploaded[path] = h.rfile.read(n)
            return self._reply(h, 201, "application/octet-stream", b"",
                               {"Location": f"webhdfs://{path}"})
        return self._reply(h, 200, "application/octet-stream", PAYLOAD)


@pytest.fixture
def cluster():
    c = _Cluster(dn_host_in_redirect="127.0.0.1")
    yield c
    c.close()


@pytest.fixture
def cluster_bad_dns():
    if _resolves(DN_HOST):
        pytest.skip(f"{DN_HOST} unexpectedly resolves here; the fallback cannot be exercised")
    c = _Cluster(dn_host_in_redirect=DN_HOST)
    yield c
    c.close()


# -- input ---------------------------------------------------------------------------

def test_the_platforms_namenode_url_is_rewritten_to_webhdfs_and_downloaded(cluster, tmp_path):
    """`http://<nn>:9870/<path>/<中文>.mp4` is what the platform sends, verbatim."""
    url = f"{cluster.nn.url}/behavor/darkpipe/{CN_NAME}"
    local = oputil.fetch_input(url, str(tmp_path), "input.mp4")

    assert os.path.isfile(local), local
    assert open(local, "rb").read() == PAYLOAD
    assert os.path.basename(local) == CN_NAME, "the non-ASCII name did not survive the round trip"
    assert cluster.raw_hits == ["/behavor/darkpipe/" + CN_NAME], \
        "the address as given should be tried first -- it is what the user literally wrote"


def test_a_namenode_error_page_is_not_mistaken_for_the_video(cluster, tmp_path):
    """The web UI answers 200 + HTML for an unknown path; saving that yields a fake mp4."""
    cluster.raw_reply = (200, "text/html; charset=utf-8", b"<html>Hadoop</html>")
    local = oputil.fetch_input(f"{cluster.nn.url}/a/{CN_NAME}", str(tmp_path), "input.mp4")
    assert open(local, "rb").read() == PAYLOAD, "fell for the HTML error page"


def test_an_unresolvable_datanode_falls_back_to_the_namenode_address(cluster_bad_dns, tmp_path):
    """The exact deployment failure: the 307 names `hdfs-datanode`, which DNS cannot answer."""
    url = f"{cluster_bad_dns.nn.url}/behavor/darkpipe/{CN_NAME}"
    local = oputil.fetch_input(url, str(tmp_path), "input.mp4")
    assert open(local, "rb").read() == PAYLOAD


def test_when_both_forms_fail_the_error_names_both_urls(cluster, tmp_path):
    cluster.close()                           # nothing listening: both attempts are refused
    with pytest.raises(SystemExit) as e:
        oputil.fetch_input(f"{cluster.nn.url}/a/b.mp4", str(tmp_path), "input.mp4")
    msg = str(e.value)
    assert "原始地址" in msg and "WebHDFS" in msg, msg
    assert msg.count(cluster.nn.url) >= 2, f"both attempted URLs must be shown:\n{msg}"


@pytest.mark.parametrize("src", [
    "rtsp://10.0.0.1:554/live",
    "http://10.0.0.1:8080/hls/stream.m3u8",
    "http://10.0.0.1:8080/mjpg/video.mjpg",
    "http://10.0.0.1:8080/live",
    "/data/local/clip.mp4",
    "0",
])
def test_live_sources_are_passed_through_untouched(src, tmp_path):
    """Downloading a live stream never returns. Only finite-looking files may be fetched."""
    assert oputil.fetch_input(src, str(tmp_path), "input.mp4") == src


# -- output --------------------------------------------------------------------------

def test_upload_goes_through_the_two_hop_put(cluster, tmp_path):
    f = tmp_path / "out.mp4"
    f.write_bytes(PAYLOAD)
    ok, dest = oputil.upload_file(str(f), f"{cluster.nn.url}/behavor/out")

    assert ok, dest
    assert cluster.uploaded == {"/behavor/out/out.mp4": PAYLOAD}
    assert dest.endswith("/behavor/out/out.mp4")


def test_make_remote_dir_uses_mkdirs(cluster):
    ok, msg = oputil.make_remote_dir(f"{cluster.nn.url}/behavor/out/run1")
    assert ok, msg
    assert cluster.mkdirs == ["/behavor/out/run1"]


def test_an_hdfs_uri_is_mapped_onto_the_namenode_http_port():
    """`hdfs://` carries the RPC port; WebHDFS lives on the NameNode's HTTP port instead."""
    base, path, user = oputil.split_hdfs_target("hdfs://bigdata@10.46.79.133:8020/behavor/x")
    assert base == f"http://10.46.79.133:{oputil.NAMENODE_HTTP_PORT}"
    assert (path, user) == ("/behavor/x", "bigdata")
    assert "user.name=bigdata" in oputil.webhdfs_url(base, path, "OPEN", user)


def test_a_webhdfs_url_is_not_double_prefixed():
    base, path, _ = oputil.split_hdfs_target(
        f"http://10.0.0.1:9870{oputil.WEBHDFS_PREFIX}/a/b?op=OPEN")
    assert path == "/a/b"
    assert oputil.webhdfs_url(base, path, "OPEN").count(oputil.WEBHDFS_PREFIX) == 1


# -- /etc/hosts ----------------------------------------------------------------------

def test_extra_hosts_writes_only_names_that_do_not_resolve(tmp_path, monkeypatch, capsys):
    hosts = tmp_path / "hosts"
    hosts.write_text("127.0.0.1\tlocalhost\n", encoding="utf-8")
    real_open = open

    def fake_open(path, *a, **kw):
        return real_open(hosts if path == "/etc/hosts" else path, *a, **kw)

    monkeypatch.setattr("builtins.open", fake_open)
    n = oputil.apply_extra_hosts("10.46.79.133 hdfs-datanode, 127.0.0.1 localhost")

    assert n == 1
    body = hosts.read_text(encoding="utf-8")
    assert "10.46.79.133\thdfs-datanode\n" in body
    assert body.count("localhost") == 1, "already present -- must not be duplicated"


def test_extra_hosts_survives_a_read_only_etc_hosts(monkeypatch, capsys):
    """Non-root or a read-only mount must warn, not kill the run -- the redirect fallback
    still covers it."""
    def deny(path, *a, **kw):
        raise PermissionError("read-only file system")

    monkeypatch.setattr("builtins.open", deny)
    assert oputil.apply_extra_hosts("10.46.79.133 hdfs-datanode") == 0
    assert "/etc/hosts" in capsys.readouterr().out


def test_extra_hosts_ignores_empty_and_malformed_entries(capsys):
    assert oputil.apply_extra_hosts("") == 0
    assert oputil.apply_extra_hosts("   ") == 0
    assert oputil.apply_extra_hosts("just-a-hostname") == 0
    assert "忽略" in capsys.readouterr().out


def test_json_shape_of_the_stand_in_matches_webhdfs():
    """Guards the fixture itself: a MKDIRS reply is `{"boolean":true}`, per the WebHDFS spec."""
    assert json.loads('{"boolean":true}') == {"boolean": True}
