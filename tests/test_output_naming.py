"""Run-directory naming: which input produced this pile of files?

Every operator lands its产出 under `<目标目录>/<会话>/`, and the session name used to be
`<时间戳>_<进程号>` — so a browsing user faced a column of `20260821_164652_78` with no way to
tell which stream or which video each one came from without opening the files. The tag comes
from the *original* address rather than the local copy, because an HDFS input is downloaded
to `input.mp4` and by then the name is already gone.
"""
import importlib.util
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("oputil", os.path.join(ROOT, "platform",
                                                                     "oputil.py"))
oputil = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oputil)

STAMP = re.compile(r"^\d{8}_\d{6}_\d+$")


@pytest.mark.parametrize("src,tag", [
    # 实时流：地址末段就是流名，这正是用户要在目录上看到的那一串
    ("rtmp://10.46.79.133:1935/live/darkpipe_input_024", "darkpipe_input_024"),
    ("rtsp://10.46.79.133:554/Streaming/Channels/101", "101"),
    # 离线视频：文件名主干，本地路径与 HDFS 地址一视同仁
    ("/mnt/nfs/videos/demo.mp4", "demo"),
    ("hdfs://hdfs@10.46.79.133:8020/behavor/darkpipe/演示视频2026-08-19_103132_971.mp4",
     "演示视频2026-08-19_103132_971"),              # 中文原样保留，目录名是给人看的
    ("http://10.46.79.133:9870/webhdfs/v1/a/b.mp4?op=OPEN&user.name=hdfs", "b"),
    ("/data/my video (1).mp4", "my_video_1"),      # 空格与括号在 NFS/shell 里都是麻烦
    ("", ""),
    ("   ", ""),
])
def test_the_tag_is_what_a_person_would_call_the_source(src, tag):
    assert oputil.source_tag(src) == tag


def test_a_host_only_address_falls_back_to_the_host_without_credentials():
    tag = oputil.source_tag("rtsp://admin:secret@10.46.79.133:554/")
    assert tag == "10.46.79.133"
    assert "secret" not in tag and "admin" not in tag, "口令不能写进目录名"


def test_the_tag_is_bounded_and_path_safe():
    tag = oputil.source_tag("/data/" + "x" * 200 + ".mp4")
    assert len(tag) == 40
    for bad in ("/", "\\", " ", ":", "?"):
        assert bad not in oputil.source_tag("/a/b/c d:e?f.mp4")


def test_live_streams_take_the_suffix_and_files_the_prefix():
    """位置不同是有意的：离线是一个目录里很多不同的文件，前缀让同一个视频排在一起；
    实时流一个流跑很久，按时间排更好找。"""
    live = oputil.run_dir_name("rtmp://ip/live/darkpipe_input_024", live=True)
    off = oputil.run_dir_name("/mnt/nfs/videos/demo.mp4")
    assert live.endswith("_darkpipe_input_024") and STAMP.match(live[:-len("_darkpipe_input_024")])
    assert off.startswith("demo_") and STAMP.match(off[len("demo_"):])


def test_an_unidentifiable_source_keeps_the_old_name():
    """取不到标识时退回原来的 时间戳_进程号，不能留下一个空的前缀或悬着的下划线。"""
    for name in (oputil.run_dir_name(), oputil.run_dir_name("", live=True),
                 oputil.run_dir_name("///")):
        assert STAMP.match(name), name


# ---------------------------------------------------------------------------
# 落地文件名。目录名说清了「哪一路输入」，文件名还得说清「哪一个产出」。
#
# 框架下发的 outputPath 是 `/tmp/outputs/<输出名>/data`：区分输出的是目录，文件名一律叫
# `data`。落地时照抄源文件名，一次运行的几个产出就会依次复制到同一个 `<run>/data` 上。
# 实测过一次：离线算子的整段视频先落地，随后被 events_json、summary_json 依次覆盖，NFS 上
# 只剩一个 15 字节的 `data`，日志却打了三行 `[done] ... -> .../data`。
# ---------------------------------------------------------------------------

def _framework_outputs(tmp_path, names):
    """按平台的真实形状造 outputPath：/tmp/outputs/<输出名>/data，内容各不相同。"""
    made = {}
    for name in names:
        d = tmp_path / "outputs" / name
        d.mkdir(parents=True)
        (d / "data").write_bytes(f"<{name}>".encode())
        made[name] = str(d / "data")
    return made


def test_outputs_that_share_the_basename_data_do_not_collapse_into_one_file(tmp_path):
    got = _framework_outputs(tmp_path, ["output_video", "events_json", "summary_json"])
    nfs = tmp_path / "nfs"
    nfs.mkdir()
    landed = oputil.deliver(
        {"output_video": (got["output_video"], "demo_enhanced.mp4"),
         "events_json": (got["events_json"], "demo_events.json"),
         "summary_json": (got["summary_json"], "demo_summary.json")},
        str(nfs), "", run="demo_20260824_120000_11")

    run_dir = nfs / "demo_20260824_120000_11"
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "demo_enhanced.mp4", "demo_events.json", "demo_summary.json"]
    # 视频还是视频，没有被后面两个 JSON 覆盖
    assert (run_dir / "demo_enhanced.mp4").read_bytes() == b"<output_video>"
    assert len(set(v[0] for v in landed.values())) == 3


def test_a_caller_that_forgets_to_name_its_outputs_still_loses_nothing(tmp_path):
    """兜底：名字没给全也不能丢文件——宁可落成 `<输出名>` 这种难看的名字。"""
    got = _framework_outputs(tmp_path, ["output_video", "events_json"])
    nfs = tmp_path / "nfs"
    nfs.mkdir()
    oputil.deliver({"output_video": got["output_video"], "events_json": got["events_json"]},
                   str(nfs), "", run="r")

    landed = sorted((nfs / "r").iterdir(), key=lambda p: p.name)
    assert len(landed) == 2, f"两个产出落成了 {len(landed)} 个文件"
    assert {p.read_bytes() for p in landed} == {b"<output_video>", b"<events_json>"}


@pytest.mark.parametrize("op", ["op_dark_behavior", "op_dark_behavior_eval"])
def test_the_batch_operators_name_every_output_they_deliver(op):
    """这两个算子交付的全部是框架 outputPath，名字必须自己给。

    实时服务算子不在此列：它交付的是自己 workdir 里 `<来源>_enhanced.mp4` 这样已经有名字的
    文件，不是 outputPath。静态检查是因为真跑一遍要 GPU，而这个错误恰恰是「跑通了、日志全绿、
    文件没了」的那一类。
    """
    import ast
    src = os.path.join(ROOT, "platform", op, "main.py")
    with open(src, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "deliver"]
    assert calls, f"{op} 没有 deliver 调用？"
    for call in calls:
        files = call.args[0]
        assert isinstance(files, ast.Dict), f"{op}: deliver 的第一个参数不是字面量字典"
        for key, value in zip(files.keys, files.values):
            assert isinstance(value, ast.Tuple), (
                f"{op}: 输出 {getattr(key, 'value', '?')} 没给落地文件名，"
                f"会与同一次运行的其他产出一起落成 <run>/data")
