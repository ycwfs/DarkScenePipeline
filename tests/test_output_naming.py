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
