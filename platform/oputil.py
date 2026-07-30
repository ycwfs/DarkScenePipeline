"""算子公共工具：平台参数解析、outputPath 父目录创建、共用文件库地址落盘。

两个算子（处理算子、测试验证算子）各自打包时都会把本文件放到 main.py 同级目录，因此两个
zip 都是自包含的，不依赖对方。
"""
import argparse
import os
import shutil
import subprocess
import sys

BOOL_TRUE = ("1", "true", "t", "yes", "y", "on", "是")
BOOL_FALSE = ("0", "false", "f", "no", "n", "off", "否")


def parse_bool(v):
    """平台 bool 组件可能下发 true/false、1/0 或空串，统一容错解析。"""
    s = str(v).strip().lower()
    if s in BOOL_TRUE:
        return True
    if s in BOOL_FALSE or s == "":
        return False
    raise argparse.ArgumentTypeError(f"不是合法的布尔值: {v!r}")


def ensure_parent(path):
    """规范要求：outputPath 是框架下发的绝对文件路径，父目录须由算子代码创建。"""
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    return path


def fetch_input(src, workdir, default_name="input.bin"):
    """把共用/模型文件库地址取到本地，返回本地可读路径。

    文件库下发的地址形如 hdfs://用户名@ip:port/a/b/c.mp4；OpenCV 与 open() 都打不开，必须
    先落盘。rtsp:// http(s):// 及本地路径原样返回（cv2.VideoCapture 可直接处理）。
    """
    if not src.startswith("hdfs://"):
        return src
    exe = shutil.which("hdfs") or shutil.which("hadoop")
    if not exe:
        sys.exit("error: 输入为 hdfs:// 地址，但容器内没有 hdfs/hadoop 客户端，无法下载。"
                 "请在镜像中安装 hadoop 客户端，或由框架把文件挂载为本地路径后再传入。")
    os.makedirs(workdir, exist_ok=True)
    local = os.path.join(workdir, os.path.basename(src.split("?")[0]) or default_name)
    cmd = ([exe, "dfs", "-get", src, local] if os.path.basename(exe) == "hdfs"
           else [exe, "fs", "-get", src, local])
    print(f"[input] 下载 {src} -> {local}")
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0 or not os.path.exists(local):
        sys.exit(f"error: 下载失败(exit={r.returncode}): {(r.stdout or '').strip()[-800:]}")
    return local


def parse_gpu_ids(value):
    """'0' 或 '0,1,2' -> ['0','1','2']；非法值直接退出。"""
    ids = [g.strip() for g in str(value).split(",") if g.strip()]
    if not ids:
        return ["0"]
    if not all(g.isdigit() for g in ids):
        sys.exit(f"error: gpu_ids 需为逗号分隔的显卡序号，如 '0' 或 '0,1,2'（收到 {value!r}）")
    if len(set(ids)) != len(ids):
        sys.exit(f"error: gpu_ids 存在重复序号（收到 {value!r}）")
    return ids
