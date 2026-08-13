"""算子公共工具：平台参数解析、outputPath 父目录创建、共用文件库地址落盘。

两个算子（处理算子、测试验证算子）各自打包时都会把本文件放到 main.py 同级目录，因此两个
zip 都是自包含的，不依赖对方。
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

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


def run_dir_name():
    """一次运行的专属子目录名，时间戳+进程号，避免同一个目标目录被反复写入时互相覆盖。"""
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"


def _hdfs_cli():
    """-> (可执行文件, 子命令) 或 (None, None)。hdfs 用 `hdfs dfs`，hadoop 用 `hadoop fs`。"""
    exe = shutil.which("hdfs") or shutil.which("hadoop")
    if not exe:
        return None, None
    return exe, ("dfs" if os.path.basename(exe) == "hdfs" else "fs")


def make_remote_dir(dest_dir):
    """建好目标目录。-> (成功?, 说明)。不退出进程，由调用方决定致命与否。"""
    if not dest_dir.startswith("hdfs://"):
        try:
            os.makedirs(dest_dir, exist_ok=True)
        except OSError as e:
            return False, f"创建目录 {dest_dir} 失败: {e}"
        return True, dest_dir
    exe, sub = _hdfs_cli()
    if not exe:
        return False, ("目标是 hdfs:// 地址，但容器内没有 hdfs/hadoop 客户端。"
                       "请在镜像中安装 hadoop 客户端。")
    r = subprocess.run([exe, sub, "-mkdir", "-p", dest_dir],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        return False, f"创建 hdfs 目录失败(exit={r.returncode}): {(r.stdout or '').strip()[-800:]}"
    return True, dest_dir


def upload_file(local_path, dest_dir):
    """把单个文件送到 dest_dir（须已存在）。-> (成功?, 落地地址或错误说明)。

    同样不退出进程：常驻服务不能因为一次上传失败就整体死掉，本地那份文件已经在盘上了。
    """
    if not dest_dir.startswith("hdfs://"):
        dest = os.path.join(dest_dir, os.path.basename(local_path))
        try:
            shutil.copy2(local_path, dest)
        except OSError as e:
            return False, f"复制 {local_path} -> {dest} 失败: {e}"
        return True, dest
    exe, sub = _hdfs_cli()
    if not exe:
        return False, "没有 hdfs/hadoop 客户端"
    dest = dest_dir.rstrip("/") + "/" + os.path.basename(local_path)
    r = subprocess.run([exe, sub, "-put", "-f", local_path, dest],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        return False, f"上传失败(exit={r.returncode}): {(r.stdout or '').strip()[-800:]}"
    return True, dest


def upload_outputs(files, dest_dir, run=None):
    """把 outputPath 落盘的产出文件送到 dest_dir，让容器退出后仍能取到。

    run: 本次运行的子目录名。同一次运行要落到多个目的地（本地/NFS 一份、HDFS 一份）时，
    调用方生成一次传进来，两边的子目录名才会一致——各自调用 run_dir_name() 会差出几秒，
    落成两个对不上的目录。

    outputPath 只是容器内的一个临时路径，镜像销毁后就没了；框架间 outputPath/inputPath 的
    传递也只发生在同一次编排内，覆盖不了「事后把结果文件取回来」这个需求。dest_dir 为
    hdfs:// 地址时走 hdfs/hadoop 客户端上传，否则按本机/已挂载目录直接复制——与 fetch_input
    对输入地址「hdfs:// 才下载，其余原样使用」是同一个对称设计，这样本地 dryrun 不需要真实
    hdfs 集群也能跑通整条产出路径。每次运行落到 dest_dir 下一个按时间戳+进程号区分的子目录，
    避免同一个 dest_dir 被反复写入时互相覆盖。

    这里的失败是致命的（sys.exit）：批处理算子跑完就退出，产出取不回来就等于这次运行白跑，
    与常驻服务保存片段时「失败只告警」是不同的取舍。

    files: {输出名: 本地路径}。返回 {输出名: 落地后的地址}。
    """
    run_dir = dest_dir.rstrip("/") + "/" + (run or run_dir_name())
    ok, msg = make_remote_dir(run_dir)
    if not ok:
        sys.exit(f"error: 产出目录 {dest_dir} 不可用：{msg}")

    urls = {}
    for name, local_path in files.items():
        ok, msg = upload_file(local_path, run_dir)
        if not ok:
            sys.exit(f"error: 输出 {name} 落地失败：{msg}")
        print(f"[output] {local_path} -> {msg}")
        urls[name] = msg
    return urls


def require_destination(local_dir, hdfs_dir):
    """两个落地目录至少填一个，否则这次运行的结果谁也拿不到。

    在跑之前校验，而不是跑完再说：批处理算子动辄几分钟 GPU 时间，等算完才告诉使用者「没地方
    放」，白跑的就是那几分钟。
    """
    if not (local_dir or "").strip() and not (hdfs_dir or "").strip():
        sys.exit("error: local_output_dir 与 hdfs_output_dir 都为空——outputPath 指向的文件"
                 "在容器退出后即不可再取，这次运行的结果将无法获得。请至少填写一个："
                 "local_output_dir 填容器内已挂载的目录（如 NFS 挂载点），"
                 "hdfs_output_dir 填 hdfs:// 地址。")


def deliver(files, local_dir, hdfs_dir, labels=None):
    """把产出送到本地目录（通常是挂进来的 NFS）和/或 HDFS。-> {输出名: [落地地址]}。

    两个目的地共用同一个运行子目录名，这样同一次运行在 NFS 和 HDFS 上是同一个名字，对得上。
    """
    labels = labels or {}
    run = run_dir_name()
    landed = {name: [] for name in files}
    for dest in [d for d in ((local_dir or "").strip(), (hdfs_dir or "").strip()) if d]:
        where = "hdfs" if dest.startswith("hdfs://") else "本地"
        for name, url in upload_outputs(files, dest, run=run).items():
            landed[name].append(url)
            print(f"[done] {labels.get(name, name)}({where}) -> {url}")
    return landed


def resolve_ckpt_dir(value, here):
    """权重目录：绝对路径原样用；相对路径优先按算子包所在目录解析。

    权重随 zip 一起发布（约 33 MB），而不是烤进镜像——镜像 6.1 GB，换一次权重要重传整个镜像，
    走 zip 只要传 33 MB。默认值因此是相对路径 `ckpts`，指向包内那一份。

    相对路径先看包内有没有，没有再按当前目录解析：仓库里开发时 main.py 在
    platform/op_xxx/ 下，权重却在仓库根的 ckpts/，两种情形要都能跑。
    """
    if os.path.isabs(value):
        return value
    packaged = os.path.join(here, value)
    return packaged if os.path.isdir(packaged) else value


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
