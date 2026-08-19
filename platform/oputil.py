"""算子公共工具：平台参数解析、outputPath 父目录创建、共用文件库地址落盘。

两个算子（处理算子、测试验证算子）各自打包时都会把本文件放到 main.py 同级目录，因此两个
zip 都是自包含的，不依赖对方。
"""
import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BOOL_TRUE = ("1", "true", "t", "yes", "y", "on", "是")
BOOL_FALSE = ("0", "false", "f", "no", "n", "off", "否")

# WebHDFS：不需要 hadoop 客户端的那条 HDFS 通道。镜像里没装 hadoop/JVM（为一次 -get/-put
# 加一套 JVM 不值当，镜像已经 6.1 GB），所以 hdfs:// 的 CLI 路径在容器里其实一直是失败的，
# 只是服务算子把上传失败当告警才没暴露。平台下发的输入地址形如
# http://10.46.79.133:9870/behavor/darkpipe/xxx.mp4 —— 9870 是 NameNode 的 HTTP 端口，
# 也说明对方本来走的就是这条 HTTP 通道。
WEBHDFS_PREFIX = "/webhdfs/v1"
NAMENODE_HTTP_PORT = 9870          # Hadoop 3.x 默认；2.x 是 50070
_REDIRECT_CODES = (301, 302, 303, 307, 308)
# 只有「看起来是一个有限长文件」的 http 地址才下载。.m3u8 / 无后缀 / mjpg 这类是实时流，
# 必须原样交给 cv2.VideoCapture 边拉边解，下载它等于永远不返回。
_FILE_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm")


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


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """关掉 urllib 的自动跳转，把 307 交回给我们。

    WebHDFS 是两段式的：NameNode 只做定位，回一个 307 指向真正存数据的 DataNode。自动跳转
    在这里帮倒忙——PUT 的请求体不会被正确重放，大文件也不该整个读进内存；更要紧的是那个
    跳转目标是**按主机名**寻址的（本项目部署里叫 hdfs-datanode），解析不了时我们要能拿到
    这个 URL 自己改写，而不是眼睁睁看着 urllib 抛一个 DNS 错误。
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _is_dns_failure(exc):
    return isinstance(exc, socket.gaierror) or isinstance(getattr(exc, "reason", None),
                                                          socket.gaierror)


def split_hdfs_target(url):
    """把 hdfs:// 或 http(s):// 的目标拆成 (WebHDFS 根, HDFS 路径, 用户名)。

    hdfs://用户名@ip:8020/a/b 里的端口是 RPC 端口，WebHDFS 不在那上面，而在 NameNode 的
    HTTP 端口（Hadoop 3.x 默认 9870）。端口不是 9870 时，直接把地址写成 http://ip:端口/a/b
    即可——多一个逃生口，不为此再加一个平台参数。
    """
    u = urllib.parse.urlsplit(url)
    user = urllib.parse.unquote(u.username or "")
    if u.scheme == "hdfs":
        scheme, port = "http", NAMENODE_HTTP_PORT
    else:
        scheme = u.scheme or "http"
        port = u.port or (443 if scheme == "https" else 80)
    path = u.path or "/"
    if path.startswith(WEBHDFS_PREFIX):          # 已经是 WebHDFS 形式，剥掉前缀取真实路径
        path = path[len(WEBHDFS_PREFIX):] or "/"
    return f"{scheme}://{u.hostname or ''}:{port}", urllib.parse.unquote(path), user


def webhdfs_url(base, path, op, user="", **params):
    """拼 WebHDFS 地址。路径要 percent-encode——平台下发的文件名是中文的。"""
    if not path.startswith("/"):
        path = "/" + path
    q = {"op": op}
    if user:
        q["user.name"] = user
    q.update({k: v for k, v in params.items() if v is not None})
    return (base + WEBHDFS_PREFIX + urllib.parse.quote(path, safe="/") + "?"
            + urllib.parse.urlencode(q))


def _webhdfs_call(url, method="GET", src_path=None, timeout=60):
    """走完 WebHDFS 的两跳，返回 DataNode 的响应。

    第二跳解析不了主机名时，把跳转 URL 的主机换成 NameNode 的 IP 再试一次：单机/同机部署下
    DataNode 就在那台机器上，这一跳能救回来。这是 /etc/hosts 那条记录写不进去（只读挂载、
    非 root）时的兜底——两层里有一层成就能跑。
    """
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        return opener.open(urllib.request.Request(url, method=method), timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code not in _REDIRECT_CODES or not e.headers.get("Location"):
            raise
        loc = e.headers["Location"]

    nn_host = urllib.parse.urlsplit(url).hostname or ""
    dn = urllib.parse.urlsplit(loc)
    targets = [loc]
    if dn.hostname and dn.hostname != nn_host:
        targets.append(urllib.parse.urlunsplit(
            dn._replace(netloc=f"{nn_host}:{dn.port}" if dn.port else nn_host)))
    last = None
    for i, target in enumerate(targets):
        req = urllib.request.Request(target, method=method)
        fh = None
        if src_path is not None:
            fh = open(src_path, "rb")            # 每次重试都重开：上一次可能已经读掉一部分
            req.data = fh
            req.add_header("Content-Length", str(os.path.getsize(src_path)))
            req.add_header("Content-Type", "application/octet-stream")
        try:
            return opener.open(req, timeout=timeout)
        except Exception as e:                   # noqa: BLE001 - 换个目标重试，最后一次再抛
            last = e
            if i + 1 < len(targets) and _is_dns_failure(e):
                print(f"[hdfs] 跳转目标 {dn.hostname} 无法解析，改用 NameNode 地址 "
                      f"{nn_host} 重试（在 /etc/hosts 里加一条 "
                      f"「<IP> {dn.hostname}」可以省掉这一步）")
                continue
            raise
        finally:
            if fh is not None:
                fh.close()
    raise last


def apply_extra_hosts(spec):
    """把 "IP 主机名[, IP 主机名]" 追加进 /etc/hosts，返回实际写入的条目数。

    WebHDFS 的第二跳是按主机名寻址的 DataNode，容器里默认解析不了这个名字，下载就断在那。
    K8s 侧的正解是 pod 的 hostAliases，但算子改不到编排文件，所以进程起来时自己补一条。
    写不进去只告警不退出：_webhdfs_call 还有一层「换成 NameNode 的 IP」的兜底。
    """
    entries = []
    for item in (spec or "").replace(";", ",").split(","):
        parts = item.split()
        if len(parts) >= 2:
            entries.append((parts[0], parts[1]))
        elif parts:
            print(f"[hosts] 忽略无法解析的条目 {item.strip()!r}（应形如 「10.0.0.1 myhost」）")
    if not entries:
        return 0

    try:
        with open("/etc/hosts", encoding="utf-8") as f:
            existing = f.read()
    except OSError as e:
        print(f"[hosts] 读取 /etc/hosts 失败：{e}；跳过（下载仍有 NameNode 地址兜底）")
        return 0

    todo = []
    for ip, host in entries:
        if f" {host}" in existing or f"\t{host}" in existing:
            print(f"[hosts] {host} 已在 /etc/hosts 中，跳过")
            continue
        try:
            resolved = socket.gethostbyname(host)
            print(f"[hosts] {host} 已可解析为 {resolved}，跳过")
            continue
        except OSError:
            todo.append((ip, host))
    if not todo:
        return 0
    try:
        with open("/etc/hosts", "a", encoding="utf-8") as f:
            for ip, host in todo:
                f.write(f"{ip}\t{host}\n")
                print(f"[hosts] 已添加 {ip} {host}")
    except OSError as e:
        print(f"[hosts] 写入 /etc/hosts 失败：{e}（多半是只读挂载或非 root）；"
              f"下载仍有 NameNode 地址兜底，不中断")
        return 0
    return len(todo)


def _save_stream(resp, local, label):
    """把响应体落盘并报进度。大视频要跑十几秒，静默下载在平台日志里看不出死活。"""
    total = int(resp.headers.get("Content-Length") or 0)
    got = 0
    t0 = last = time.time()
    with open(local, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            now = time.time()
            if now - last >= 5.0:
                last = now
                pct = f" ({got / total * 100:.0f}%)" if total else ""
                print(f"[input] {label} 已下载 {got / 1e6:.0f} MB{pct}")
    dt = max(time.time() - t0, 1e-9)
    print(f"[input] {label} 完成：{got / 1e6:.1f} MB / {dt:.1f}s = {got / 1e6 / dt:.1f} MB/s "
          f"-> {local}")
    return got


def _local_name(url, workdir, default_name):
    os.makedirs(workdir, exist_ok=True)
    base = os.path.basename(urllib.parse.unquote(urllib.parse.urlsplit(url).path))
    return os.path.join(workdir, base or default_name)


def _looks_like_file_url(url):
    path = urllib.parse.urlsplit(url).path.lower()
    return path.endswith(_FILE_SUFFIXES) or path.startswith(WEBHDFS_PREFIX)


def _normalize_url(url):
    """Percent-encode the path so a non-ASCII address can reach the wire at all.

    平台下发的地址里文件名是中文且**没有编码**。http.client 要求请求行是纯 ASCII，直接发会
    在本地抛 UnicodeEncodeError，请求根本出不去。safe 里留着 % 是为了不把已经编码过的地址
    二次编码（%E6 变成 %25E6）。
    """
    u = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit(u._replace(path=urllib.parse.quote(u.path, safe="/%")))


def _fetch_http(src, workdir, default_name):
    """下载一个 http(s) 文件地址。先按原样取，再按 WebHDFS 取。

    先原样：这是使用者字面上给的地址，能成就零意外。平台给的
    http://<nn>:9870/<路径> 会在这一步 404（9870 是 NameNode 的 Web UI 端口，不是文件服务），
    很便宜，然后第二步改写成 /webhdfs/v1/<路径>?op=OPEN 拿到真身。两条都失败时把**两个
    试过的地址和各自的状态**一起报出来——「cannot open input source」这种含糊报错正是这次
    排查花掉时间的原因。
    """
    local = _local_name(src, workdir, default_name)
    base, path, user = split_hdfs_target(src)
    if urllib.parse.urlsplit(src).path.startswith(WEBHDFS_PREFIX):
        attempts = [("WebHDFS", _normalize_url(src))]
    else:
        attempts = [("原始地址", _normalize_url(src)),
                    ("WebHDFS", webhdfs_url(base, path, "OPEN", user))]

    errors = []
    for label, url in attempts:
        print(f"[input] 尝试{label}：{url}")
        try:
            resp = _webhdfs_call(url, "GET")
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if ctype.startswith("text/html"):
                # NameNode 的 Web UI 会用 200 + HTML 回一个错误页，落盘就是个假 mp4。
                resp.close()
                raise RuntimeError(f"返回的是网页而不是文件（Content-Type: {ctype}）")
            with resp:
                _save_stream(resp, local, label)
            return local
        except Exception as e:                   # noqa: BLE001 - 换下一种形式再试
            code = getattr(e, "code", "")
            errors.append(f"  {label} {url}\n    -> {type(e).__name__}"
                          f"{f' {code}' if code else ''}: {e}")
    sys.exit("error: 输入地址取不下来，以下形式都试过了：\n" + "\n".join(errors)
             + "\n  提示：若卡在 DataNode 主机名解析上，用 extra_hosts 参数补一条 "
               "「<IP> <主机名>」；若 WebHDFS 端口不是 "
             + f"{NAMENODE_HTTP_PORT}，直接把地址写成 http://<ip>:<端口>/<路径>。")


def _fetch_hdfs(src, workdir, default_name):
    """hdfs:// 地址：有 hadoop 客户端就用它，没有就退到 WebHDFS。

    退到 WebHDFS 而不是像以前那样直接 sys.exit —— 镜像里本来就没有 hadoop 客户端，
    以前那条路等于永远走不通。
    """
    exe, sub = _hdfs_cli()
    local = _local_name(src, workdir, default_name)
    if exe:
        print(f"[input] 下载 {src} -> {local}（hadoop 客户端）")
        r = subprocess.run([exe, sub, "-get", src, local],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if r.returncode == 0 and os.path.exists(local):
            return local
        print(f"[input] hadoop 客户端下载失败(exit={r.returncode})，改走 WebHDFS："
              f"{(r.stdout or '').strip()[-400:]}")
    else:
        print("[input] 容器内没有 hdfs/hadoop 客户端，改走 WebHDFS（HTTP）")
    base, path, user = split_hdfs_target(src)
    url = webhdfs_url(base, path, "OPEN", user)
    print(f"[input] 尝试 WebHDFS：{url}")
    try:
        with _webhdfs_call(url, "GET") as resp:
            _save_stream(resp, local, "WebHDFS")
        return local
    except Exception as e:                       # noqa: BLE001 - 统一成一句可定位的报错
        sys.exit(f"error: 下载 {src} 失败。\n  WebHDFS 地址：{url}\n"
                 f"  -> {type(e).__name__}: {e}\n"
                 f"  提示：hdfs:// 里的端口是 RPC 端口，WebHDFS 走的是 NameNode 的 HTTP 端口"
                 f"（这里按 {NAMENODE_HTTP_PORT} 推导）。端口不同就把地址写成 "
                 f"http://<ip>:<端口>/<路径>。")


def fetch_input(src, workdir, default_name="input.bin"):
    """把共用/模型文件库地址取到本地，返回本地可读路径。

    文件库下发的地址形如 hdfs://用户名@ip:port/a/b/c.mp4，也可能是 WebHDFS 的
    http://ip:9870/a/b/c.mp4；OpenCV 与 open() 都打不开，必须先落盘。
    rtsp:// rtmp:// 、实时 HTTP 流（.m3u8/无后缀）及本地路径原样返回。
    """
    src = (src or "").strip()
    scheme = src.split("://", 1)[0].lower() if "://" in src else ""
    if scheme == "hdfs":
        return _fetch_hdfs(src, workdir, default_name)
    if scheme in ("http", "https") and _looks_like_file_url(src):
        return _fetch_http(src, workdir, default_name)
    return src


def run_dir_name():
    """一次运行的专属子目录名，时间戳+进程号，避免同一个目标目录被反复写入时互相覆盖。"""
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"


def _hdfs_cli():
    """-> (可执行文件, 子命令) 或 (None, None)。hdfs 用 `hdfs dfs`，hadoop 用 `hadoop fs`。"""
    exe = shutil.which("hdfs") or shutil.which("hadoop")
    if not exe:
        return None, None
    return exe, ("dfs" if os.path.basename(exe) == "hdfs" else "fs")


def is_remote_dir(dest_dir):
    """dest_dir 是不是 HDFS 地址（hdfs:// 或 WebHDFS 的 http(s)://），而不是本机目录。"""
    return str(dest_dir).startswith(("hdfs://", "http://", "https://"))


def make_remote_dir(dest_dir):
    """建好目标目录。-> (成功?, 说明)。不退出进程，由调用方决定致命与否。"""
    if not is_remote_dir(dest_dir):
        try:
            os.makedirs(dest_dir, exist_ok=True)
        except OSError as e:
            return False, f"创建目录 {dest_dir} 失败: {e}"
        return True, dest_dir
    exe, sub = _hdfs_cli()
    if exe and dest_dir.startswith("hdfs://"):
        r = subprocess.run([exe, sub, "-mkdir", "-p", dest_dir],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if r.returncode == 0:
            return True, dest_dir
        print(f"[output] hadoop 客户端建目录失败(exit={r.returncode})，改走 WebHDFS："
              f"{(r.stdout or '').strip()[-400:]}")
    base, path, user = split_hdfs_target(dest_dir)
    url = webhdfs_url(base, path, "MKDIRS", user)
    print(f"[output] 建目录（WebHDFS）：{url}")
    try:
        with _webhdfs_call(url, "PUT"):
            pass
    except Exception as e:                       # noqa: BLE001 - 调用方决定这是否致命
        return False, f"创建 hdfs 目录失败：{type(e).__name__}: {e}（{url}）"
    return True, dest_dir


def upload_file(local_path, dest_dir):
    """把单个文件送到 dest_dir（须已存在）。-> (成功?, 落地地址或错误说明)。

    同样不退出进程：常驻服务不能因为一次上传失败就整体死掉，本地那份文件已经在盘上了。
    """
    if not is_remote_dir(dest_dir):
        dest = os.path.join(dest_dir, os.path.basename(local_path))
        try:
            shutil.copy2(local_path, dest)
        except OSError as e:
            return False, f"复制 {local_path} -> {dest} 失败: {e}"
        return True, dest
    name = os.path.basename(local_path)
    dest = dest_dir.rstrip("/") + "/" + name
    exe, sub = _hdfs_cli()
    if exe and dest_dir.startswith("hdfs://"):
        r = subprocess.run([exe, sub, "-put", "-f", local_path, dest],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if r.returncode == 0:
            return True, dest
        print(f"[output] hadoop 客户端上传失败(exit={r.returncode})，改走 WebHDFS："
              f"{(r.stdout or '').strip()[-400:]}")
    base, path, user = split_hdfs_target(dest_dir)
    url = webhdfs_url(base, path.rstrip("/") + "/" + name, "CREATE", user, overwrite="true")
    size = os.path.getsize(local_path)
    print(f"[output] 上传（WebHDFS）{local_path} ({size / 1e6:.1f} MB) -> {url}")
    t0 = time.time()
    try:
        with _webhdfs_call(url, "PUT", src_path=local_path):
            pass
    except Exception as e:                       # noqa: BLE001 - 调用方决定这是否致命
        return False, f"上传失败：{type(e).__name__}: {e}（{url}）"
    dt = max(time.time() - t0, 1e-9)
    print(f"[output] 上传完成 {size / 1e6:.1f} MB / {dt:.1f}s = {size / 1e6 / dt:.1f} MB/s")
    return True, dest


def upload_outputs(files, dest_dir, run=None):
    """把 outputPath 落盘的产出文件送到 dest_dir，让容器退出后仍能取到。

    run: 本次运行的子目录名。同一次运行要落到多个目的地（本地/NFS 一份、HDFS 一份）时，
    调用方生成一次传进来，两边的子目录名才会一致——各自调用 run_dir_name() 会差出几秒，
    落成两个对不上的目录。

    outputPath 只是容器内的一个临时路径，镜像销毁后就没了；框架间 outputPath/inputPath 的
    传递也只发生在同一次编排内，覆盖不了「事后把结果文件取回来」这个需求。dest_dir 为
    hdfs:// 或 WebHDFS 的 http:// 地址时走 HDFS 上传，否则按本机/已挂载目录直接复制——与
    fetch_input 对输入地址「远端才下载，其余原样使用」是同一个对称设计，这样本地 dryrun
    不需要真实 hdfs 集群也能跑通整条产出路径。每次运行落到 dest_dir 下一个按时间戳+进程号
    区分的子目录，避免同一个 dest_dir 被反复写入时互相覆盖。

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
                 "hdfs_output_dir 填 hdfs:// 或 WebHDFS 的 http://<ip>:9870/<路径> 地址。")


def deliver(files, local_dir, hdfs_dir, labels=None, run=None):
    """把产出送到本地目录（通常是挂进来的 NFS）和/或 HDFS。-> {输出名: [落地地址]}。

    两个目的地共用同一个运行子目录名，这样同一次运行在 NFS 和 HDFS 上是同一个名字，对得上。
    run 显式传入时用调用方的名字——实时服务算子的会话号同时用在 clip_dir、HDFS 和
    session_json 上，三处必须是同一个名字才对得上。
    """
    labels = labels or {}
    run = run or run_dir_name()
    landed = {name: [] for name in files}
    for dest in [d for d in ((local_dir or "").strip(), (hdfs_dir or "").strip()) if d]:
        where = "hdfs" if is_remote_dir(dest) else "本地"
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
