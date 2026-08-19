"""暗光场景行为识别实时服务算子——平台入口。

与处理算子（op_dark_behavior）同一套 darkpipe 代码路径，区别只在运行形态：处理算子是
「跑完退出、产出文件」的批处理任务，本算子是常驻服务，边拉流边出结果，对外给三条流：

    GET /stream   MJPEG 视频流（增强+超分+标签条），用于实时演示
    GET /events   SSE 事件流，每识别出一次行为推一条 JSON
    GET /health   健康检查与实时计数（含片段保存统计）
    GET /config   当前生效配置

并按事件流把「不是 other」的行为切成一个个片段视频存下来：先写容器内 clip_dir（可挂载
NFS 后本地浏览），再按需上传 hdfs_output_dir。两处保存的目录结构完全一致：

    <clip_dir 或 hdfs_output_dir>/<会话>/<行为>/<时间>_<行为>_<序号>.mp4  + 同名 .json

平台规范在本文件中的落实：
  * 主入口文件名 main.py，与 suanzi.json 的 command 一致；
  * 用户填写的参数一律走 inputs；outputs 只有一个由框架下发路径的 session_json；
  * outputPath 的父目录由算子自行创建；
  * 日志一律 print 到标准输出，由框架收集；
  * 失败时打印堆栈并以非零码退出。

离线模式：**输入是一个有限长的文件、且没有填 rtmp_push_url** 时，本算子自动转为离线推理
——一次跑完整段视频，产出「整段处理后的视频 + 事件 JSON」，落到 clip_dir 与
hdfs_output_dir，然后进程正常退出。既不起服务，也不按动作切片。输入地址可以是 hdfs://
或 WebHDFS 的 http://<namenode>:9870/<路径>，会先下载到本地再处理。
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

# 打包后 darkpipe/ 与 oputil.py 就在 main.py 同级；仓库内开发时 oputil.py 在上一级目录。
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]

from oputil import (resolve_ckpt_dir, apply_extra_hosts, deliver, ensure_parent,  # noqa: E402
                    fetch_input, make_remote_dir, parse_bool, parse_gpu_ids, run_dir_name,
                    upload_file)


def build_parser():
    p = argparse.ArgumentParser(
        prog="main.py",
        description="暗光场景行为识别实时服务算子：拉流 -> 增强 -> 识别 -> 三条流 + 片段留存")
    # 参数名与 suanzi.json 的 inputs[].name / outputs[].name 逐一对应
    p.add_argument("--video_path", required=True,
                   help="待处理实时流：国标(GB28181)/rtsp:// / http(s):// (flv、hls)，"
                        "或容器内本地视频文件、hdfs:// 地址、WebHDFS 的 "
                        "http://<namenode>:9870/<路径> 地址。填文件且 rtmp_push_url 留空时"
                        "自动转为离线推理：跑完整段视频、产出整段结果后退出")
    p.add_argument("--enhance", default="retinexformer", choices=["off", "retinexformer"])
    p.add_argument("--sr", default="bicubic", choices=["off", "bicubic"])
    p.add_argument("--sr_scale", type=int, default=2, choices=[2, 3, 4])
    p.add_argument("--denoise", default="fast",
                   choices=["off","fast","quality","quality_high"],
                   help="画面去噪：off/fast(双边~5ms,35%)/quality(NLM窗7~119ms,51%)/quality_high(NLM窗15~376ms,75%)；识别之后进行，实时流与片段共用同一帧")
    p.add_argument("--color_saturation", type=float, default=1.0,
                   help="画面色彩饱和度倍数，1.0=不处理；增强会把画面拉灰，2.0-2.6 可恢复色彩。在 Lab 空间缩放色度，色相不变；识别之后进行，不影响识别")
    # 不提供 off：本算子靠事件流切片段，没有识别器就既没有事件也没有片段
    p.add_argument("--recognize", default="behavior", choices=["behavior"])
    p.add_argument("--proc_max_side", type=int, default=840,
                   help="处理前把画面长边缩到不超过该值，0=按原分辨率。增强耗时与像素量成正比，而识别内部固定缩到 224，所以缩放不损识别精度。默认 840 是双卡下画质与帧率的折中(23.8 fps/p95 289 ms)；只分到 1 张卡时 840 只有 12.9 fps，达不到 15 fps，需改填 720")
    p.add_argument("--reco_span_sec", type=float, default=1.0)
    p.add_argument("--reco_min_conf", type=float, default=0.0,
                   help="行为判定阈值(0-1)：最高分的具名行为达不到该值时报为 other，既不显示也不存片段；0=关闭")
    p.add_argument("--label_bar", type=parse_bool, default=True)
    p.add_argument("--gpu_ids", default="0,1",
                   help="容器内 GPU 卡号(从 0 起编，与宿主机序号无关)，逗号分隔。填多张时按帧轮询分发，识别与定序固定在第一张卡上。实际可见的卡少于填写的会自动降级并告警，真正决定给几张卡的是 suanzi.json 里的 metadata.gpu.count")
    p.add_argument("--ckpt_dir", default="ckpts",
                   help="权重目录。默认 ckpts 指算子包内随包发布的那一份；也可填绝对路径改用镜像内或挂载进来的权重")
    p.add_argument("--serve_port", type=int, default=8000)
    p.add_argument("--stream_formats", default="mjpeg,flv",
                   choices=["mjpeg", "mjpeg,flv", "mjpeg,flv,hls", "mjpeg,hls"],
                   help="对外的实时视频流格式；flv/hls 由镜像内的 ffmpeg 封装")
    p.add_argument("--rtmp_push_url", default="",
                   help="推流到外部流媒体服务器，如 rtmp://ip:1935/live/key；留空不推。"
                        "留空且 video_path 是一个文件时，算子转为离线推理模式")
    p.add_argument("--stream_bitrate", default="4M",
                   help="所有 H.264 出流的码率上限，如 4M/8M；留空则不限制")
    p.add_argument("--max_flv_clients", type=int, default=4)
    p.add_argument("--jpeg_quality", type=int, default=85)
    p.add_argument("--max_stream_fps", type=float, default=15.0)
    p.add_argument("--clip_dir", default="/opt/darkpipe/clips",
                   help="片段保存目录（容器内路径，建议挂载 NFS 后本地浏览）；"
                        "离线模式下整段处理结果也落在这里")
    p.add_argument("--clip_pre_sec", type=float, default=2.0)
    p.add_argument("--clip_post_sec", type=float, default=2.0)
    p.add_argument("--clip_max_sec", type=float, default=30.0)
    p.add_argument("--clip_skip_labels", default="other")
    p.add_argument("--clip_denoise", default="quality",
                   choices=["off","fast","quality","quality_high"],
                   help="仅对保存的片段去噪，在写盘线程执行，不占 GPU；取值同 denoise。注意 quality(NLM) 是纯 CPU 重活，事件密集时会和管线抢 CPU 把端到端时延顶到秒级，要守 1 秒指标就填 off 或 fast")

    # 唯一没有 default 的可选参数：规范里「有 default 即必填不能为空」，要允许留空就不能给
    # default。留空表示不推 HDFS，只保留 clip_dir 里的那一份。
    p.add_argument("--hdfs_output_dir", default="",
                   help="片段（离线模式下是整段视频）另存的 HDFS 目录，如 "
                        "hdfs://用户名@ip:port/a/b，或 WebHDFS 的 http://<ip>:9870/a/b；"
                        "留空则不推送。镜像内没有 hadoop 客户端，实际走的是 WebHDFS(HTTP)")
    p.add_argument("--extra_hosts", default="10.46.79.133 hdfs-datanode",
                   help="启动时追加到 /etc/hosts 的域名解析，格式「IP 主机名」，多条用逗号"
                        "分隔。WebHDFS 会 307 跳到按主机名寻址的 DataNode，容器里默认解析"
                        "不了；留空表示不加。写不进 /etc/hosts 时会退回用 NameNode 的地址重试")
    p.add_argument("--run_seconds", type=float, default=0.0,
                   help="运行时长上限（秒），0 表示一直运行到容器被停止；离线模式忽略此项")
    p.add_argument("--session_json", required=True)
    return p


class HdfsClipSink:
    """把已经落到 clip_dir 的片段再送一份到 HDFS。

    失败只告警不退出：常驻服务不该因为一次网络抖动整体死掉，而且本地那份片段已经在盘上，
    HDFS 这份是副本不是唯一产出——这与批处理算子里「上传失败即致命」是有意不同的取舍。
    目录是首次用到时才建的：服务可能连续几小时没有事件，没必要在启动时就去碰 HDFS。
    """

    def __init__(self, dest_dir, session):
        self.root = dest_dir.rstrip("/") + "/" + session
        self.made = set()
        self.uploaded = 0
        self.failed = 0

    def ensure_dir(self, rel_dir):
        target = self.root + "/" + rel_dir if rel_dir else self.root
        if target in self.made:
            return target
        ok, msg = make_remote_dir(target)
        if not ok:
            print(f"[clip] HDFS 目录不可用，本次跳过上传：{msg}")
            return None
        self.made.add(target)
        return target

    def __call__(self, files, rel_dir, meta):
        target = self.ensure_dir(rel_dir)
        if target is None:
            self.failed += len(files)
            return
        for f in files:
            ok, msg = upload_file(f, target)
            if ok:
                self.uploaded += 1
                print(f"[clip] 已上传 {msg}")
            else:
                self.failed += 1
                print(f"[clip] 上传 {os.path.basename(f)} 失败（本地仍保留）：{msg}")


def endpoints(port, formats, push_url):
    """只列出本次真正开启的接口——列出没启用的地址等于给下游一个必然打不开的 URL。"""
    host = "http://<容器地址>:%d" % port
    eps = {"events_sse": f"{host}/events", "health": f"{host}/health",
           "config": f"{host}/config"}
    fmts = [f.strip() for f in formats.split(",") if f.strip()]
    if "mjpeg" in fmts:
        eps["stream_mjpeg"] = f"{host}/stream"
    if "flv" in fmts:
        eps["stream_flv"] = f"{host}/live.flv"
    if "hls" in fmts:
        eps["stream_hls"] = f"{host}/hls/index.m3u8"
    if push_url:
        eps["rtmp_push"] = push_url
    return eps


def _print_gpu_inventory(gpus):
    """打印每张卡的型号、显存和 UUID，让日志自己回答「到底给了几张卡」。

    只看 `[serve] 多卡按帧轮询：cuda:0, cuda:1` 那行是不够的——它打印的是**请求**的卡号，
    不是实际拿到的硬件。UUID 才是判据：两行 UUID 不同 = 两张物理卡；相同 = 平台把一张卡
    虚拟成了两个设备（本平台装了 HAMi vGPU，这是它能做到的），此时多卡不会有加速，两个
    实例只是在同一块卡上互相分时间片，实测反而比单卡慢。
    """
    try:
        import torch
        if not torch.cuda.is_available():
            print("[gpu] CUDA 不可用")
            return
        n = torch.cuda.device_count()
        print(f"[gpu] torch 可见 {n} 张卡；本次使用 {', '.join('cuda:' + g for g in gpus)}")
        seen = {}
        for g in gpus:
            i = int(g)
            if i >= n:
                print(f"[gpu]   cuda:{i} 不存在（可见的只有 0..{n - 1}）")
                continue
            p = torch.cuda.get_device_properties(i)
            uu = str(getattr(p, "uuid", "?"))
            print(f"[gpu]   cuda:{i} {p.name} {p.total_memory / 2**30:.1f}GiB uuid={uu}")
            seen.setdefault(uu, []).append(i)
        dup = [v for v in seen.values() if len(v) > 1]
        if dup:
            print(f"[gpu] 警告：{dup} 是同一块物理卡的多个设备号，多卡分发不会带来加速")
    except Exception as e:                                   # 诊断信息，不值得让服务起不来
        print(f"[gpu] 读取 GPU 信息失败（不影响运行）：{e}")


def write_session(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def probe_video(path):
    """-> (帧数, fps) 或 (0, 0.0)。只为把「输入到底是什么」写进日志，失败不影响运行。"""
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS) or 0)
        cap.release()
        return max(n, 0), fps
    except Exception:                                        # noqa: BLE001 - 诊断信息而已
        return 0, 0.0


def run(args):
    """决定这次是常驻服务还是一次性离线推理，然后转给对应的实现。

    判据只有一条，与需求逐字对应：**取回来之后它是磁盘上一个有限长的文件，且没有要推的流**。
    hdfs:// 与 WebHDFS 的 http:// 地址在 fetch_input 里已经落成本地文件，所以这条判据同时
    覆盖了「平台下发 HDFS 地址」这种情形。本地文件 + 填了推流地址仍走实时模式（循环播放
    本地文件推出去，是原有的演示用法，不动它）。
    """
    # 先补 /etc/hosts：WebHDFS 的 307 会跳到按主机名寻址的 DataNode，这一步要在任何取数据
    # 的动作之前完成。写不进去也只是告警——_webhdfs_call 里还有「改用 NameNode 地址重试」兜底。
    apply_extra_hosts(args.extra_hosts)

    session = run_dir_name()
    # 下载目录不能用 tempfile 的自动清理：实时模式下这个文件要陪着服务跑一整天。
    workdir = tempfile.mkdtemp(prefix="darkserve_")
    try:
        src = fetch_input(args.video_path, workdir, "input.mp4")
        if os.path.isfile(src) and not (args.rtmp_push_url or "").strip():
            n, fps = probe_video(src)
            size = os.path.getsize(src) / 1e6
            print(f"[mode] 离线推理：输入是有限长文件（{size:.0f} MB / "
                  f"{n or '?'} 帧 / {fps:.2f} fps）且未填写 rtmp_push_url，"
                  f"处理完整段后退出，不切片、不起服务")
            return run_offline_mode(args, src, session, workdir)
        why = ("已填写 rtmp_push_url" if (args.rtmp_push_url or "").strip()
               else "输入不是本地文件（按实时流处理）")
        print(f"[mode] 实时服务：{why}，常驻运行并按动作切片")
        return run_serve_mode(args, src, session)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run_offline_mode(args, src, session, workdir):
    """跑完整段视频、交付、退出。复用离线算子那条已经验证过的代码路径。"""
    from darkpipe.config import PipelineConfig, validate

    gpus = parse_gpu_ids(args.gpu_ids)
    _print_gpu_inventory(gpus)
    ensure_parent(args.session_json)

    stem = os.path.splitext(os.path.basename(src))[0] or "output"
    out_video = os.path.join(workdir, f"{stem}_enhanced.mp4")
    out_events = os.path.join(workdir, f"{stem}_events.json")
    cfg = validate(PipelineConfig(
        mode="offline", input=src, output=out_video, events_json=out_events,
        enhance=args.enhance, sr=args.sr,
        sr_scale=(args.sr_scale if args.sr != "off" else None),
        recognize=args.recognize, device=f"cuda:{gpus[0]}",
        gpus=(",".join(gpus) if len(gpus) > 1 else ""),
        ckpt_dir=resolve_ckpt_dir(args.ckpt_dir, _HERE), reco_min_conf=args.reco_min_conf,
        proc_max_side=args.proc_max_side, color_saturation=args.color_saturation,
        denoise=args.denoise,
        reco_span_sec=(args.reco_span_sec if args.reco_span_sec > 0 else None),
        no_label_bar=(not args.label_bar)))
    for w in cfg.warnings:
        print(f"[warn] {w}")
    print(f"[config] enhance={cfg.enhance} sr={cfg.sr_name()} recognize={cfg.recognize} "
          f"device={cfg.device} gpu_ids={','.join(gpus)} span={cfg.reco_span_sec}"
          f" reco_min_conf={cfg.reco_min_conf} proc_max_side={cfg.proc_max_side}"
          f" color_saturation={cfg.color_saturation}")
    print("[config] 离线模式：保存整段处理后的视频，clip_* 系列参数与 run_seconds 不生效")

    t0 = time.time()
    if len(gpus) > 1:
        from darkpipe.shard import run_offline_sharded
        events = run_offline_sharded(cfg, gpus)
    else:
        from darkpipe.pipeline import run_offline
        events = run_offline(cfg)
    dt = time.time() - t0

    events = [e if isinstance(e, dict) else e.to_dict() for e in events]
    if not os.path.exists(out_events):        # recognize=off 时管线不写事件文件
        with open(out_events, "w", encoding="utf-8") as f:
            json.dump(events, f, ensure_ascii=False, indent=2)
    counts = {}
    for e in events:
        counts[e["label"]] = counts.get(e["label"], 0) + 1
    st = cfg.stats
    print(f"[metrics] 帧数={st.get('frames', 0)} 处理耗时={st.get('seconds', 0)}s "
          f"fps={st.get('fps', 0)} 总耗时={dt:.1f}s 事件={len(events)} "
          f"行为分布={counts or '无'}")

    hdfs_dir = (args.hdfs_output_dir or "").strip()
    print(f"[deliver] 本地 -> {args.clip_dir}/{session}"
          + (f"；HDFS -> {hdfs_dir.rstrip('/')}/{session}" if hdfs_dir
             else "；未填写 hdfs_output_dir，不回传 HDFS"))
    landed = deliver({"output_video": out_video, "events_json": out_events},
                     args.clip_dir, hdfs_dir,
                     {"output_video": "整段处理结果", "events_json": "识别事件"},
                     run=session)

    payload = {
        "session": session, "mode": "offline", "status": "finished",
        "source": args.video_path, "local_input": src,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
        "stopped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_video": landed.get("output_video", []),
        "events_json": landed.get("events_json", []),
        "events_total": len(events), "label_counts": counts,
        "stats": dict(st, wall_seconds=round(dt, 3)),
        "config": {"enhance": cfg.enhance, "sr": cfg.sr_name(), "sr_scale": cfg.sr_scale,
                   "recognize": cfg.recognize, "gpu_ids": ",".join(gpus),
                   "reco_span_sec": cfg.reco_span_sec,
                   "proc_max_side": cfg.proc_max_side},
    }
    write_session(args.session_json, payload)
    print(f"[done] 会话信息 -> {args.session_json}")
    print("[done] 离线推理完成，进程退出")


def run_serve_mode(args, src, session):
    """常驻拉流服务：三条对外流 + 按动作切片。这条路径的行为与本次改动前逐字一致。"""
    from darkpipe.config import PipelineConfig, validate
    from darkpipe.server import run_server

    gpus = parse_gpu_ids(args.gpu_ids)
    # 多卡时按帧轮询分发（第 1 帧给卡 A、第 2 帧给卡 B……），不是离线那种按帧段切片——
    # 实时流没有「未来的帧」可切，但轮询也不需要。识别与定序固定在 gpus[0] 上：同一块卡
    # 上出现两个提交 CUDA 任务的线程会互相时间片切分，实测比串行还慢。
    if len(gpus) > 1:
        print(f"[serve] 多卡按帧轮询：{', '.join('cuda:' + g for g in gpus)}"
              f"（识别固定在 cuda:{gpus[0]}）")
    _print_gpu_inventory(gpus)
    ensure_parent(args.session_json)

    cfg = validate(PipelineConfig(
        mode="serve", input=src,
        enhance=args.enhance, sr=args.sr,
        sr_scale=(args.sr_scale if args.sr != "off" else None),
        recognize=args.recognize, device=f"cuda:{gpus[0]}",
        gpus=(",".join(gpus) if len(gpus) > 1 else ""),
        ckpt_dir=resolve_ckpt_dir(args.ckpt_dir, _HERE), reco_min_conf=args.reco_min_conf, proc_max_side=args.proc_max_side, color_saturation=args.color_saturation, denoise=args.denoise,
        reco_span_sec=(args.reco_span_sec if args.reco_span_sec > 0 else None),
        no_label_bar=(not args.label_bar),
        host="0.0.0.0", port=args.serve_port, jpeg_quality=args.jpeg_quality,
        max_stream_fps=args.max_stream_fps,
        stream_formats=args.stream_formats, rtmp_push_url=args.rtmp_push_url,
        max_flv_clients=args.max_flv_clients, stream_bitrate=args.stream_bitrate,
        clip_dir=args.clip_dir, clip_pre_sec=args.clip_pre_sec,
        clip_post_sec=args.clip_post_sec, clip_max_sec=args.clip_max_sec,
        clip_skip_labels=args.clip_skip_labels, clip_denoise=args.clip_denoise,
        # One session name for the local tree, the HDFS tree and session_json alike; letting
        # each side stamp its own produced directories a second apart.
        clip_session=session))
    for w in cfg.warnings:
        print(f"[warn] {w}")
    print(f"[config] enhance={cfg.enhance} sr={cfg.sr_name()} recognize={cfg.recognize} "
          f"device={cfg.device} span={cfg.reco_span_sec} port={cfg.port}"
          f" reco_min_conf={cfg.reco_min_conf} proc_max_side={cfg.proc_max_side}"
          f" color_saturation={cfg.color_saturation}")

    hdfs_dir = (args.hdfs_output_dir or "").strip()
    sink = None
    if hdfs_dir:
        sink = HdfsClipSink(hdfs_dir, session)
        print(f"[config] 片段将另存到 {sink.root}")
    else:
        print("[config] 未填写 hdfs_output_dir，片段只保存在 clip_dir，不推送 HDFS")

    clip_root = os.path.join(args.clip_dir, session)
    payload = {
        "session": session, "mode": "serve", "status": "running",
        "source": args.video_path, "local_input": src,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "endpoints": endpoints(args.serve_port, cfg.stream_formats, args.rtmp_push_url),
        "clip_dir": clip_root,
        "event_log": os.path.join(clip_root, "events.jsonl"),
        "hdfs_clip_dir": (sink.root if sink else ""),
        "clip_skip_labels": args.clip_skip_labels,
        "config": {"enhance": cfg.enhance, "sr": cfg.sr_name(), "recognize": cfg.recognize,
                   "gpu_ids": ",".join(gpus), "reco_span_sec": cfg.reco_span_sec,
                   "serve_port": args.serve_port, "run_seconds": args.run_seconds},
    }
    # 先写一份「运行中」：框架给的 outputPath 必须存在，而常驻服务可能被直接杀掉，
    # 等收尾再写就可能永远写不出来。
    write_session(args.session_json, payload)
    print(f"[done] 会话信息 -> {args.session_json}")

    st = run_server(cfg, on_clip=sink, stop_after=args.run_seconds)

    stats = st.clipper.stats() if st is not None and st.clipper else {}
    log_stats = st.eventlog.stats() if st is not None and st.eventlog else {}
    payload.update(status="stopped", stopped_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                   clips=stats, event_log_stats=log_stats,
                   events_total=(st.events_total if st is not None else 0),
                   hdfs_uploaded=(sink.uploaded if sink else 0),
                   hdfs_failed=(sink.failed if sink else 0))

    # events.jsonl is appended to for the whole run, so it can only be uploaded once at the
    # end — unlike clips, which go up as each one completes. A container killed outright
    # therefore leaves it only on clip_dir, which is the copy that matters anyway.
    log_path = payload["event_log"]
    if sink and os.path.exists(log_path):
        target = sink.ensure_dir("")
        if target:
            ok, msg = upload_file(log_path, target)
            print(f"[done] 事件日志(hdfs) -> {msg}" if ok else f"[warn] 事件日志上传失败：{msg}")
    write_session(args.session_json, payload)
    print(f"[metrics] 事件={payload['events_total']} 片段={stats.get('clips_saved', 0)} "
          f"上传={payload['hdfs_uploaded']} 上传失败={payload['hdfs_failed']}")
    print(f"[done] 会话信息 -> {args.session_json}")
    print(f"[done] 片段目录 -> {clip_root}")
    print(f"[done] 事件日志 -> {payload['event_log']}")
    if sink:
        print(f"[done] 片段HDFS目录 -> {sink.root}")

    # 会话文件也放一份到片段目录，挂 NFS 浏览时不用回头找框架收走的那份
    try:
        os.makedirs(clip_root, exist_ok=True)
        write_session(os.path.join(clip_root, "session.json"), payload)
    except OSError as e:
        print(f"[warn] 会话信息副本写入 {clip_root} 失败: {e}")


def write_failed_session(args, reason):
    """失败时也要把 session_json 写出来，否则平台只报「取不到 artifact」，真原因被埋掉。

    实测过一次：算子在多卡分片那一步退出 1，堆栈明明就在日志里，编排层显示的却是
    `cannot save artifact /tmp/outputs/session_json/data: no such file or directory`。
    这个文件是本算子唯一的框架级输出，写不出来就等于失败原因不可见。
    """
    path = (getattr(args, "session_json", "") or "").strip()
    if not path:
        return
    try:
        ensure_parent(path)
        write_session(path, {
            "status": "failed",
            "error": reason,
            "source": getattr(args, "video_path", ""),
            "stopped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        print(f"[failed] 失败原因已写入 {path}")
    except BaseException as e:                        # noqa: BLE001 - 收尾不能盖住真错误
        print(f"[warn] 失败信息写入 {path} 失败: {e}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        print("[serve] 收到中断，已退出")
    except SystemExit as e:
        # 参数/配置被拒时走的是 sys.exit("error: ...")，退出码本身就是那句话；
        # 纯数字退出码就没有更多信息可写，指回日志。退出码 0 是正常收尾，别覆盖。
        if e.code not in (0, None):
            write_failed_session(args, e.code if isinstance(e.code, str)
                                 else f"算子以退出码 {e.code} 结束，原因见日志")
        raise
    except BaseException as e:
        traceback.print_exc()
        write_failed_session(args, f"{type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
