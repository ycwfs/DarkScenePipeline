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
"""
import argparse
import json
import os
import sys
import time
import traceback

# 打包后 darkpipe/ 与 oputil.py 就在 main.py 同级；仓库内开发时 oputil.py 在上一级目录。
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]

from oputil import (ensure_parent, make_remote_dir, parse_bool,  # noqa: E402
                    parse_gpu_ids, run_dir_name, upload_file)


def build_parser():
    p = argparse.ArgumentParser(
        prog="main.py",
        description="暗光场景行为识别实时服务算子：拉流 -> 增强 -> 识别 -> 三条流 + 片段留存")
    # 参数名与 suanzi.json 的 inputs[].name / outputs[].name 逐一对应
    p.add_argument("--video_path", required=True,
                   help="待处理实时流：国标(GB28181)/rtsp:// / http(s):// (flv、hls)，"
                        "或容器内本地视频文件（本地文件会循环播放，便于没有摄像头时演示）")
    p.add_argument("--enhance", default="retinexformer", choices=["off", "retinexformer"])
    p.add_argument("--sr", default="bicubic", choices=["off", "bicubic"])
    p.add_argument("--sr_scale", type=int, default=2, choices=[2, 3, 4])
    # 不提供 off：本算子靠事件流切片段，没有识别器就既没有事件也没有片段
    p.add_argument("--recognize", default="behavior", choices=["behavior"])
    p.add_argument("--proc_max_side", type=int, default=1280,
                   help="处理前把画面长边缩到不超过该值，0=按原分辨率。增强耗时与像素量成正比，而识别内部固定缩到 224，1080p 填 1280 可省四分之三算力且不损识别精度")
    p.add_argument("--reco_span_sec", type=float, default=1.0)
    p.add_argument("--label_bar", type=parse_bool, default=True)
    p.add_argument("--gpu_ids", default="0")
    p.add_argument("--ckpt_dir", default="/opt/darkpipe/ckpts")
    p.add_argument("--serve_port", type=int, default=8000)
    p.add_argument("--stream_formats", default="mjpeg,flv",
                   choices=["mjpeg", "mjpeg,flv", "mjpeg,flv,hls", "mjpeg,hls"],
                   help="对外的实时视频流格式；flv/hls 由镜像内的 ffmpeg 封装")
    p.add_argument("--rtmp_push_url", default="",
                   help="推流到外部流媒体服务器，如 rtmp://ip:1935/live/key；留空不推")
    p.add_argument("--stream_bitrate", default="4M",
                   help="所有 H.264 出流的码率上限，如 4M/8M；留空则不限制")
    p.add_argument("--max_flv_clients", type=int, default=4)
    p.add_argument("--jpeg_quality", type=int, default=85)
    p.add_argument("--max_stream_fps", type=float, default=15.0)
    p.add_argument("--clip_dir", default="/opt/darkpipe/clips",
                   help="片段保存目录（容器内路径，建议挂载 NFS 后本地浏览）")
    p.add_argument("--clip_pre_sec", type=float, default=2.0)
    p.add_argument("--clip_post_sec", type=float, default=2.0)
    p.add_argument("--clip_max_sec", type=float, default=30.0)
    p.add_argument("--clip_skip_labels", default="other")
    # 唯一没有 default 的可选参数：规范里「有 default 即必填不能为空」，要允许留空就不能给
    # default。留空表示不推 HDFS，只保留 clip_dir 里的那一份。
    p.add_argument("--hdfs_output_dir", default="",
                   help="片段另存的 HDFS 目录，如 hdfs://用户名@ip:port/a/b；留空则不推送")
    p.add_argument("--run_seconds", type=float, default=0.0,
                   help="运行时长上限（秒），0 表示一直运行到容器被停止")
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


def write_session(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run(args):
    from darkpipe.config import PipelineConfig, validate
    from darkpipe.server import run_server

    gpus = parse_gpu_ids(args.gpu_ids)
    if len(gpus) > 1:
        # 实时流没有「未来的帧」可切段，多卡分片是离线专属能力（darkpipe.config 也会拒绝）
        print(f"[warn] 实时服务不做多卡分片，gpu_ids={args.gpu_ids} 只取第一张卡 {gpus[0]}")
    ensure_parent(args.session_json)

    session = run_dir_name()
    cfg = validate(PipelineConfig(
        mode="serve", input=args.video_path,
        enhance=args.enhance, sr=args.sr,
        sr_scale=(args.sr_scale if args.sr != "off" else None),
        recognize=args.recognize, device=f"cuda:{gpus[0]}",
        ckpt_dir=args.ckpt_dir, proc_max_side=args.proc_max_side,
        reco_span_sec=(args.reco_span_sec if args.reco_span_sec > 0 else None),
        no_label_bar=(not args.label_bar),
        host="0.0.0.0", port=args.serve_port, jpeg_quality=args.jpeg_quality,
        max_stream_fps=args.max_stream_fps,
        stream_formats=args.stream_formats, rtmp_push_url=args.rtmp_push_url,
        max_flv_clients=args.max_flv_clients, stream_bitrate=args.stream_bitrate,
        clip_dir=args.clip_dir, clip_pre_sec=args.clip_pre_sec,
        clip_post_sec=args.clip_post_sec, clip_max_sec=args.clip_max_sec,
        clip_skip_labels=args.clip_skip_labels,
        # One session name for the local tree, the HDFS tree and session_json alike; letting
        # each side stamp its own produced directories a second apart.
        clip_session=session))
    for w in cfg.warnings:
        print(f"[warn] {w}")
    print(f"[config] enhance={cfg.enhance} sr={cfg.sr_name()} recognize={cfg.recognize} "
          f"device={cfg.device} span={cfg.reco_span_sec} port={cfg.port}")

    hdfs_dir = (args.hdfs_output_dir or "").strip()
    sink = None
    if hdfs_dir:
        sink = HdfsClipSink(hdfs_dir, session)
        print(f"[config] 片段将另存到 {sink.root}")
    else:
        print("[config] 未填写 hdfs_output_dir，片段只保存在 clip_dir，不推送 HDFS")

    clip_root = os.path.join(args.clip_dir, session)
    payload = {
        "session": session, "status": "running", "source": args.video_path,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "endpoints": endpoints(args.serve_port, cfg.stream_formats, args.rtmp_push_url),
        "clip_dir": clip_root,
        "event_log": os.path.join(clip_root, "events.jsonl"),
        "hdfs_clip_dir": (sink.root if sink else ""),
        "clip_skip_labels": args.clip_skip_labels,
        "config": {"enhance": cfg.enhance, "sr": cfg.sr_name(), "recognize": cfg.recognize,
                   "gpu_ids": gpus[0], "reco_span_sec": cfg.reco_span_sec,
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


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        print("[serve] 收到中断，已退出")
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
