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

离线模式：**输入是一个有限长的文件**时，本算子自动转为离线推理——整段视频只处理一遍、
不循环播放，产出「整段处理后的视频（<视频名>-处理后.mp4）+ 按动作切分的片段视频
（按时间顺序+中文动作名命名，如 2-3s-喝水.mp4，与实时模式的 <时间戳>_<动作key>_<序号>
命名不同）+ 事件 JSON」，落到 clip_dir 与 hdfs_output_dir，然后进程正常退出。不起常驻
服务，run_seconds 与 rtmp_push_url 不生效。输入地址可以是 hdfs:// 或 WebHDFS 的
http://<namenode>:9870/<路径>，会先下载到本地再处理。
"""
import argparse
import faulthandler
import glob
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback

# 打包后 darkpipe/ 与 oputil.py 就在 main.py 同级；仓库内开发时 oputil.py 在上一级目录。
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]

from oputil import (resolve_ckpt_dir, apply_extra_hosts, deliver, ensure_parent,  # noqa: E402
                    make_remote_dir, parse_bool, parse_gpu_ids, resolve_input,
                    run_dir_name, source_tag, upload_file, OUTPUT_SUFFIX)


def build_parser():
    p = argparse.ArgumentParser(
        prog="main.py",
        description="暗光场景行为识别实时服务算子：拉流 -> 增强 -> 识别 -> 三条流 + 片段留存")
    # 参数名与 suanzi.json 的 inputs[].name / outputs[].name 逐一对应
    p.add_argument("--video_path", required=True,
                   help="待处理实时流：国标(GB28181)/rtsp:// / http(s):// (flv、hls)，"
                        "或容器内本地视频文件、hdfs:// 地址、WebHDFS 的 "
                        "http://<namenode>:9870/<路径> 地址。填文件时自动转为离线推理："
                        "整段只处理一遍、产出整段结果与动作片段后退出，不循环播放")
    p.add_argument("--enhance", default="retinexformer", choices=["off", "retinexformer"])
    p.add_argument("--sr", default="bicubic", choices=["off", "bicubic"])
    p.add_argument("--sr_scale", type=int, default=2, choices=[2, 3, 4])
    p.add_argument("--denoise", default="fast",
                   choices=["off","fast","quality","quality_high"],
                   help="画面去噪：off/fast(双边~5ms,35%%)/quality(NLM窗7~119ms,51%%)/quality_high(NLM窗15~376ms,75%%)；识别之后进行，实时流与片段共用同一帧")
    p.add_argument("--color_saturation", type=float, default=1.0,
                   help="画面色彩饱和度倍数，1.0=不处理；增强会把画面拉灰，2.0-2.6 可恢复色彩。在 Lab 空间缩放色度，色相不变；识别之后进行，不影响识别")
    # 不提供 off：本算子靠事件流切片段，没有识别器就既没有事件也没有片段
    p.add_argument("--recognize", default="behavior", choices=["behavior"])
    p.add_argument("--proc_max_side", type=int, default=840,
                   help="处理前把画面长边缩到不超过该值，0=按原分辨率。增强耗时与像素量成正比，而识别内部固定缩到 224，所以缩放不损识别精度。默认 840 是双卡下画质与帧率的折中(23.8 fps/p95 289 ms)；只分到 1 张卡时 840 只有 12.9 fps，达不到 15 fps，需改填 720")
    p.add_argument("--reco_span_sec", type=float, default=1.0)
    p.add_argument("--reco_min_conf", type=float, default=0.5,
                   help="行为判定阈值(0-1)：最高分的具名行为达不到该值时报为 other，既不显示也不存片段；0=关闭")
    p.add_argument("--skel_fusion_alpha", type=float, default=0.7,
                   help="骨架分支融合权重(0-1)：p=α·RGB+(1-α)·骨架；α≥0.999 完全关闭骨架分支按纯 RGB 运行。0.70 为现场三套件实测推荐档")
    p.add_argument("--skel_no_person", default="soft", choices=["other", "soft", "rgb"],
                   help="识别窗内骨架检测不到人时的处理：other=直接报 other(压误报最狠，但姿态漏检真行为时会被误杀)；soft=骨架以(1-α)权重投 other 票，RGB 证据足够强(RGB置信≥0.65/α)仍可报出行为，兼顾漏报与误报，默认；rgb=骨架弃权，该窗按纯 RGB 识别(误报回到 v1 水平)。仅 fuse_mode=alpha 时生效")
    p.add_argument("--fuse_mode", default="alpha", choices=["alpha", "max_prior"],
                   help="双分支决策模式：alpha=加权融合 p=α·RGB+(1-α)·骨架，四套件实测最优，默认。max_prior=会议三部曲(v2)——画面提亮前的原图照度低于 --lum_gate 时仅用 RGB；否则 RGB 与骨架(窗内最多6个合格骨架中非other且置信度>--skel_tau 者为骨架结果，无人合格则骨架弃权)分别分类取置信度高者为初步结果，置信度<--act_tau 直接报 other，捡东西/挥手/喝水/握手再过先验二次判断(不满足直接报 other)。注意三部曲四套件实测误报明显更多，仅作可选档保留")
    p.add_argument("--lum_gate", type=float, default=75.0,
                   help="三部曲步骤1的照度门(0-255，量的是画面提亮之前的原图平均亮度，不是提亮之后的；0=关闭)：低于该值说明场景真的太暗，姿态和检测本来就不可靠，只跑 RGB 识别。默认 75=现场全部视频实测的姿态/检测性能开始明显下降的亮度拐点，作为保险留用")
    p.add_argument("--act_tau", type=float, default=0.65,
                   help="三部曲步骤2的启动阈值(0-1)：初步结果(两分支中置信度高者)的置信度低于该值时直接报 other，不再做先验二次判断")
    p.add_argument("--skel_tau", type=float, default=0.65,
                   help="三部曲步骤2的骨架识别阈值(0-1)：合格骨架中仅有非other且置信度高于该值的人可当选骨架结果；无人合格则骨架弃权、采用 RGB 结果")
    p.add_argument("--label_bar", type=parse_bool, default=True)
    p.add_argument("--gpu_ids", default="0,1,2",
                   help="容器内 GPU 卡号(从 0 起编，与宿主机序号无关)，逗号分隔。填多张时最后一张专供行为识别，其余卡按帧轮询做增强(定序在第一张增强卡上)；填一张时增强与识别共用该卡。实际可见的卡少于填写的会自动降级并告警，真正决定给几张卡的是 suanzi.json 里的 metadata.gpu.count")
    p.add_argument("--ckpt_dir", default="ckpts",
                   help="权重目录。默认 ckpts 指算子包内随包发布的那一份；也可填绝对路径改用镜像内或挂载进来的权重")
    p.add_argument("--serve_port", type=int, default=8000)
    p.add_argument("--stream_formats", default="mjpeg,flv",
                   choices=["mjpeg", "mjpeg,flv", "mjpeg,flv,hls", "mjpeg,hls"],
                   help="对外的实时视频流格式；flv/hls 由镜像内的 ffmpeg 封装")
    p.add_argument("--rtmp_push_url", default="",
                   help="推流到外部流媒体服务器，如 rtmp://ip:1935/live/key；留空不推。"
                        "仅对实时流输入生效：video_path 是文件时走离线推理，此项被忽略")
    # These two defaults are kept equal to suanzi.json's, even though the platform always passes
    # them explicitly and the manifest is what production actually uses. A local `python main.py`
    # otherwise reproduces neither the deployment nor the bug being chased.
    p.add_argument("--stream_bitrate", default="3M",
                   help="所有 H.264 出流的码率上限，如 3M/4M/8M；0 或留空表示不限制")
    p.add_argument("--max_flv_clients", type=int, default=4)
    p.add_argument("--jpeg_quality", type=int, default=85)
    p.add_argument("--max_stream_fps", type=float, default=15.0)
    p.add_argument("--clip_dir", default="/opt/darkpipe/clips",
                   help="片段保存目录（容器内路径，建议挂载 NFS 后本地浏览）；"
                        "离线模式下整段处理结果也落在这里")
    p.add_argument("--clip_pre_sec", type=float, default=1.0)
    p.add_argument("--clip_post_sec", type=float, default=1.0)
    p.add_argument("--clip_max_sec", type=float, default=15.0)
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
    """决定这次是常驻服务、单文件离线、还是目录批处理，然后转给对应的实现。

    判据只看**取回来之后磁盘上的事实**，不看地址字符串长什么样：

      * 是一个文件 -> 离线。整段只处理一遍、产出整段视频与动作片段后退出，绝不循环播放，
        rtmp_push_url 在这条路径上被忽略（推流是给「永不结束的流」准备的）。hdfs:// 与
        WebHDFS 的 http:// 地址在这一步之前已经落成本地文件，所以「平台下发 HDFS 地址」也
        走这里。
      * 是一个目录 -> 离线批处理，里面的视频文件依次处理（见 run_offline_batch）。
      * 都不是、且看得出是流地址 -> 常驻实时服务。
      * 都不是、也不像流地址 -> `resolve_input` 直接退出。以前这里是「否则一律当实时流」，
        于是一个取不到的本地路径变成了无限重连、却报告健康的常驻服务。

    想拿本地文件演示实时流，用 ffmpeg 在算子外把文件循环推成流、再把流地址填进
    video_path（见 README）。
    """
    # 先补 /etc/hosts：WebHDFS 的 307 会跳到按主机名寻址的 DataNode，这一步要在任何取数据
    # 的动作之前完成。写不进去也只是告警——_webhdfs_call 里还有「改用 NameNode 地址重试」兜底。
    _BOOT.mark("写 /etc/hosts")
    apply_extra_hosts(args.extra_hosts)

    # 下载目录不能用 tempfile 的自动清理：实时模式下这个文件要陪着服务跑一整天。
    workdir = tempfile.mkdtemp(prefix="darkserve_")
    try:
        # 打点到这里是因为 hdfs://、WebHDFS http:// 会在这一步整段下载，而下载卡住时在
        # 「[mode]」之前，此前日志里一个字都没有。rtsp:// 一类原样返回，瞬间就过。
        _BOOT.mark(f"取输入 {args.video_path}")
        kind, payload = resolve_input(args.video_path, workdir, "input.mp4")
        if kind == "batch":
            return run_offline_batch(args, payload, workdir)
        # 目录名带上输入的标识，不然一堆 20260821_164652_78 谁也说不出是哪路输入的结果。
        # 标识取自 video_path（原始地址）而不是取回来那份：HDFS 输入下载到本地就叫
        # input.mp4，文件名已经丢了。
        session = run_dir_name(args.video_path)
        if kind == "file":
            src = payload
            n, fps = probe_video(src)
            size = os.path.getsize(src) / 1e6
            print(f"[mode] 离线推理：输入是有限长文件（{size:.0f} MB / "
                  f"{n or '?'} 帧 / {fps:.2f} fps），整段只处理一遍、不循环播放，"
                  f"产出整段处理后的视频并按动作切片（命名与实时模式不同），完成即退出")
            if (args.rtmp_push_url or "").strip():
                print("[mode] rtmp_push_url 对文件输入不生效，本次忽略；"
                      "要推流请把 video_path 换成实时流地址")
            return run_offline_mode(args, src, session, workdir)
        print("[mode] 实时服务：输入是实时流，常驻运行并按动作切片")
        return run_serve_mode(args, payload, session)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _offline_gpus(args):
    """离线这一路实际能用的卡。切片与多卡分片不兼容，此时只留第一张。"""
    gpus = parse_gpu_ids(args.gpu_ids)
    if (args.clip_dir or "").strip() and len(gpus) > 1:
        # Clips are cut from one continuous timeline by a single ClipRecorder/writer thread;
        # --gpus sharding hands each GPU an independent, non-overlapping frame range with no
        # cross-shard coordination, so the two cannot be reconciled. clip_dir has a non-empty
        # default, so this triggers for real whenever gpu_ids names more than one card.
        print(f"[config] 已启用切片保存，与多卡分片(--gpus)不兼容：本次仅用 "
              f"cuda:{gpus[0]}（放弃 {','.join(gpus[1:])}）")
        gpus = gpus[:1]
    return gpus


class _FatalBatchError(RuntimeError):
    """环境级错误：下一个文件必然重现，继续跑只是白烧 GPU 时间。见 run_offline_batch。"""


def run_offline_one(args, src, session, workdir, stem, gpus, verbose=True):
    """处理一个文件并就地交付，返回这一项的结果 dict。不写 session_json。

    单文件离线与目录批处理共用这一段——批处理只是把 session 变成 `<会话>/<stem>` 并循环调用，
    于是「已经端到端验证过的那条路径」在两种形态下逐字相同。
    """
    from darkpipe.config import PipelineConfig, validate

    out_video = os.path.join(workdir, f"{stem}{OUTPUT_SUFFIX}")
    out_events = os.path.join(workdir, f"{stem}_events.json")
    # Staged locally under workdir, then merged into the same deliver() call as out_video/
    # out_events below -- clip_dir on the platform side is the *delivered* destination, and
    # naming="range" (set inside run_offline itself once clip_dir is non-empty) already
    # leaves each clip's local basename equal to its final <start>-<end>s-<中文动作>.mp4 name.
    clip_dir_local = os.path.join(workdir, "clips")
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
        no_label_bar=(not args.label_bar),
        skel_fusion_alpha=args.skel_fusion_alpha,
        skel_no_person=args.skel_no_person,
        fuse_mode=args.fuse_mode, lum_gate=args.lum_gate,
        act_tau=args.act_tau, skel_tau=args.skel_tau,
        clip_dir=clip_dir_local, clip_pre_sec=args.clip_pre_sec,
        clip_post_sec=args.clip_post_sec, clip_max_sec=args.clip_max_sec,
        clip_skip_labels=args.clip_skip_labels, clip_denoise=args.clip_denoise,
        clip_session=session))
    if verbose:
        # 配置与文件无关，批处理下打一次就够；打 N 次只会把 [batch] 进度挤到要翻页才找得到。
        for w in cfg.warnings:
            print(f"[warn] {w}")
        print(f"[config] enhance={cfg.enhance} sr={cfg.sr_name()} recognize={cfg.recognize} "
              f"device={cfg.device} gpu_ids={','.join(gpus)} span={cfg.reco_span_sec}"
              f" reco_min_conf={cfg.reco_min_conf} proc_max_side={cfg.proc_max_side}"
              f" fuse_mode={cfg.fuse_mode} lum_gate={cfg.lum_gate} act_tau={cfg.act_tau} skel_tau={cfg.skel_tau} color_saturation={cfg.color_saturation}")
        print("[config] 离线模式：保存整段处理后的视频，并按动作切分保存片段"
              "（clip_* 系列参数生效）")
        # Same reason as the serve-mode line below: these are manifest-settable and had no
        # printed trace, so "the default I changed did not take effect" was undiagnosable.
        print(f"[config] clip_pre_sec={cfg.clip_pre_sec:g} clip_post_sec={cfg.clip_post_sec:g}"
              f" clip_max_sec={cfg.clip_max_sec:g} clip_denoise={cfg.clip_denoise}"
              f" clip_min_conf={cfg.clip_min_conf:g}"
              f" clip_skip_labels={cfg.clip_skip_labels or '（不跳过）'}")
        print("[config] 离线模式下 run_seconds 不生效，整段视频跑完即退出")

    t0 = time.time()
    # 离线也打这个点，理由同实时：run_offline 里要加载权重、预热、开解码/编码线程，任何一步
    # 卡住在打点之前都只是「日志停在 [config]」。离线跑完就退，所以打点之后不会再有停留。
    if verbose:
        _BOOT.mark("离线处理（加载权重 -> 预热 -> 解码/处理/编码）")
    if len(gpus) > 1:
        from darkpipe.shard import run_offline_sharded
        events = run_offline_sharded(cfg, gpus)
    else:
        from darkpipe.pipeline import run_offline
        events = run_offline(cfg, on_ready=_BOOT.done)
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

    # clip_dir_local/<session>/*.mp4 : ClipRecorder(naming="range") already left each local
    # basename equal to its final <start>-<end>s-<中文动作>.mp4 name, so dest_name below is
    # just that basename -- not a rename, a guarantee it survives deliver()'s own defaulting.
    files = {"output_video": out_video, "events_json": out_events}
    labels = {"output_video": "整段处理结果", "events_json": "识别事件"}
    clip_names = []
    for mp4 in sorted(glob.glob(os.path.join(clip_dir_local, session, "*.mp4"))):
        base = os.path.splitext(os.path.basename(mp4))[0]
        clip_names.append(base)
        files[f"clip_video_{base}"] = (mp4, os.path.basename(mp4))
        labels[f"clip_video_{base}"] = f"动作片段：{base}"
        meta = os.path.join(clip_dir_local, session, base + ".json")
        if os.path.exists(meta):
            files[f"clip_meta_{base}"] = (meta, os.path.basename(meta))
            labels[f"clip_meta_{base}"] = f"动作片段元数据：{base}"
    if clip_names:
        print(f"[deliver] 动作片段 {len(clip_names)} 段：{', '.join(clip_names)}")

    try:
        landed = deliver(files, args.clip_dir, hdfs_dir, labels, run=session)
    except Exception as e:                    # noqa: BLE001 - 分类之后原样抛出
        # 离线模式一向把交付失败当致命（跑完就退，产出取不回来等于这次 GPU 时间白花）。
        # 批处理下它还多一层理由：目标目录/HDFS 不可写这类问题，下一个文件必然重现。
        raise _FatalBatchError(f"交付失败：{type(e).__name__}: {e}") from e

    return {
        "file": src, "stem": stem, "session": session, "status": "finished",
        "local_input": src, "clip_names": clip_names,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
        "stopped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_video": landed.get("output_video", []),
        "events_json": landed.get("events_json", []),
        "events_total": len(events), "label_counts": counts,
        "clips": st.get("clips") or {},
        "clip_videos": [landed[f"clip_video_{n}"] for n in clip_names
                        if f"clip_video_{n}" in landed],
        "stats": dict(st, wall_seconds=round(dt, 3)),
        "config": {"enhance": cfg.enhance, "sr": cfg.sr_name(), "sr_scale": cfg.sr_scale,
                   "recognize": cfg.recognize, "gpu_ids": ",".join(gpus),
                   "reco_span_sec": cfg.reco_span_sec,
                   "proc_max_side": cfg.proc_max_side},
    }


def run_offline_mode(args, src, session, workdir):
    """单文件离线：跑完整段、交付、退出。产出与 session_json 的形状与本次改动前逐字一致。"""
    gpus = _offline_gpus(args)
    _print_gpu_inventory(gpus)
    ensure_parent(args.session_json)

    # 文件名同样取自原始地址：HDFS 输入落到本地叫 input.mp4，用它命名产出就成了
    # input-处理后.mp4，一样看不出是哪个视频。
    stem = (source_tag(args.video_path)
            or os.path.splitext(os.path.basename(src))[0] or "output")
    item = run_offline_one(args, src, session, workdir, stem, gpus)

    payload = {"session": session, "mode": "offline", "status": "finished",
               "source": args.video_path}
    payload.update({k: item[k] for k in (
        "local_input", "started_at", "stopped_at", "output_video", "events_json",
        "events_total", "label_counts", "clips", "clip_videos", "stats", "config")})
    write_session(args.session_json, payload)
    print(f"[done] 会话信息 -> {args.session_json}")
    print(f"[done] 离线推理完成，共 {len(item['clip_names'])} 段动作片段，进程退出")


def _batch_source(video_path, paths):
    """批处理会话名要用的「源」字符串（会再过一遍 run_dir_name -> source_tag）。

    只有 1 个文件时用那个文件本身，于是「目录里就一个视频」与「直接给一个文件」产出完全同一
    个形状——平台侧任何现有解析都不会因为这次改动而失效。
    多个文件时用原始 video_path；但平台把参数值物化成 `/mnt/nfs/args/<参数下标>`，目录名就
    只是个下标，拿它当片段目录名毫无意义（会得到 `1/20260901_.../`，正是上一轮改动要消灭的
    那类名字），这种情况退回字面量 `batch`——诚实，且一眼看得出是一批。
    """
    if len(paths) == 1:
        return paths[0]
    tag = source_tag(video_path)
    return video_path if tag and not tag.isdigit() else "batch"


def _unique_stems(paths):
    """每个文件的产出词干，保证互不相同。

    重名真会发生（扫描是两层的，`a/demo.mp4` 与 `b/demo.mp4` 会同时进来），而 stem 同时是
    产出子目录名和 `{stem}-处理后.mp4` 的词干，撞名就是互相覆盖。
    """
    seen, out = set(), []
    for p in paths:
        base = source_tag(p) or "video"
        cand, k = base, 1
        while cand in seen:
            k += 1
            cand = f"{base}_{k}"
        seen.add(cand)
        out.append(cand)
    return out


def _is_environment_error(exc):
    """这是环境的问题还是这一个文件的问题？环境的问题在下一个文件上必然重现。"""
    if isinstance(exc, _FatalBatchError):
        return True
    s = f"{type(exc).__name__}: {exc}".lower()
    return any(k in s for k in ("out of memory", "no space left", "disk quota",
                                "cuda error", "cudnn", "device-side assert"))


def _write_batch_session(args, session, paths, items, t0):
    """把一批的结果写进 session_json。

    只**新增**字段、不改动既有字段的含义。目录里只有 1 个文件时，顶层每个字段的取值与今天
    单文件运行**逐字段相同**（连 `<会话>/<stem>/` 那一层都不加，见 run_offline_batch），
    `items` 长度为 1——这样平台侧任何现有解析都不会因为这次改动而失效。
    多文件时 events_total / clips / stats 是合计，output_video / events_json / clip_videos
    是各文件拼接。mode 仍然是 `offline`（不改成 offline_batch，否则按 mode 分流的平台侧逻辑
    会认不出来）。
    """
    ok = [it for it in items if it["status"] == "finished"]
    counts = {}
    for it in ok:
        for k, v in (it.get("label_counts") or {}).items():
            counts[k] = counts.get(k, 0) + v
    payload = {
        "session": session, "mode": "offline",
        "status": "finished" if len(ok) == len(paths) else "partial",
        "source": args.video_path,
        "local_input": paths[0] if len(paths) == 1 else os.path.commonpath(paths),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
        "stopped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files_total": len(paths), "files_ok": len(ok),
        "files_failed": len(paths) - len(ok),
        "output_video": [p for it in ok for p in it.get("output_video", [])],
        "events_json": [p for it in ok for p in it.get("events_json", [])],
        "events_total": sum(it.get("events_total", 0) for it in ok),
        "label_counts": counts,
        "clips": _merge_clips(session, ok),
        "clip_videos": [p for it in ok for p in it.get("clip_videos", [])],
        "stats": _merge_stats(ok),
        "items": items,
    }
    if ok:
        # 配置对整批是同一份（它与文件无关），取第一个成功项的即可。
        payload["config"] = ok[0]["config"]
    write_session(args.session_json, payload)
    print(f"[done] 会话信息 -> {args.session_json}")


def _merge_clips(session, ok):
    """合计各文件的片段计数。只有 1 个文件时原样返回它自己的，保证与单文件运行逐字段相同。"""
    if len(ok) == 1:
        return ok[0].get("clips") or {}
    out = {"session": session, "clips_saved": 0, "clips_abandoned": 0, "frames_dropped": 0}
    for it in ok:
        for k, v in (it.get("clips") or {}).items():
            if isinstance(v, int):
                out[k] = out.get(k, 0) + v
    return out


def _merge_stats(ok):
    """合计各文件的处理统计。

    只有 1 个文件时原样返回。多文件时把可加的量相加，**fps 按合计重算**而不是求平均——
    各文件长短不一，平均 fps 会被一个 3 秒的小文件带偏，而「总帧数 / 总耗时」才是这次运行
    真实的吞吐。
    """
    if len(ok) == 1:
        return ok[0].get("stats") or {}
    out = {}
    for it in ok:
        for k, v in (it.get("stats") or {}).items():
            if k in ("frames", "seconds", "frames_written", "events", "wall_seconds"):
                out[k] = round(out.get(k, 0) + v, 3)
    out["fps"] = round(out.get("frames", 0) / max(out.get("seconds", 0), 1e-9), 2)
    out["files"] = len(ok)
    return out


def run_offline_batch(args, paths, workdir):
    """目录输入：里面的视频文件依次处理，每个文件跑完立刻交付。

    为什么是「依次」而不是并行：GPU 已经是瓶颈，同卡并行提交只会互相切分时间片。同类实测见
    CLAUDE.md——把识别单独拆一个线程与增强共卡时，推理从 73 ms 涨到 250 ms。

    为什么每个文件都重新加载一遍权重（`run_offline` 每次调用都 build_stages + load）：
    recognize.py 的 warmup docstring 记着首次推理 **4611 ms**、稳态 **235 ms**，而那 4611 ms
    是 cuDNN autotune + lazy CUDA module load + fp16 autocast 初始化，**三样都是进程级**的，
    不随第二次 load 重付；权重本身只有 34.5 MB。所以第 2..N 个文件只多付「模型构造 + 上卡」，
    几百毫秒量级。为省这个去把 run_offline 拆成 load/process 两半，赔的是这个包里唯一一条
    端到端验证过的离线路径，还要自己记得调 `recognizer.reset()`（忘了是静默错误，见
    stages/base.py 的 reset docstring）——不值得。

    每个文件跑完**立刻**交付，不攒到最后：长跑任务被杀时已完成的部分要保得住，这与实时模式
    「上传失败只告警、因为本地片段还在」是同一个精神。

    失败策略与 CLAUDE.md 的 fail-fast 约定有一处**有意的例外**：单个文件打不开/解码失败只
    跳过并继续，末尾非零退出。理由是「第 3 个坏了就丢掉后面 7 个的 GPU 时间」与那条约定背后
    的「产出取不回来等于白花 GPU 时间」是同一个诉求。环境级错误（交付失败、CUDA OOM、盘满）
    仍然当场中断——它们在下一个文件上只会重演。
    """
    gpus = _offline_gpus(args)
    _print_gpu_inventory(gpus)
    ensure_parent(args.session_json)
    session = run_dir_name(_batch_source(args.video_path, paths))
    stems = _unique_stems(paths)
    # 只有 1 个文件时不加 `<会话>/<stem>/` 这一层：分层是为了防止两个视频的
    # `2-3s-喝水.mp4` 互相覆盖（离线片段是平铺在会话目录下的），一个文件没有可撞的对象，
    # 加了只会得到 `demo_dark/20260901_.../demo_dark/` 这种把名字说两遍的路径。省掉之后
    # 「目录里就一个视频」与「直接给一个文件」产出完全同一个形状。
    subs = [f"{session}/{s}" for s in stems] if len(paths) > 1 else [session]
    print(f"[mode] 离线批处理：目录输入，{len(paths)} 个视频文件依次处理，"
          + (f"每个文件一个产出子目录 <会话>/<文件名>/，" if len(paths) > 1 else "")
          + "全部完成即退出")
    if (args.rtmp_push_url or "").strip():
        print("[mode] rtmp_push_url 对目录输入不生效，本次忽略")

    items, t0 = [], time.time()
    for i, (src, stem, sub) in enumerate(zip(paths, stems, subs), 1):
        n, fps = probe_video(src)
        print(f"[batch] {i}/{len(paths)} 开始 {src}"
              f"（{os.path.getsize(src) / 1e6:.0f} MB / {n or '?'} 帧 / {fps:.2f} fps）"
              f" -> {sub}")
        try:
            items.append(run_offline_one(args, src, sub, workdir, stem,
                                         gpus, verbose=(i == 1)))
        except Exception as e:                # noqa: BLE001 - 分类后决定跳过还是中断
            reason = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            items.append({"file": src, "stem": stem, "session": sub,
                          "status": "failed", "error": reason})
            if _is_environment_error(e):
                print(f"[batch] {i}/{len(paths)} 失败：{reason}")
                print(f"[batch] 这是环境级错误，剩下 {len(paths) - i} 个文件必然重现，"
                      f"中断整批")
                _write_batch_session(args, session, paths, items, t0)
                sys.exit(f"error: 批处理中断于 {src}：{reason}")
            print(f"[batch] {i}/{len(paths)} 失败：{reason}（已记录，继续下一个）")
        else:
            print(f"[batch] {i}/{len(paths)} 完成 {stem}："
                  f"{items[-1]['events_total']} 个事件、"
                  f"{len(items[-1]['clip_videos'])} 段动作片段")

    _write_batch_session(args, session, paths, items, t0)
    bad = [it for it in items if it["status"] != "finished"]
    print(f"[done] 离线批处理完成：{len(paths)} 个文件，{len(paths) - len(bad)} 成功、"
          f"{len(bad)} 失败")
    if bad:
        for it in bad:
            print(f"[done] 失败：{os.path.basename(it['file'])}（{it['error']}）")
        # 非零退出：有文件没跑成，这次运行不干净，平台不该看成成功。已写出的
        # session_json 会被 write_failed_session 就地补上 status/error，items 保留。
        sys.exit(f"error: {len(bad)}/{len(paths)} 个视频处理失败，详见日志与 session_json")


def run_serve_mode(args, src, session):
    """常驻拉流服务：三条对外流 + 按动作切片。这条路径的行为与本次改动前逐字一致。"""
    from darkpipe.config import PipelineConfig, validate
    from darkpipe.server import run_server

    gpus = parse_gpu_ids(args.gpu_ids)
    # 多卡时最后一张卡专供行为识别，其余卡做增强（≥2 张增强卡时按帧轮询分发，第 1 帧给卡 A、
    # 第 2 帧给卡 B……，不是离线那种按帧段切片——实时流没有「未来的帧」可切，但轮询也不需要）。
    # 识别独占一张卡是为了守住「一张卡上只有一个提交 CUDA 任务的线程」：同卡两个提交线程会互相
    # 时间片切分，实测推理从 73ms 涨到 250ms。此前识别与增强同线程，每次推理都会把整条出流路径
    # 卡住 242ms，实测出流时延在 74ms 与 327ms 之间反复跳变、20% 推出去的帧是重复帧。
    if len(gpus) > 1:
        print(f"[serve] 增强 {', '.join('cuda:' + g for g in gpus[:-1])}"
              f"{'（按帧轮询）' if len(gpus) > 2 else ''}；"
              f"识别独占 cuda:{gpus[-1]}")
    # 这一步是本进程第一次 import torch 并初始化 CUDA。vGPU 切分下它可以很慢，而在打点之前
    # 它是「日志停住」最容易被误读成 hang 的一段。
    _BOOT.mark("清点 GPU（首次 import torch / 初始化 CUDA）")
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
        skel_fusion_alpha=args.skel_fusion_alpha,
        skel_no_person=args.skel_no_person,
        fuse_mode=args.fuse_mode, lum_gate=args.lum_gate,
        act_tau=args.act_tau, skel_tau=args.skel_tau,
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
          f" fuse_mode={cfg.fuse_mode} lum_gate={cfg.lum_gate} act_tau={cfg.act_tau} skel_tau={cfg.skel_tau} color_saturation={cfg.color_saturation}")
    # The output path's own knobs, on their own line because they are the ones that decide what
    # a viewer actually receives, and because none of them used to be printed at all. That cost a
    # deployment ~10% of its stream cadence: jpeg_quality was 100 everywhere, 598 KB JPEGs were
    # saturating ffmpeg's stdin, and the only visible trace was `pipe=65Mbps` in the [push]
    # telemetry -- the actual quality value had to be inferred from it rather than read.
    print(f"[config] formats={cfg.stream_formats} max_stream_fps={cfg.max_stream_fps:g}"
          f" jpeg_quality={cfg.jpeg_quality}"
          f" stream_bitrate={cfg.stream_bitrate or 'off（不限）'}")
    # The clip knobs, on their own line for the same reason the output line exists: they are
    # settable from the manifest, so they carry the same "the platform is still building the
    # command line from an older suanzi.json" failure mode, and they were the only group of
    # parameters with no printed trace at all. That cost a diagnosis: with clip_denoise=quality
    # the NLM pass ran ~160 ms/frame over the whole pending buffer inside one synchronous call,
    # holding the processing thread for 1.3-6.8 s per clip close (measured p90 latency 3251 ms,
    # wait_cpu 86%, and up to 174 consecutive re-sends of the same JPEG on the output side).
    # Whether the fix had actually been deployed could only be inferred from wait_cpu dropping
    # to 0-1% and from every clip suddenly closing on clip_max_sec instead of post_sec.
    if cfg.clip_dir:
        print(f"[config] clip_pre_sec={cfg.clip_pre_sec:g} clip_post_sec={cfg.clip_post_sec:g}"
              f" clip_max_sec={cfg.clip_max_sec:g} clip_denoise={cfg.clip_denoise}"
              f" clip_min_conf={cfg.clip_min_conf:g}"
              f" clip_skip_labels={cfg.clip_skip_labels or '（不跳过）'}")

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

    # 最后一个打点。此后一切都在 run_server 里：加载权重、预热、开线程、起 uvicorn。这一段
    # 里任何一步卡住（识别权重 load 到某张卡上、warmup、cv2 打开 rtsp——它没有超时，可以无限期
    # 阻塞），在打点之前都只表现为「日志停在 [done] 会话信息」。看护由 on_ready 撤销——首帧产出
    # 时——而不是等 run_server 返回：它常驻，永远不返回（见 _BootWatch.done）。
    _BOOT.mark("启动服务（加载权重 -> 预热 -> 开线程 -> uvicorn）")
    st = run_server(cfg, on_clip=sink, stop_after=args.run_seconds, on_ready=_BOOT.done)
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

    已经有内容时**就地补写**而不是覆盖：实时模式启动时已写过一份 `running`（带会话名与目录），
    批处理每跑完一个文件也会刷新一次（带 items）。整份换成一个只有 error 的骨架，等于把「哪
    些文件已经跑成了、产出在哪」这些还有用的信息一起丢掉——而这些恰恰是失败之后最想知道的。
    """
    path = (getattr(args, "session_json", "") or "").strip()
    if not path:
        return
    try:
        ensure_parent(path)
        payload = {}
        try:
            with open(path, encoding="utf-8") as f:
                old = json.load(f)
            if isinstance(old, dict):
                payload = old
        except Exception:                             # noqa: BLE001 - 没有/坏了就当没有
            pass
        payload.update({
            "status": "failed",
            "error": reason,
            "source": getattr(args, "video_path", ""),
            "stopped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        write_session(path, payload)
        print(f"[failed] 失败原因已写入 {path}")
    except BaseException as e:                        # noqa: BLE001 - 收尾不能盖住真错误
        print(f"[warn] 失败信息写入 {path} 失败: {e}")


class _BootWatch:
    """把「卡在哪一步」变成日志里的一行，并在停住时转储全部线程栈。

    为什么需要它：出现过一次「完全没有任何日志输出，像 hang 住」的部署。这种症状下所有既有
    遥测都无用——`[process]`/`[push]` 要等线程跑起来才打，而问题恰恰是没跑起来。容器又登不
    进去，`py-spy`/`gdb` 都用不上，所以唯一的办法是让进程自己说它停在哪。

    三件事分别针对三种「静默」：
      * `mark()` 逐步打点，于是「零输出」与「停在第 N 步」不再是同一个现象；
      * 栈转储走 **stdout**，因为本算子的日志约定是 stdout（见模块 docstring），而
        faulthandler 默认写 stderr——平台只收 stdout 时那等于没写；
      * `faulthandler.enable()` 兜住原生崩溃：mamba_ssm / causal_conv1d 的 selective-scan
        是 CUDA 原生代码，段错误或 abort 不会留下 Python 堆栈，进程就那么消失。

    诊断绝不影响运行：这里所有异常一律吞掉改为打印（与 `probe_video`、
    `_print_gpu_inventory` 同一条约定）。
    """

    def __init__(self, stall=45.0, max_dumps=3):
        self._stage = "进程启动"
        self._t = time.time()
        self._dumps = 0
        self._stall = stall
        self._max = max_dumps
        self._off = False
        self._lock = threading.Lock()

    def start(self):
        try:
            # stdout 而不是 stderr：见类 docstring。拿不到 fileno（被包成非文件对象）时
            # 放弃这一层，打点与栈转储两层仍然有效。
            faulthandler.enable(file=sys.stdout)
        except Exception as e:                        # noqa: BLE001
            print(f"[boot] faulthandler 未启用（不影响运行）：{e}", flush=True)
        print(f"[boot] 进程启动 pid={os.getpid()} python={sys.version.split()[0]} "
              f"cwd={os.getcwd()}", flush=True)
        threading.Thread(target=self._loop, daemon=True, name="bootwatch").start()

    def mark(self, stage):
        with self._lock:
            self._stage, self._t = stage, time.time()
        print(f"[boot] {stage}", flush=True)

    def done(self, why):
        """撤哨。启动结束之后再转储栈就是纯噪声。

        必须有这一步，因为最后一个打点「启动服务」之后 `run_server` 就再也不返回了
        （uvicorn 常驻），于是「一直停在这一步」既是卡死的样子，也是**健康服务**的样子。
        实测代价：一次正常的 60 秒运行里，栈转储占了 609 行日志中的 254 行（42%），把
        `[process]`/`[push]` 遥测挤到要翻页才找得到——正是本类 docstring 里说要避免的事。

        撤哨的判据是「整条管线产出过一帧」，不是「uvicorn 起来了」：权重加载、预热、
        `cv2` 打开 rtsp 全都发生在 uvicorn 之后，而它们才是当初怀疑会卡住的地方。
        """
        with self._lock:
            if self._off:
                return
            self._off = True
        print(f"[boot] 启动完成（{why}），撤销卡死看护", flush=True)

    def _loop(self):
        # 转储次数封顶：卡死时栈每隔 stall 秒都长一样，刷满日志反而把有用的东西挤掉。
        # 三次足以区分「慢」（栈在动）与「死」（三次完全一致）。
        while self._dumps < self._max and not self._off:
            time.sleep(1.0)
            with self._lock:
                stage, waited = self._stage, time.time() - self._t
            if waited < self._stall:
                continue
            self._dumps += 1
            print(f"[boot] 已在「{stage}」停留 {waited:.0f}s，转储全部线程栈"
                  f"（第 {self._dumps}/{self._max} 次）", flush=True)
            try:
                faulthandler.dump_traceback(file=sys.stdout)
                sys.stdout.flush()
            except Exception as e:                    # noqa: BLE001
                print(f"[boot] 栈转储失败（不影响运行）：{e}", flush=True)
            with self._lock:
                self._t = time.time()                 # 下一次再等满一个 stall


_BOOT = _BootWatch()


def main(argv=None):
    _BOOT.start()
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as e:
        # argparse 把用法与错误写到 stderr。平台只收 stdout 时，一个参数名对不上就表现为
        # 「进程起来了、什么都没打、退出」——正是要排除的那种静默。所以在 stdout 留一句。
        if e.code not in (0, None):
            print(f"[boot] 命令行参数被拒绝（退出码 {e.code}），详情在 stderr。"
                  f"suanzi.json 的 inputs[]/args[] 必须与 main.py 的 argparse 逐一对应，"
                  f"改过参数就要重新导入清单", flush=True)
        raise
    _BOOT.mark("参数解析完成")
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
