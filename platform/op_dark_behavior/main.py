"""暗光场景行为识别算子（处理算子）——平台入口。

本文件是薄适配层，不重复实现任何算法：解析平台参数 -> 组装 PipelineConfig -> 调用
darkpipe.config.validate 校验 -> 调用 darkpipe.pipeline.run_offline（多卡时
darkpipe.shard.run_offline_sharded）执行。与命令行 `darkpipe` 完全同一套代码路径。

平台规范在本文件中的落实：
  * 主入口文件名 main.py，与 suanzi.json 的 command 一致；
  * 所有 outputPath 在运行前先创建父目录（框架给的是绝对路径，父目录不保证存在）；
  * 日志一律 print 到标准输出，由框架收集；
  * 失败时打印堆栈并以非零码退出，框架据此判定任务失败。
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

# 打包后 darkpipe/ 与 oputil.py 就在 main.py 同级；仓库内开发时 oputil.py 在上一级目录
# （platform/，由打包脚本复制进 zip），darkpipe 则由仓库根目录提供。两条路径都加上。
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]

from oputil import (resolve_ckpt_dir, deliver, ensure_parent, fetch_input, parse_bool,  # noqa: E402
                     parse_gpu_ids, require_destination, run_dir_name)


def build_parser():
    p = argparse.ArgumentParser(
        prog="main.py", description="暗光场景行为识别算子：低照度增强 + 超分辨率 + 行为识别")
    # 参数名与 suanzi.json 的 inputs[].name / outputs[].name 逐一对应
    p.add_argument("--video_path", required=True,
                   help="待处理视频：国标(GB28181)/rtsp:// / http(s):// (flv、hls) 实时流地址，"
                        "或容器内本地路径；不使用 hdfs://")
    p.add_argument("--enhance", default="retinexformer", choices=["off", "retinexformer"])
    p.add_argument("--sr", default="bicubic", choices=["off", "bicubic"])
    p.add_argument("--sr_scale", type=int, default=2, choices=[2, 3, 4])
    p.add_argument("--denoise", default="fast",
                   choices=["off","fast","quality","quality_high"],
                   help="画面去噪：off/fast(双边~5ms,35%)/quality(NLM窗7~119ms,51%)/quality_high(NLM窗15~376ms,75%)；识别之后进行，实时流与片段共用同一帧")
    p.add_argument("--color_saturation", type=float, default=1.0,
                   help="画面色彩饱和度倍数，1.0=不处理；增强会把画面拉灰，2.0-2.6 可恢复色彩。在 Lab 空间缩放色度，色相不变；识别之后进行，不影响识别")
    p.add_argument("--recognize", default="behavior", choices=["off", "behavior"])
    p.add_argument("--reco_span_sec", type=float, default=1.0)
    p.add_argument("--reco_min_conf", type=float, default=0.8,
                   help="行为判定阈值(0-1)：最高分的具名行为达不到该值时报为 other，既不显示也不存片段；0=关闭")
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--label_bar", type=parse_bool, default=True)
    p.add_argument("--gpu_ids", default="0")
    # 默认 4：实测 640x480 下 chunk 4 与 8 吞吐量持平（27.3 / 27.9 fps），显存却只要一半
    # （2.38 GiB vs 4.70 GiB）。显存紧张时可继续调小，代价是吞吐量线性下降。
    p.add_argument("--proc_max_side", type=int, default=0,
                   help="处理前把画面长边缩到不超过该值，0=按原分辨率。增强耗时与像素量成正比，而识别内部固定缩到 224，1080p 填 1280 可省四分之三算力且不损识别精度")
    p.add_argument("--enhance_chunk", type=int, default=4)
    p.add_argument("--ckpt_dir", default="ckpts",
                   help="权重目录。默认 ckpts 指算子包内随包发布的那一份；也可填绝对路径改用镜像内或挂载进来的权重")
    p.add_argument("--local_output_dir", default="",
                   help="产出另存到这个容器内目录（挂载 NFS 后即可本地浏览），如 /mnt/nfs/darkout；"
                        "留空则不另存本地")
    p.add_argument("--hdfs_output_dir", default="",
                   help="产出上传到这个 HDFS 目录，如 hdfs://用户名@ip:port/a/b；留空则不上传。"
                        "与 local_output_dir 至少要填一个，否则结果随容器一起消失")
    p.add_argument("--output_video", required=True)
    p.add_argument("--events_json", required=True)
    p.add_argument("--summary_json", required=True)
    return p


def frame_count(path):
    import cv2
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(n, 0)


def to_dicts(events):
    """run_offline 返回 RecognitionEvent，run_offline_sharded 返回已 to_dict 的字典。"""
    return [e if isinstance(e, dict) else e.to_dict() for e in events]


def run(args):
    from darkpipe.config import PipelineConfig, validate

    require_destination(args.local_output_dir, args.hdfs_output_dir)
    gpus = parse_gpu_ids(args.gpu_ids)
    for p in (args.output_video, args.events_json, args.summary_json):
        ensure_parent(p)

    # 框架给的 outputPath 只保证是绝对文件路径，不保证带扩展名，而 cv2 的 VideoWriter 靠
    # 扩展名选容器，路径不带 .mp4 就直接打不开。因此先写到带扩展名的临时文件，收尾时改名到
    # 框架指定的路径——算子不去假设框架怎么命名。
    vid_tmp = args.output_video
    if os.path.splitext(vid_tmp)[1].lower() not in (".mp4", ".avi", ".mkv", ".mov"):
        vid_tmp = args.output_video + ".mp4"

    workdir = tempfile.mkdtemp(prefix="darkop_")
    try:
        src = fetch_input(args.video_path, workdir, "input.mp4")
        cfg = validate(PipelineConfig(
            mode="offline", input=src, output=vid_tmp,
            enhance=args.enhance, sr=args.sr,
            sr_scale=(args.sr_scale if args.sr != "off" else None),
            recognize=args.recognize, device=f"cuda:{gpus[0]}",
            gpus=(",".join(gpus) if len(gpus) > 1 else ""),
            ckpt_dir=resolve_ckpt_dir(args.ckpt_dir, _HERE), reco_min_conf=args.reco_min_conf, proc_max_side=args.proc_max_side, color_saturation=args.color_saturation, denoise=args.denoise, enhance_chunk=max(1, args.enhance_chunk),
            reco_span_sec=(args.reco_span_sec if args.reco_span_sec > 0 else None),
            no_label_bar=(not args.label_bar),
            events_json=args.events_json,
            max_frames=(args.max_frames or None)))
        for w in cfg.warnings:
            print(f"[warn] {w}")
        print(f"[config] enhance={cfg.enhance} sr={cfg.sr_name()} recognize={cfg.recognize} "
              f"device={cfg.device} gpu_ids={','.join(gpus)} span={cfg.reco_span_sec}"
          f" reco_min_conf={cfg.reco_min_conf} proc_max_side={cfg.proc_max_side}"
          f" color_saturation={cfg.color_saturation}")

        t0 = time.time()
        if len(gpus) > 1:
            from darkpipe.shard import run_offline_sharded
            events = run_offline_sharded(cfg, gpus)
        else:
            from darkpipe.pipeline import run_offline
            events = run_offline(cfg)
        dt = time.time() - t0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    if vid_tmp != args.output_video:
        os.replace(vid_tmp, args.output_video)

    events = to_dicts(events)
    # recognize=off 时 run_offline 不写事件文件，但 outputPath 必须存在
    if not os.path.exists(args.events_json):
        with open(args.events_json, "w") as f:
            json.dump(events, f, indent=2)

    counts = {}
    for e in events:
        counts[e["label"]] = counts.get(e["label"], 0) + 1
    # counts 按事件顺序插入，所以并列时 max() 取的是最先出现的那个标签——结果是确定的。
    # 但「并列」本身是使用者需要知道的信息：一段视频里 3 个 drink、3 个 other，报哪个都不是
    # 结论。这里显式标出来，避免下游把 majority_label 当成唯一判定。
    top = max(counts.values()) if counts else 0
    majority_tied = sum(1 for v in counts.values() if v == top) > 1
    # cfg.stats 是管线自己计时的处理段（不含模型加载）；wall_seconds 才是含加载的总耗时。
    # 二者都给出来，吞吐量指标才不会被一次性的启动开销拖低。
    st = cfg.stats
    summary = {
        "input": args.video_path,
        "output_video": args.output_video,
        "frames": st.get("frames", frame_count(args.output_video)),
        "frames_written": st.get("frames_written", 0),
        "seconds": st.get("seconds", round(dt, 3)),
        "fps": st.get("fps", 0.0),
        "wall_seconds": round(dt, 3),
        "gpu_count": st.get("gpus", len(gpus)),
        "event_count": len(events),
        "majority_label": max(counts, key=counts.get) if counts else "",
        "majority_tied": majority_tied,
        "label_counts": counts,
        "config": {"enhance": cfg.enhance, "sr": cfg.sr_name(), "sr_scale": cfg.sr_scale,
                   "recognize": cfg.recognize, "gpu_ids": ",".join(gpus),
                   "reco_span_sec": cfg.reco_span_sec, "enhance_chunk": cfg.enhance_chunk},
    }
    with open(args.summary_json, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[metrics] frames={summary['frames']} seconds={summary['seconds']} "
          f"fps={summary['fps']} wall_seconds={summary['wall_seconds']} "
          f"events={summary['event_count']} "
          f"label={summary['majority_label'] or 'none'}{'(并列)' if majority_tied else ''}")
    print(f"[done] 输出视频 -> {args.output_video}")
    print(f"[done] 识别事件 -> {args.events_json}")
    print(f"[done] 汇总信息 -> {args.summary_json}")

    # 子目录名带上输入视频的名字（`demo_20260821_164652_78`），不然落地目录里一排
    # 时间戳，看不出哪一批产出对应哪个视频。
    deliver({"output_video": args.output_video, "events_json": args.events_json,
             "summary_json": args.summary_json},
            args.local_output_dir, args.hdfs_output_dir,
            {"output_video": "输出视频", "events_json": "识别事件",
             "summary_json": "汇总信息"},
            run=run_dir_name(args.video_path))


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
