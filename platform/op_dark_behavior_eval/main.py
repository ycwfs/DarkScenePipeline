"""暗光场景行为识别测试验证算子——平台入口。

对一份带标注的视频清单逐条推理，输出准确率、宏平均 F1、各类别召回率与混淆矩阵。
按规范要求，测试类算子的日志至少包含处理的数据集行数与处理时间，并输出准确率/F1/召回率。

推理协议与在线管线保持一致：每个片段均匀抽取 T 帧 -> 经同一个增强算子 ->
darkpipe.stages.recognize.preprocess_frame -> 识别器 _infer。这两个函数就是管线内部使用的
同一份实现（其文档亦明确供离线评估脚本直接调用），因此评测域不会与部署域产生漂移。
超分不参与识别：管线中识别器取的是增强后、超分前的帧。
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

from oputil import (deliver, ensure_parent, fetch_input,  # noqa: E402
                    parse_gpu_ids, require_destination)


def build_parser():
    p = argparse.ArgumentParser(
        prog="main.py", description="暗光场景行为识别测试验证算子：在带标注数据集上评测识别精度")
    p.add_argument("--dataset_manifest", required=True,
                   help="带标注清单文件，每行 `视频路径,类别标签`，逗号或制表符分隔，可含表头")
    p.add_argument("--enhance", default="retinexformer", choices=["off", "retinexformer"])
    p.add_argument("--recognize", default="behavior", choices=["behavior"])
    p.add_argument("--max_clips", type=int, default=0)
    p.add_argument("--gpu_ids", default="0")
    # 与处理算子同一含义：增强阶段一次送入 GPU 的帧数。评测每条片段抽 32 帧，若整批送入，
    # 显存占用会随片段分辨率线性膨胀；分块后显存上限与分辨率无关地受控。
    p.add_argument("--proc_max_side", type=int, default=0,
                   help="处理前把画面长边缩到不超过该值，0=按原分辨率。增强耗时与像素量成正比，而识别内部固定缩到 224，1080p 填 1280 可省四分之三算力且不损识别精度")
    p.add_argument("--enhance_chunk", type=int, default=4)
    p.add_argument("--ckpt_dir", default="/opt/darkpipe/ckpts")
    p.add_argument("--local_output_dir", default="",
                   help="产出另存到这个容器内目录（挂载 NFS 后即可本地浏览），如 /mnt/nfs/darkout；"
                        "留空则不另存本地")
    p.add_argument("--hdfs_output_dir", default="",
                   help="产出上传到这个 HDFS 目录，如 hdfs://用户名@ip:port/a/b；留空则不上传。"
                        "与 local_output_dir 至少要填一个，否则结果随容器一起消失")
    p.add_argument("--metrics_json", required=True)
    p.add_argument("--confusion_csv", required=True)
    return p


def read_manifest(path, limit=0):
    """-> [(视频绝对路径, 标签)]。相对路径按清单文件所在目录解析。"""
    base = os.path.dirname(os.path.abspath(path))
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [c.strip() for c in (line.split("\t") if "\t" in line else line.split(","))]
            if len(parts) < 2:
                sys.exit(f"error: 清单第 {ln + 1} 行格式错误，需要 `视频路径,类别标签`: {line!r}")
            vid, label = parts[0], parts[1]
            if ln == 0 and label.lower() in ("label", "class", "标签", "类别"):
                continue                                    # 表头
            if not os.path.isabs(vid):
                vid = os.path.join(base, vid)
            rows.append((vid, label))
            if limit and len(rows) >= limit:
                break
    if not rows:
        sys.exit(f"error: 清单 {path} 中没有可用样本")
    return rows


def sample_frames(path, t):
    """均匀抽取 t 帧 BGR，不足则重复末帧；读不到返回 []。"""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        return []
    idx = np.linspace(0, len(frames) - 1, t).round().astype(int)
    return [frames[i] for i in idx]


def enhance_chunked(frames, stages, chunk):
    """按 chunk 帧分批过增强阶段——与在线管线中 VideoReader 的分块方式一致。"""
    out = []
    for i in range(0, len(frames), chunk):
        part = frames[i:i + chunk]
        for s in stages:
            part = s(part)
        out.extend(part)
    return out


def metrics_from_confusion(labels, conf):
    """conf[i][j] = 真实 i 被预测为 j。-> (准确率, 宏平均F1, 每类 precision/recall/f1)。"""
    total = sum(sum(r) for r in conf)
    correct = sum(conf[i][i] for i in range(len(labels)))
    per = {}
    f1s = []
    for i, lab in enumerate(labels):
        tp = conf[i][i]
        support = sum(conf[i])
        pred = sum(conf[r][i] for r in range(len(labels)))
        recall = tp / support if support else 0.0
        precision = tp / pred if pred else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per[lab] = {"support": support, "precision": round(precision, 4),
                    "recall": round(recall, 4), "f1": round(f1, 4)}
        if support:                       # 数据集中不存在的类别不拉低宏平均
            f1s.append(f1)
    accuracy = correct / total if total else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return accuracy, macro_f1, per


def run(args):
    import numpy as np
    from darkpipe.config import PipelineConfig, validate
    from darkpipe.stages import build_stages
    from darkpipe.stages.recognize import preprocess_frame

    require_destination(args.local_output_dir, args.hdfs_output_dir)
    gpus = parse_gpu_ids(args.gpu_ids)
    device = f"cuda:{gpus[0]}"
    chunk = max(1, args.enhance_chunk)
    for p in (args.metrics_json, args.confusion_csv):
        ensure_parent(p)

    manifest = fetch_input(args.dataset_manifest, "/tmp/darkeval", "manifest.csv")
    rows = read_manifest(manifest, args.max_clips)
    print(f"[data] 数据集清单 {args.dataset_manifest}，待评测样本 {len(rows)} 条")

    # 超分与评测无关（识别取增强后、超分前的帧），关闭以免拖慢评测
    cfg = validate(PipelineConfig(input=rows[0][0], enhance=args.enhance, sr="off",
                                  recognize=args.recognize, device=device,
                                  ckpt_dir=args.ckpt_dir, proc_max_side=args.proc_max_side, output="/tmp/darkeval/unused.mp4"))
    stages, recognizer = build_stages(cfg)
    if recognizer is None:
        sys.exit(f"error: recognize={args.recognize} 没有识别器，无法评测")
    for s in stages:
        print(f"[load] {s.name}")
        s.load(device)
    print(f"[load] recognize:{recognizer.name} (labels={len(recognizer.labels)})")
    recognizer.load(device)

    labels = list(recognizer.labels)
    index = {l: i for i, l in enumerate(labels)}
    conf = [[0] * len(labels) for _ in labels]
    skipped, unknown = [], set()
    t0 = time.time()

    for n, (vid, truth) in enumerate(rows, 1):
        if truth not in index:
            unknown.add(truth)
            skipped.append((vid, f"标签 {truth!r} 不在模型类别表中"))
            continue
        frames = sample_frames(vid, recognizer.window)
        if not frames:
            skipped.append((vid, "无法解码"))
            continue
        frames = enhance_chunked(frames, stages, chunk)
        arr = np.stack([preprocess_frame(f, recognizer.cfg) for f in frames])
        prob = recognizer._infer(arr)
        conf[index[truth]][int(np.argmax(prob))] += 1
        if n % 50 == 0 or n == len(rows):
            done = sum(sum(r) for r in conf)
            acc = sum(conf[i][i] for i in range(len(labels))) / max(done, 1)
            print(f"[eval] 已处理 {n}/{len(rows)} 行，累计准确率 {acc:.4f}，"
                  f"耗时 {time.time() - t0:.1f}s")

    dt = time.time() - t0
    evaluated = sum(sum(r) for r in conf)
    accuracy, macro_f1, per = metrics_from_confusion(labels, conf)

    for s in stages:
        s.close()
    recognizer.close()

    with open(args.confusion_csv, "w", encoding="utf-8") as f:
        f.write("true\\pred," + ",".join(labels) + "\n")
        for i, lab in enumerate(labels):
            f.write(lab + "," + ",".join(str(c) for c in conf[i]) + "\n")

    result = {
        "dataset_manifest": args.dataset_manifest,
        "rows_in_manifest": len(rows),
        "rows_evaluated": evaluated,
        "rows_skipped": len(skipped),
        "elapsed_seconds": round(dt, 3),
        "seconds_per_clip": round(dt / evaluated, 4) if evaluated else 0.0,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per,
        "labels": labels,
        "confusion_matrix": conf,
        "config": {"enhance": args.enhance, "recognize": args.recognize,
                   "gpu_ids": ",".join(gpus), "enhance_chunk": chunk},
        "skipped": [{"video": v, "reason": r} for v, r in skipped[:50]],
    }
    with open(args.metrics_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 规范要求的测试类算子日志：数据集行数、处理时间、准确率/F1/召回率
    print(f"[metrics] 数据集行数={len(rows)} 实际评测={evaluated} 跳过={len(skipped)}")
    print(f"[metrics] 处理时间={dt:.1f}s 平均每条={dt / max(evaluated, 1):.3f}s")
    print(f"[metrics] 准确率(accuracy)={accuracy:.4f} 宏平均F1(macro-F1)={macro_f1:.4f}")
    for lab in labels:
        p = per[lab]
        print(f"[metrics] 类别 {lab}: 样本数={p['support']} 召回率={p['recall']:.4f} "
              f"精确率={p['precision']:.4f} F1={p['f1']:.4f}")
    if unknown:
        print(f"[warn] 清单中有 {len(unknown)} 个标签不在模型类别表内，已跳过: "
              f"{sorted(unknown)[:10]}")
    print(f"[done] 指标 -> {args.metrics_json}")
    print(f"[done] 混淆矩阵 -> {args.confusion_csv}")

    deliver({"metrics_json": args.metrics_json, "confusion_csv": args.confusion_csv},
            args.local_output_dir, args.hdfs_output_dir,
            {"metrics_json": "指标", "confusion_csv": "混淆矩阵"})


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
