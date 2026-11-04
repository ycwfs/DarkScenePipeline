"""CPU/GPU-independent focused YOLO26l check on frozen 720-enhanced crops."""
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = OUT.parents[1]
INPUTS = OUT / "inputs.json"
WEIGHTS = ROOT.parent / "yolo26l-detect.pt"
DEST = OUT / "yolo26l-focused"
CONF = 0.35
IOU = 0.5
REVIEW_NAMES = ["bottle", "cup", "phone", "camera", "paper", "portable_object", "robot_or_exhibit"]
REVIEW_TO_COCO = {0: (39,), 1: (41,), 2: (67,)}
DRINK_COCO = {39, 40, 41, 45}


def letterbox(image):
    h, w = image.shape[:2]
    scale = min(640 / w, 640 / h)
    nw, nh = round(w * scale), round(h * scale)
    left, top = (640 - nw) // 2, (640 - nh) // 2
    canvas = np.full((640, 640, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(image, (nw, nh))
    x = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
    return x, scale, left, top


def iou(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    inter = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    area_a = np.maximum(0, a[2:] - a[:2]).prod()
    area_b = np.maximum(0, b[2:] - b[:2]).prod()
    return float(inter / max(area_a + area_b - inter, 1e-6))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def infer(net, image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"cannot read {image_path}")
    x, scale, left, top = letterbox(image)
    with torch.inference_mode():
        result = net(torch.from_numpy(x))
    pred = result[0][0].cpu().numpy().copy()
    pred = pred[pred[:, 4] >= 0.01]
    if len(pred):
        pred[:, [0, 2]] = (pred[:, [0, 2]] - left) / scale
        pred[:, [1, 3]] = (pred[:, [1, 3]] - top) / scale
        pred[:, :4] = np.clip(pred[:, :4], [0, 0, 0, 0], [image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
    return image, pred


def map_holdout(pred, info, record):
    pred = pred[pred[:, 4] >= CONF].copy()
    if len(pred):
        pred[:, :4] *= np.tile(np.asarray(info["scale"], dtype=float), 2)
        pred[:, :4] += np.tile(np.asarray(info["origin"], dtype=float) - np.asarray(record["rect"][:2], dtype=float), 2)
        size = np.asarray(record["rect"][2:], dtype=float) - np.asarray(record["rect"][:2], dtype=float)
        pred[:, :4] = np.clip(pred[:, :4], [0, 0, 0, 0], [size[0], size[1], size[0], size[1]])
    return pred


def truth_in_variant(boxes, info, record):
    origin = np.asarray(info["origin"], dtype=float)
    scale = np.asarray(info["scale"], dtype=float)
    rect_origin = np.asarray(record["rect"][:2], dtype=float)
    mapped = []
    for box in boxes:
        value = dict(box)
        xyxy = np.asarray(box["bbox"], dtype=float)
        xyxy[:2] = (xyxy[:2] + rect_origin - origin) / scale
        xyxy[2:] = (xyxy[2:] + rect_origin - origin) / scale
        value["bbox"] = xyxy.tolist()
        mapped.append(value)
    return mapped


def counts(pred, boxes):
    out = {}
    for cls in range(len(REVIEW_NAMES)):
        gt = [np.asarray(b["bbox"], dtype=float) for b in boxes if b["class_id"] == cls]
        rows = sorted([p for p in pred if p[4] >= CONF and int(p[5]) in REVIEW_TO_COCO.get(cls, ())], key=lambda p: -p[4])
        used, tp = set(), 0
        for p in rows:
            matches = [(iou(p[:4], g), i) for i, g in enumerate(gt) if i not in used]
            if matches and max(matches)[0] >= IOU:
                used.add(max(matches)[1])
                tp += 1
        out[REVIEW_NAMES[cls]] = dict(tp=tp, fp=len(rows) - tp, fn=len(gt) - tp,
                                      gt=len(gt), detector_class_supported=cls in REVIEW_TO_COCO)
    return out


def nms_same_class(rows):
    kept = []
    for row in sorted(rows, key=lambda p: -p[4]):
        if not any(int(row[5]) == int(old[5]) and iou(row[:4], old[:4]) > IOU for old in kept):
            kept.append(row)
    return kept


def draw_tile(image, rows, title, truth_boxes=None, scale=2.0):
    h, w = image.shape[:2]
    factor = min(scale, 700 / max(h, w))
    factor = max(factor, 1.0)
    canvas = cv2.resize(image, (round(w * factor), round(h * factor)), interpolation=cv2.INTER_CUBIC)
    for box in truth_boxes or []:
        x1, y1, x2, y2 = [round(v * factor) for v in box["bbox"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 210, 0), 2)
    for row in rows:
        if row[4] < CONF:
            continue
        x1, y1, x2, y2 = [round(v * factor) for v in row[:4]]
        cls_name = COCO_NAMES[int(row[5])]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 40, 255), 2)
        cv2.putText(canvas, f"{cls_name} {row[4]:.2f}", (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 40, 255), 1, cv2.LINE_AA)
    return canvas, title


def main():
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.backends.mkldnn.enabled = True
    inputs = json.loads(INPUTS.read_text())
    windows = inputs["windows"]
    images = inputs["images"]
    positive = next(w for w in windows if w["id"] == "new-01_1935")

    selected = {}
    holdout_keys = [k for k, r in images.items() if r.get("kind") == "holdout" and r.get("split") in ("val", "test")]
    for key in holdout_keys:
        selected[key] = images[key]["variants"]["small_enhanced"]["path"]
    positive_keys = []
    for sample in positive["samples"]:
        for key in sample["views"]:
            selected[key] = images[key]["variants"]["small_enhanced"]["path"]
            positive_keys.append(key)
    positive_keys = list(dict.fromkeys(positive_keys))

    model = YOLO(str(WEIGHTS))
    COCO_NAMES.update(model.names)
    net = model.model.float().eval().to("cpu")
    net.end2end = True
    net.fuse(verbose=False)

    predictions, timings = {}, []
    for i, (key, relpath) in enumerate(selected.items(), 1):
        path = OUT / relpath
        started = time.perf_counter()
        image, rows = infer(net, path)
        timings.append(time.perf_counter() - started)
        predictions[key] = dict(path=relpath, shape=list(image.shape), detections=rows.tolist())
        if i % 20 == 0 or i == len(selected):
            print(f"[infer] {i}/{len(selected)}", flush=True)

    holdout_rows = []
    for key in holdout_keys:
        record = images[key]
        info = record["variants"]["small_enhanced"]
        raw = np.asarray(predictions[key]["detections"], dtype=float).reshape(-1, 6)
        mapped = map_holdout(raw, info, record)
        summary = counts(mapped, record.get("boxes", []))
        display = [{"class": COCO_NAMES[int(p[5])], "confidence": float(p[4]),
                    "bbox": [round(float(x), 2) for x in p[:4]]} for p in mapped]
        holdout_rows.append(dict(id=key, split=record["split"], source=record["source"],
                                 truth=record.get("boxes", []), counts=summary, detections=display))

    positive_frames = []
    for sample in positive["samples"]:
        full_dets = []
        display_dets = []
        for key in sample["views"]:
            record = images[key]
            info = record["variants"]["small_enhanced"]
            raw = np.asarray(predictions[key]["detections"], dtype=float).reshape(-1, 6)
            raw = raw[raw[:, 4] >= CONF].copy()
            if len(raw):
                raw[:, :4] *= np.tile(np.asarray(info["scale"], dtype=float), 2)
                raw[:, :4] += np.tile(np.asarray(info["origin"], dtype=float), 2)
            full_dets.extend(raw.tolist())
            display_dets.extend(raw.tolist())
        full_dets = nms_same_class(full_dets)
        roi = sample.get("person_roi")
        associated = []
        if sample.get("binding_valid") and roi:
            for p in full_dets:
                cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
                if int(p[5]) in DRINK_COCO and roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]:
                    associated.append(p)
        positive_frames.append(dict(frame=sample["frame"], raw_luma=sample.get("raw_luma"),
            binding_valid=sample.get("binding_valid"), person_roi=roi,
            container_detections=[dict(class_name=COCO_NAMES[int(p[5])], confidence=float(p[4]),
                                       bbox=[round(float(x), 2) for x in p[:4]]) for p in associated],
            all_detections=[dict(class_name=COCO_NAMES[int(p[5])], confidence=float(p[4]),
                                 bbox=[round(float(x), 2) for x in p[:4]]) for p in full_dets]))

    totals = {split: {name: dict(tp=0, fp=0, fn=0, gt=0) for name in REVIEW_NAMES}
              for split in ("val", "test")}
    for row in holdout_rows:
        for name, metric in row["counts"].items():
            for field in ("tp", "fp", "fn", "gt"):
                totals[row["split"]][name][field] += metric[field]
    for split in totals:
        for metric in totals[split].values():
            metric["precision"] = metric["tp"] / max(metric["tp"] + metric["fp"], 1)
            metric["recall"] = metric["tp"] / max(metric["tp"] + metric["fn"], 1)

    DEST.mkdir(parents=True, exist_ok=True)
    payload = dict(model=str(WEIGHTS), model_sha256=digest(WEIGHTS), device="cpu",
        input_variant="small_enhanced (long side 720, Retinexformer, person/ROI crop)",
        confidence=CONF, iou=IOU, luma_gate=30,
        detector_head="Ultralytics one-to-one end-to-end head, 640 letterbox, FP32",
        image_count=len(selected), positive_window=dict(id=positive["id"], timestamp=positive["timestamp"],
            user_confirmed_truth="true drinking", sample_count=len(positive_frames), frames=positive_frames),
        holdout_summary=totals, holdout_rows=holdout_rows,
        median_seconds_per_crop=float(np.median(timings)), p95_seconds_per_crop=float(np.quantile(timings, .95)),
        note="Holdout labels are auxiliary visual annotations; positive action is user-confirmed, but object box itself has no per-frame confirmed label.")
    (DEST / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    with (DEST / "holdout.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "split", "source", "class", "gt", "tp", "fp", "fn", "supported_by_coco", "detections"])
        for row in holdout_rows:
            for name, metric in row["counts"].items():
                writer.writerow([row["id"], row["split"], row["source"], name, metric["gt"], metric["tp"],
                                 metric["fp"], metric["fn"], metric["detector_class_supported"],
                                 "; ".join(f"{d['class']}:{d['confidence']:.2f}" for d in row["detections"])])
    with (DEST / "true-drink-frames.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "raw_luma", "binding_valid", "container_supported", "containers", "all_detections"])
        for row in positive_frames:
            writer.writerow([row["frame"], row["raw_luma"], row["binding_valid"], bool(row["container_detections"]),
                "; ".join(f"{d['class_name']}:{d['confidence']:.3f}" for d in row["container_detections"]),
                "; ".join(f"{d['class_name']}:{d['confidence']:.3f}" for d in row["all_detections"])])

    report = ["# YOLO26l 720增强裁剪复测", "",
        f"- 权重 SHA256：`{payload['model_sha256']}`", "- 设备：CPU（当前会话 NVIDIA 驱动不可用）",
        f"- 输入：{payload['input_variant']}；置信度 `{CONF}`；GT IoU `{IOU}`；原图亮度门 `30`。",
        f"- 测试 `{len(holdout_keys)}` 个带辅助物体框的val/test裁剪，以及真喝水窗口 `{positive['id']}` 的 `{len(positive_frames)}` 个采样帧人物ROI。",
        "", "## 确认的真喝水窗口", "",
        f"用户确认样本为原片01/T317、149.92秒，原始亮度 `{positive['raw_luma']:.2f}`。检测结果只在同人ROI内统计瓶、酒杯、杯、碗（COCO类39/40/41/45）；物体类别框没有独立人工逐帧确认。",
        "", "|帧|绑定有效|同人ROI容器检出|类别/置信度|", "|---:|:---:|:---:|---|"]
    for row in positive_frames:
        found = bool(row["container_detections"])
        names = ", ".join(f"{d['class_name']} {d['confidence']:.2f}" for d in row["container_detections"]) or "无"
        report.append(f"|{row['frame']}|{row['binding_valid']}|{'是' if found else '否'}|{names}|")
    report += ["", f"窗口内5帧至少一帧有同人ROI容器支持：**{'是' if any(r['container_detections'] for r in positive_frames) else '否'}**。", "",
        "## 其他物品标注集", "", "按旧辅助框逐类评估；模型是COCO 80类，没有camera、paper、portable_object、robot_or_exhibit这些目标类别。"]
    for split, metrics in totals.items():
        report += ["", f"### {split}", "", "|类别|TP|FP|FN|GT|检测器支持该类|", "|---|---:|---:|---:|---:|:---:|"]
        for name, m in metrics.items():
            supported = name in ("bottle", "cup", "phone")
            report.append(f"|{name}|{m['tp']}|{m['fp']}|{m['fn']}|{m['gt']}|{'是' if supported else '否'}|")
    report += ["", "逐图记录见 `holdout.csv`；同人喝水窗口见 `true-drink-frames.csv`。这一轮只验证模型是否能检测到物体，不据此更改平台配置或宣称动作先验已上线。", ""]
    (DEST / "README.md").write_text("\n".join(report), encoding="utf-8")

    # Keep a compact visual audit: the positive person's five sampled full-person crops,
    # plus one annotated holdout crop for each available reviewed object class.
    tiles, labels = [], []
    for sample, result in zip(positive["samples"], positive_frames):
        key = sample["views"][0]
        rec = images[key]
        path = OUT / rec["variants"]["small_enhanced"]["path"]
        image = cv2.imread(str(path))
        rows = np.asarray(predictions[key]["detections"], dtype=float).reshape(-1, 6)
        rows = rows[rows[:, 4] >= CONF]
        tile, _ = draw_tile(image, rows, str(sample["frame"]))
        tiles.append(tile)
        labels.append(f"positive drink, frame {sample['frame']}")
    chosen = []
    for cls, name in enumerate(REVIEW_NAMES):
        for row in sorted(holdout_rows, key=lambda r: r["split"] != "test"):
            if any(b["class_id"] == cls for b in row["truth"]):
                chosen.append((cls, row))
                break
    for cls, row in chosen:
        key = row["id"]
        rec = images[key]
        path = OUT / rec["variants"]["small_enhanced"]["path"]
        image = cv2.imread(str(path))
        rows = np.asarray(predictions[key]["detections"], dtype=float).reshape(-1, 6)
        boxes = truth_in_variant(row["truth"], rec["variants"]["small_enhanced"], rec)
        tile, _ = draw_tile(image, rows, row["source"], boxes)
        tiles.append(tile)
        labels.append(f"{REVIEW_NAMES[cls]} GT / {row['source']}")
    cell_w, cell_h, cols = 340, 320, 4
    board = np.full((int(np.ceil(len(tiles) / cols)) * cell_h, cols * cell_w, 3), 245, np.uint8)
    for i, (tile, label) in enumerate(zip(tiles, labels)):
        x, y = (i % cols) * cell_w, (i // cols) * cell_h
        factor = min((cell_w - 16) / tile.shape[1], (cell_h - 45) / tile.shape[0])
        shown = cv2.resize(tile, (max(1, round(tile.shape[1] * factor)), max(1, round(tile.shape[0] * factor))), interpolation=cv2.INTER_AREA)
        board[y:y + shown.shape[0], x:x + shown.shape[1]] = shown
        cv2.putText(board, label[:46], (x + 4, y + cell_h - 10), cv2.FONT_HERSHEY_SIMPLEX, .42, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.imwrite(str(DEST / "visual-review.jpg"), board, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[done] {DEST} crops={len(selected)} median={payload['median_seconds_per_crop']:.3f}s", flush=True)


COCO_NAMES = {}

if __name__ == "__main__":
    main()
