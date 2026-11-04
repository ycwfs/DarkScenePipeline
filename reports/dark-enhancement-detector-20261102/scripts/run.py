"""Compare raw/NTIRE/onsite-finetuned enhancement before the frozen person detector.

The 44 test frames and boxes are the frozen, filtered auxiliary labels from the
2026-10-24 dark-person experiment.  They are deliberately treated as incomplete
labels: an unmatched prediction is reported as a candidate unmatched detection,
not as a verified false positive.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
PIPE = ROOT / "platform/dark-demo-v2"
sys.path.insert(0, str(ROOT / "reports/dark-finetune-20261024/vendor"))
sys.path.insert(0, str(PIPE))
from darkpipe.vendor.retinexformer_arch import RetinexFormer  # noqa: E402
from darkpipe.person.core import person_boxes  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def downscale(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    ratio = min(720 / max(h, w), 1)
    if ratio == 1:
        return image.copy()
    return cv2.resize(image, (round(w * ratio) // 2 * 2, round(h * ratio) // 2 * 2),
                      interpolation=cv2.INTER_AREA)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    area = float(inter[0] * inter[1])
    union = float(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - area)
    return area / max(union, 1e-9)


def greedy_match(gt: np.ndarray, pred: np.ndarray, threshold: float = .5) -> list[tuple[int, int, float]]:
    # Maximise valid match count first, then IoU. Historical name retained for
    # downstream scripts; unlike greedy matching this handles crowded frames.
    if not len(gt) or not len(pred):
        return []
    scores = np.array([[iou(a, b) for b in pred] for a in gt])
    reward = (scores >= threshold) * (min(len(gt), len(pred)) + 1) + scores
    rr, cc = linear_sum_assignment(-reward)
    return [(int(i), int(j), float(scores[i, j])) for i, j in zip(rr, cc)
            if scores[i, j] >= threshold]


class Detector:
    def __init__(self, path: Path):
        self.path = path
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    @staticmethod
    def letterbox(image: np.ndarray):
        h, w = image.shape[:2]
        ratio = min(640 / h, 640 / w)
        nh, nw = round(h * ratio), round(w * ratio)
        top, left = (640 - nh) // 2, (640 - nw) // 2
        canvas = np.full((640, 640, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(
            image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return canvas, ratio, left, top

    def __call__(self, image: np.ndarray, threshold: float = .35) -> np.ndarray:
        canvas, ratio, left, top = self.letterbox(image)
        tensor = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
        rows = self.session.run(None, {self.input_name: np.ascontiguousarray(tensor)})[0][0]
        rows = rows[rows[:, 4] >= threshold].copy()
        if len(rows):
            rows[:, [0, 2]] = (rows[:, [0, 2]] - left) / ratio
            rows[:, [1, 3]] = (rows[:, [1, 3]] - top) / ratio
            h, w = image.shape[:2]
            rows[:, [0, 2]] = np.clip(rows[:, [0, 2]], 0, w)
            rows[:, [1, 3]] = np.clip(rows[:, [1, 3]], 0, h)
        return person_boxes(rows, threshold)


def make_enhancer(path: Path):
    model = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1,
                          num_blocks=[1, 2, 2])
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["params"], strict=True)
    return model.eval().float()


@torch.inference_mode()
def enhance(model, image: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div_(255)
    height, width = tensor.shape[-2:]
    ph, pw = (-height) % 4, (-width) % 4
    if ph or pw:
        tensor = torch.nn.functional.pad(tensor, (0, pw, 0, ph), mode="reflect")
    output = model(tensor)[:, :, :height, :width].clamp(0, 1)
    result = output.mul(255).round().byte()[0].permute(1, 2, 0).numpy()
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def source_frames(samples):
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["source"]].append(sample)
    for source, rows in grouped.items():
        cap = cv2.VideoCapture(str(ROOT / source))
        wanted = {r["frame"] for r in rows}; current = 0; found = {}
        while wanted:
            ok, frame = cap.read()
            if not ok:
                break
            if current in wanted:
                found[current] = frame; wanted.remove(current)
            current += 1
        cap.release()
        if wanted:
            raise RuntimeError(f"missing frames {source}: {sorted(wanted)}")
        for row in rows:
            yield row, found[row["frame"]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "frames").mkdir(exist_ok=True)
    manifest = json.loads((ROOT / "reports/dark-finetune-20261024/manifest.json").read_text())
    samples = [s for s in manifest["samples"] if s["split"] == "test"]
    if len(samples) != 44:
        raise AssertionError(f"expected frozen 44 test frames, got {len(samples)}")
    protocol = {
        "question": "same dark test frames: raw/native, raw720, NTIRE720, onsite-finetuned720",
        "sample_count": len(samples),
        "label_status": "filtered auxiliary boxes, incomplete and not independent exhaustive human GT",
        "iou_threshold": .5,
        "person_confidence": .35,
        "downscale": "same 720 long-side even-dimension rule as deployment",
        "models": {},
        "runtime": {"torch_cuda_available": bool(torch.cuda.is_available()),
                    "ort_providers": ort.get_available_providers(), "device": "CPU"},
    }
    for key, path in {
        "detector_formal": PIPE / "ckpts/yolo26m_person_20261024.onnx",
        "detector_original": PIPE / "ckpts/yolo26m_detect.onnx",
        "enhancer_ntire": PIPE / "ckpts/NTIRE.pth",
        "enhancer_onsite_ft": ROOT / "ckpts_realdeploy_ft/NTIRE.pth",
    }.items():
        protocol["models"][key] = {"path": str(path.relative_to(ROOT)),
                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    save(OUT / "protocol.json", protocol)

    detectors = {"formal": Detector(PIPE / "ckpts/yolo26m_person_20261024.onnx"),
                 "original": Detector(PIPE / "ckpts/yolo26m_detect.onnx")}
    ft = make_enhancer(ROOT / "ckpts_realdeploy_ft/NTIRE.pth")
    records = []
    started = time.time()
    for n, (sample, native) in enumerate(source_frames(samples), 1):
        raw720 = downscale(native)
        ntire = cv2.imread(str(ROOT / "reports/dark-finetune-20261024/dataset/images/test/"
                               f"{sample['video_id']}_{sample['frame']:06d}_enhanced.png"))
        if ntire is None:
            raise RuntimeError(f"missing cached NTIRE frame for {sample['id']}")
        ft720 = enhance(ft, raw720)
        cv2.imwrite(str(OUT / "frames" / f"{sample['id']:03d}_raw720.png"), raw720)
        cv2.imwrite(str(OUT / "frames" / f"{sample['id']:03d}_ntire720.png"), ntire)
        cv2.imwrite(str(OUT / "frames" / f"{sample['id']:03d}_ft720.png"), ft720)
        images = {"raw_native": native, "raw720": raw720, "ntire720": ntire, "ft720": ft720}
        truth_native = np.asarray([p["bbox_xyxy"] for p in sample["people"]], np.float32)
        row = {"id": sample["id"], "source": sample["source"], "frame": sample["frame"],
               "luma": sample["luma"], "native_shape": list(native.shape[:2]),
               "gt_boxes_native": truth_native.tolist(), "conditions": {}}
        for image_name, image in images.items():
            scale = np.array([image.shape[1] / native.shape[1], image.shape[0] / native.shape[0]], np.float32)
            truth = truth_native * np.tile(scale, 2)
            row["conditions"][image_name] = {"shape": list(image.shape[:2]),
                                               "gt_heights": (truth[:, 3] - truth[:, 1]).tolist(),
                                               "gt_widths": (truth[:, 2] - truth[:, 0]).tolist()}
            for detector_name, detector in detectors.items():
                pred = detector(image)
                matched = greedy_match(truth, pred[:, :4])
                row["conditions"][image_name][detector_name] = {
                    "predictions": int(len(pred)),
                    "matched": int(len(matched)),
                    "unmatched_predictions": int(len(pred) - len(matched)),
                    "unmatched_gt": int(len(truth) - len(matched)),
                    "ious": [float(x[2]) for x in matched],
                    "matched_gt_indices": [int(x[0]) for x in matched],
                    "matched_pred_indices": [int(x[1]) for x in matched],
                    "pred_boxes": pred[:, :4].tolist(),
                    "pred_scores": pred[:, 4].tolist(),
                }
        records.append(row)
        print(f"[{n}/{len(samples)}] {sample['source']} F{sample['frame']}", flush=True)
    save(OUT / "records.json", records)
    save(OUT / "run-summary.json", {"seconds": time.time() - started,
                                    "records": len(records), "conditions": list(records[0]["conditions"])})


if __name__ == "__main__":
    main()
