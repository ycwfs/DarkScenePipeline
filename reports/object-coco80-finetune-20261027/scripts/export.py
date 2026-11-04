"""Select by old validation one-to-one mAP, lock, export and compare numeric outputs."""
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import onnxruntime as ort
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from common import OUT, BASE, save


def letterbox(image: np.ndarray) -> tuple:
    h, w = image.shape[:2]
    scale = min(640 / w, 640 / h)
    nw, nh = round(w * scale), round(h * scale)
    left, top = (640 - nw) // 2, (640 - nh) // 2
    canvas = np.full((640, 640, 3), 114, np.uint8)
    canvas[top:top+nh, left:left+nw] = cv2.resize(image, (nw, nh))
    x = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
    return x, scale, left, top


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
    return float(inter / max(np.prod(a[2:]-a[:2]) + np.prod(b[2:]-b[:2]) - inter, 1e-6))


def main() -> None:
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    selected = json.loads((OUT/'selection.json').read_text())['selected']
    target = OUT/'models/yolo26m_coco80_candidate.pt'
    target.parent.mkdir(exist_ok=True)
    shutil.copy2(selected['path'], target)
    shutil.copy2(BASE, OUT/'models/yolo26m_coco80_original.pt')
    original = YOLO(str(OUT/'models/yolo26m_coco80_original.pt'))
    candidate_for_check = YOLO(str(target))
    if selected['family'] == 'retention':
        targets = {0,39,41,65,67,73}
        protected = [c for c in range(80) if c not in targets]
        reference = original.model.float().state_dict()
        actual = candidate_for_check.model.float().state_dict()
        allowed = json.loads((OUT/'retention-initialization.json').read_text())['classifier_tensors']
        differences = []
        for name in reference:
            a,b = reference[name],actual[name]
            if name in allowed:a,b = a[protected],b[protected]
            if not torch.equal(a,b):differences.append(name)
        assert not differences, differences
        save(OUT/'preservation-check.json',dict(passed=True,unchanged_shared_features_and_localization=True,
            unchanged_other_class_rows=74,updated_existing_classes=sorted(targets),nc=80))
    original.export(format='onnx', imgsz=640, opset=17, simplify=False, dynamic=False, batch=1, device='cpu', half=False, nms=False)
    model = YOLO(str(target))
    assert len(model.names) == 80 and list(model.names.values()) == json.loads((OUT/'classes.json').read_text())
    model.model.end2end = True
    exported = model.export(format='onnx', imgsz=640, opset=17, simplify=False, dynamic=False,
        batch=1, device='cpu', half=False, nms=False)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    runner = ort.InferenceSession(exported, sess_options=opts, providers=['CPUExecutionProvider'])
    model.model.float().eval()
    images = [Path(p) for p in (OUT/'coco_val.txt').read_text().splitlines()[:5]] + [Path(p) for p in (OUT/'site_val.txt').read_text().splitlines()[:5]]
    checks = []
    for i in np.linspace(0, len(images)-1, min(10, len(images))).round().astype(int):
        x, *_ = letterbox(cv2.imread(str(images[i])))
        with torch.no_grad():
            expected = model.model(torch.from_numpy(x))[0].numpy()[0]
        actual = runner.run(None, {runner.get_inputs()[0].name:x})[0][0]
        assert actual.shape == (300, 6)
        a = actual[actual[:, 4] >= .25]
        b = expected[expected[:, 4] >= .25]
        assert len(a) == len(b)
        error = 0.
        if len(a):
            cost = np.array([[1-iou(aa[:4], bb[:4]) + (aa[5]!=bb[5])*10 for bb in b] for aa in a])
            ii, jj = linear_sum_assignment(cost)
            error = float(np.max(np.abs(a[ii]-b[jj])))
            assert error < .15, (images[i], error)
        checks.append(dict(image=images[i].name, predictions=len(a), max_abs_error=error))
    save(OUT/'export-verification.json', dict(passed=True, checks=checks, output_shape=[1,300,6],
        pt_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        onnx_sha256=hashlib.sha256(OUT.joinpath('models/yolo26m_coco80_candidate.onnx').read_bytes()).hexdigest()))


if __name__ == '__main__':
    main()
