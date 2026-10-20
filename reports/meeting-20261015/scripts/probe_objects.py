"""Reproduce the CPU object-detection diagnostics on the preserved JPEG examples."""
from pathlib import Path
import json
import sys

import cv2
import numpy as np
import onnxruntime as ort

OUT = Path(__file__).resolve().parents[1]
REPO = OUT.parents[1]
PKG = REPO / 'platform/dark-demo-v2'
sys.path.insert(0, str(PKG))
from darkpipe.stages.object_prior import CONTAINER_CLS, YoloOnnxDetector

options = ort.SessionOptions()
options.intra_op_num_threads = 4
options.inter_op_num_threads = 1
detector = YoloOnnxDetector(str(PKG / 'ckpts'), 'cpu')
# This is a CPU functionality diagnostic, not a production latency benchmark.
detector.sess = ort.InferenceSession(str(PKG / 'ckpts/yolo26m_detect.onnx'),
    sess_options=options, providers=['CPUExecutionProvider'])
records = []
for stem in ['喝水16', '捡东西2']:
    for kind in ['raw', 'enhanced']:
        path = OUT / 'sources' / f'{stem}-{kind}.jpg'
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f'Cannot read {path}')
        boxes = detector([frame])[0]
        records.append(dict(name=stem, kind=kind, path=str(path.relative_to(OUT)),
            shape=list(frame.shape), detections=boxes.tolist(),
            containers=int(np.isin(boxes[:, 5], CONTAINER_CLS).sum()),
            non_person=int((boxes[:, 5] != 0).sum())))
report = dict(input_size=640, conf_threshold=.25, provider='CPUExecutionProvider',
    model='yolo26m_detect.onnx',
    scope='One saved JPEG per case, raw and enhanced diagnostic; not a sequence/accuracy benchmark',
    records=records)
(OUT / 'evidence/object-detections.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False, indent=2))
