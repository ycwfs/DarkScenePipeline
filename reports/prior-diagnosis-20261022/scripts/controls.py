"""Check export/preprocessing agreement and isolate missing-evidence prior behavior."""
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import onnxruntime as ort
import onnx
from ultralytics import YOLO

from probe import OUT, REPO, PKG, detect, sampled_frames
from darkpipe.stages.object_prior import PriorGate, YoloOnnxDetector
from darkpipe.stages.recognize import BehaviorRecognizer


def main() -> None:
    torch.set_num_threads(4)
    torch.cuda.set_device(1)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(PKG / 'ckpts/yolo26m_detect.onnx'), sess_options=options,
        providers=[('CUDAExecutionProvider', {'device_id': 1})])
    assert 'CUDAExecutionProvider' in session.get_providers()
    names = ast.literal_eval(session.get_modelmeta().custom_metadata_map['names'])
    detector = YoloOnnxDetector(str(PKG / 'ckpts'), 'cuda:1')
    detector.sess = session
    pt = YOLO(str(REPO.parent / 'yolo26m-detect.pt'))
    pt.model.end2end = True  # Export explicitly enables one-to-one head; old PT loader defaults differ.
    # CPU cross-check separates GPU numeric settings from export/weight differences.
    cpu_session = ort.InferenceSession(str(PKG / 'ckpts/yolo26m_detect.onnx'),
        sess_options=options, providers=['CPUExecutionProvider'])
    pt_cpu = YOLO(str(REPO.parent / 'yolo26m-detect.pt'))
    pt_cpu.model.end2end = True
    import ultralytics
    bus = cv2.imread(str(Path(ultralytics.__file__).parent / 'assets/bus.jpg'))
    frame = sampled_frames(REPO / 'real-actiondata-all/喝水16.mp4')[5][1]
    frame = cv2.resize(frame, (720, 404), interpolation=cv2.INTER_AREA)
    comparisons = []
    for name, img in [('bus_control', bus), ('喝水16_frame34_raw720', frame)]:
        canvas, scale, left, top = YoloOnnxDetector._letterbox(img)
        x = torch.from_numpy(np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None])).float()/255
        result = pt.predict(x, device=1, imgsz=640, conf=.25, nms=False, verbose=False)[0]
        b = result.boxes.data.cpu().numpy().copy()
        b[:, [0, 2]] = (b[:, [0, 2]] - left)/scale
        b[:, [1, 3]] = (b[:, [1, 3]] - top)/scale
        b[:, :4] = np.clip(b[:, :4], [0]*4, [img.shape[1], img.shape[0]]*2)
        a = detect(session, img)
        a = a[a[:, 4]>.25]
        deployed = detector([img])[0]
        assert np.allclose(a, deployed, atol=1e-5), 'Probe differs from deployment adapter'
        # Compare by class and score order; the positive control has several same-class persons.
        a = a[np.lexsort((-a[:, 4], a[:, 5]))]
        b = b[np.lexsort((-b[:, 4], b[:, 5]))]
        aligned = len(a) == len(b) and np.array_equal(a[:, 5], b[:, 5])
        delta_box = float(np.max(np.abs(a[:, :4]-b[:, :4]))) if aligned else None
        delta_score = float(np.max(np.abs(a[:, 4]-b[:, 4]))) if aligned else None
        print(name, 'box_error_px', delta_box, 'score_error', delta_score,
              'ONNX', a.tolist(), 'PT', b.tolist(), flush=True)
        assert aligned and delta_box < 1.0 and delta_score < .001, 'ONNX/PT mismatch with same inference head'
        cpu_onnx = detect(cpu_session, img)
        cpu_onnx = cpu_onnx[cpu_onnx[:, 4]>.25]
        cpu_pt = pt_cpu.predict(x, device='cpu', imgsz=640, conf=.25, nms=False, verbose=False)[0].boxes.data.numpy()
        def labels_scores(boxes):
            return [dict(label=names[int(r[5])], score=float(r[4])) for r in boxes]
        comparisons.append(dict(name=name, detections=labels_scores(a), pytorch_gpu=labels_scores(b),
            onnx_cpu=labels_scores(cpu_onnx), pytorch_cpu=labels_scores(cpu_pt),
            max_box_error_px=delta_box, max_score_error=delta_score, probe_equals_deploy=True))

    model = onnx.load(str(PKG / 'ckpts/yolo26m_detect.onnx'))
    sd = pt_cpu.predictor.model.model.state_dict()
    weight_diffs = []
    for init in model.graph.initializer:
        if init.name in sd:
            v = onnx.numpy_helper.to_array(init)
            t = sd[init.name].cpu().numpy()
            if v.shape == t.shape:
                weight_diffs.append(dict(name=init.name, max_error=float(np.max(np.abs(v-t)))))

    gate = PriorGate(str(PKG / 'ckpts'), 'cpu')
    sk = np.zeros((32, 6, 17, 3), np.float32)
    valid = np.zeros(6, bool)
    frames = [np.zeros((40, 72, 3), np.uint8)]*32
    info = dict(p=None, valid=valid, slot_p=np.zeros((6, 10)), sk=sk)
    stub = SimpleNamespace(labels=BehaviorRecognizer.labels, lum_gate=75, skel_tau=.65,
                           act_tau=.65, prior=gate, t_parts={'prior': 0.0})
    prob = np.full(10, .1/9, np.float32)
    prob[stub.labels.index('drink')] = .9
    observations = []
    for variant, det in [('unavailable_detector', lambda f: None),
                         ('active_detector_empty_result', lambda f: [np.empty((0, 6)) for _ in f])]:
        gate.det = det
        p, reason = BehaviorRecognizer._decide_max_prior(stub, prob, info, frames, [100.]*32)
        observations.append(dict(synthetic_case=variant, valid_skeletons=0, rgb='drink 0.9',
            result=stub.labels[int(p.argmax())], reason=reason))
    (OUT / 'controls.json').write_text(json.dumps(dict(export_controls=comparisons,
        fused_weight_check=dict(matched=len(weight_diffs), max_error=max(r['max_error'] for r in weight_diffs),
            largest=sorted(weight_diffs, key=lambda r:r['max_error'], reverse=True)[:10]),
        synthetic_missing_evidence=observations), ensure_ascii=False, indent=2))
    print((OUT / 'controls.json').read_text())


if __name__ == '__main__':
    main()
