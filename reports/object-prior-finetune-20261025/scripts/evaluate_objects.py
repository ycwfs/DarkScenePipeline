"""Paired full-image/native-ROI comparison within explicitly reviewed holdout regions."""
import json
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort
from build_dataset import OUT, ROOT, save
from select_export import letterbox, iou


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    s = ort.InferenceSession(str(path), sess_options=options,
        providers=[('CUDAExecutionProvider', {'device_id':0})])
    assert s.get_providers()[0] == 'CUDAExecutionProvider'
    return s


def detect(s, frame: np.ndarray) -> np.ndarray:
    x, scale, left, top = letterbox(frame)
    pred = s.run(None, {s.get_inputs()[0].name:x})[0][0].copy()
    pred[:, [0,2]] = (pred[:, [0,2]]-left)/scale
    pred[:, [1,3]] = (pred[:, [1,3]]-top)/scale
    pred[:, :4] = np.clip(pred[:, :4], [0]*4, [frame.shape[1],frame.shape[0]]*2)
    return pred[pred[:,4] >= .01]


def counts(pred: np.ndarray, boxes: list, threshold=.35) -> dict:
    results = {}
    for cls in range(7):
        gt = [b['bbox'] for b in boxes if b['class_id']==cls]
        rows = sorted([p for p in pred if int(p[5])==cls and p[4]>=threshold],key=lambda p:-p[4])
        used, tp = set(), 0
        for p in rows:
            options = [(iou(p[:4],np.array(g)),i) for i,g in enumerate(gt) if i not in used]
            if options and max(options)[0]>=.5:
                used.add(max(options)[1]);tp+=1
        results[str(cls)] = dict(tp=tp,fp=len(rows)-tp,fn=len(gt)-tp)
    return results


def main() -> None:
    cv2.setNumThreads(1)
    data = json.loads((OUT/'dataset-manifest.json').read_text())['records']
    base = session(ROOT/'platform/dark-demo-v2/ckpts/yolo26m_detect.onnx')
    fine = session(OUT/'models/yolo26m_objects_candidate.onnx')
    rows, elapsed = [], []
    for r in data:
        if r['split'] not in ('val','test'):
            continue
        roi = cv2.imread(r['dataset_image'])
        native = cv2.imread(str(__import__('pathlib').Path(r['image_path']).parent/r['raw_file']))
        if r.get('brightness_gamma',1.)!=1.:
            lut = np.round((np.arange(256)/255.)**r['brightness_gamma']*255).astype(np.uint8)
            native = cv2.LUT(native, lut)
        rect = np.array(r['reviewed_crop_xyxy_raw'])
        variants = {}
        for model, s in [('pretrained',base),('finetuned',fine)]:
            for scope, image in [('full',native),('roi',roi)]:
                start=time.perf_counter();pred=detect(s,image);duration=time.perf_counter()-start
                elapsed.append(dict(model=model,scope=scope,seconds=duration))
                if model=='pretrained':
                    mapping={39:0,41:1,67:2}
                    pred=pred[np.isin(pred[:,5],[39,41,67])]
                    for p in pred:p[5]=mapping[int(p[5])]
                if scope=='full':
                    center=(pred[:,:2]+pred[:,2:4])/2
                    pred=pred[((center>=rect[:2])&(center<=rect[2:])).all(1)]
                    pred[:,:4]-=np.tile(rect[:2],2)
                    pred[:,:4]=np.clip(pred[:,:4],[0]*4,[r['width'],r['height']]*2)
                variants[f'{model}_{scope}']=dict(detections=pred.tolist(),counts=counts(pred,r['boxes']))
        rows.append(dict(id=r['id'],split=r['split'],group=r['group'],source=r['source'],
            image=r['dataset_image'],boxes=r['boxes'],variants=variants))
    summary={}
    for split in ('val','test'):
        summary[split]={}
        for variant in ('pretrained_full','pretrained_roi','finetuned_full','finetuned_roi'):
            totals={str(c):{k:sum(r['variants'][variant]['counts'][str(c)][k] for r in rows if r['split']==split)
                            for k in ('tp','fp','fn')} for c in range(7)}
            for total in totals.values():
                total['precision']=total['tp']/max(total['tp']+total['fp'],1)
                total['recall']=total['tp']/max(total['tp']+total['fn'],1)
            summary[split][variant]=totals
    save(OUT/'object-evaluation.json',dict(records=rows,summary=summary,latency=elapsed,
        confidence=.35,iou=.5,pretrained_supported_classes=[0,1,2],
        gamma_preprocessing='rounded uint8 LUT, same as annotated ROI and action replay',
        note='Full predictions scored only in reviewed crop; no GT assumed elsewhere. Single scene auxiliary GT.'))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
