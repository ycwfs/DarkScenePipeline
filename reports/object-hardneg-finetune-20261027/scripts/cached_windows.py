"""Secondary old heldout regression: same cached windows/crops, detector-only swap.

These 55 historical candidates were selected with the older action configuration.
Do not interpret this diagnostic as an all-data run with current action thresholds.
"""
from collections import Counter
import json
import sys

import cv2
import numpy as np
from build_dataset import OUT,ROOT,CACHE,OLD,save
from evaluate_objects import session,predict

sys.path.insert(0,str(ROOT/'platform/dark-demo-v2'))
from darkpipe.person.object_prior import filter_object_evidence
from darkpipe.person.core import iou


def main():
    cv2.setNumThreads(1)
    data=json.loads((CACHE/'inputs.json').read_text())
    windows=[w for w in data['windows']if not w['source_id'].startswith('new-')]
    keys=sorted({key for w in windows for s in w['samples']for key in s['views']})
    sessions=dict(baseline=session(OLD/'models/yolo26m_objects_candidate.onnx'),
                  candidate=session(OUT/'models/yolo26m_objects_candidate.onnx'))
    predictions={}
    for key in keys:
        v=data['images'][key]['variants']['small_enhanced'];im=cv2.imread(str(CACHE/v['path']));assert im is not None
        predictions[key]={}
        for name,s in sessions.items():
            pred=predict(s,im);pred=pred[pred[:,4]>=.581]
            pred[:,:4]=pred[:,:4]*np.array(v['scale']*2)+np.array(v['origin']*2)
            predictions[key][name]=pred.tolist()
    result=[]
    for w in windows:
        variants={}
        for name in sessions:
            samples=[]
            for sample in w['samples']:
                row={k:v for k,v in sample.items()if k!='views'};boxes=[]
                for key in sample['views']:boxes.extend(predictions[key][name])
                detections=[]
                for b in sorted(boxes,key=lambda b:-b[4]):
                    if not any(int(b[5])==int(p[5])and iou(np.array(b[:4]),np.array(p[:4]))>.5 for p in detections):detections.append(b)
                row.update(detections=detections,available=bool(sample['views']));samples.append(row)
            variants[name]=filter_object_evidence(w['baseline'],samples,w['raw_luma'],.581,35.)
        result.append(dict(w,variants=variants))
    summary={}
    for split in sorted({r['split']for r in result}):
        subset=[r for r in result if r['split']==split]
        summary[split]=dict(windows=len(subset),sources=len({r['source_id']for r in subset}),models={})
        for name in sessions:
            summary[split]['models'][name]=dict(
                true_actions_retained=sum(r['truth']in ('drink','pick_up')and r['variants'][name]['label']==r['truth']for r in subset),
                true_actions=sum(r['truth']in ('drink','pick_up')for r in subset),
                other_false_triggers=sum(r['truth']=='other'and r['variants'][name]['label']!='other'for r in subset),
                other_windows=sum(r['truth']=='other'for r in subset),
                labels=dict(Counter(r['variants'][name]['label']for r in subset)))
    save(OUT/'cached-old-window-evaluation.json',dict(records=result,summary=summary,
        detector_threshold=.581,lum_gate=35.,
        note='Secondary diagnostic on 55 historical selected action windows, frozen old upstream predictions/bindings, '
             'not a current complete pipeline or current action .925/RGB .972 rerun. Only detector changed.'))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
