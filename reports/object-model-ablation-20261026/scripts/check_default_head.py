"""Supplement: public default NMS head on native crops, after the frozen primary run."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).parent))
from compare import OUT, BASE, MODELS, map_rows, save, counts, filter_object_evidence, digest
from ultralytics import YOLO


def main(name):
    torch.set_num_threads(2);cv2.setNumThreads(1)
    manifest=json.loads((OUT/'inputs.json').read_text())
    plans=json.loads((BASE/'demo-plan.json').read_text())
    case_ids={f"{r['source_id']}_{r['event_index']}" for r in plans}
    windows=[w for w in manifest['windows'] if w['split']!='feedback' or w['id'] in case_ids]
    keys={key for w in windows for s in w['samples'] for key in s['views']}
    keys.update(k for k,r in manifest['images'].items() if r['kind']=='holdout')
    keys=sorted(keys);predictions={};model=YOLO(str(MODELS[name]));objects=[]
    for start in range(0,len(keys),8):
        batch=keys[start:start+8]
        images=[cv2.imread(str(OUT/manifest['images'][k]['variants']['native_raw']['path']))for k in batch]
        outputs=model.predict(images,imgsz=640,conf=.1,rect=False,device=0,quantize=32,
                              nms=None,iou=.7,verbose=False)
        for key,out in zip(batch,outputs):
            rows=out.boxes.data.cpu().numpy();predictions[key]=rows.tolist()
            meta=manifest['images'][key]
            if meta['kind']=='holdout':
                objects.append(dict(id=key,split=meta['split'],counts=counts(map_rows(rows.tolist(),name),meta['boxes'])))
    decisions=[]
    for w in windows:
        samples=[]
        for s in w['samples']:
            dets=[]
            for key in s['views']:
                rows=map_rows(predictions[key],name)
                rows[:,:4]+=np.tile(manifest['images'][key]['variants']['native_raw']['origin'],2)
                dets+=rows.tolist()
            samples.append(dict(s,detections=dets,available=bool(s['views'])))
        decisions.append({k:v for k,v in w.items() if k!='samples'}|
                         dict(decision=filter_object_evidence(w['baseline'],samples,w['raw_luma'])))
    save(OUT/f'default-head-{name}.json',dict(model=name,model_sha256=digest(MODELS[name]),
        images=len(keys),native_raw_only=True,head='official predict default nms=None (one-to-many + external NMS IoU=.7)',
        note='supplementary diagnostic, not used to select a model; all old windows plus predefined cases only',
        objects=objects,actions=decisions,predictions=predictions))
    print('Default-head check completed',name,len(keys),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['n','s','m'],required=True)
    main(parser.parse_args().model)
