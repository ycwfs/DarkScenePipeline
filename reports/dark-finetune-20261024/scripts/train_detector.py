"""Small-LR person adaptation of the existing YOLO26m; original split frozen."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

from prepare import OUT, ROOT, save


def labels() -> Path:
    samples=json.loads((OUT/'manifest.json').read_text())['samples']
    for s in samples:
        views=['enhanced','raw'] if s['split']=='train' else ['enhanced']
        for view in views:
            name=f'{s["video_id"]}_{s["frame"]:06d}_{view}'
            image=OUT/'dataset/images'/s['split']/(name+'.png')
            if not image.exists():raise FileNotFoundError(image)
            target=OUT/'dataset/labels'/s['split']/(name+'.txt')
            target.parent.mkdir(parents=True,exist_ok=True)
            rows=[]
            for p in s['people']:
                b=np.array(p['bbox_xyxy'])/np.array([s['width'],s['height']]*2)
                rows.append('0 '+' '.join(f'{x:.8f}' for x in [*(b[:2]+b[2:])/2,*(b[2:]-b[:2])]))
            target.write_text('\n'.join(rows)+'\n')
    model=YOLO(str(ROOT.parent/'yolo26m-detect.pt'))
    data=dict(path=str(OUT/'dataset'),train='images/train',val='images/val',test='images/test',
              names=model.names,nc=len(model.names))
    path=OUT/'detector.yaml';path.write_text(yaml.safe_dump(data,allow_unicode=True))
    return path


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=12)
    parser.add_argument('--labels-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(1)
    path=labels()
    if not args.labels_only:
        model=YOLO(str(ROOT.parent/'yolo26m-detect.pt'))
        result=model.train(data=str(path),epochs=args.epochs,imgsz=640,batch=12,device=0,
            workers=2,project=str(OUT/'runs'),name='detector',exist_ok=False,seed=24,
            deterministic=True,optimizer='AdamW',lr0=.0001,lrf=.2,weight_decay=.0005,
            warmup_epochs=1,freeze=10,patience=12,amp=True,cache=False,
            mosaic=0.,mixup=0.,copy_paste=0.,degrees=0.,translate=.03,scale=.1,
            fliplr=.5,hsv_h=.005,hsv_s=.1,hsv_v=.1,close_mosaic=0,
            plots=False,save_period=1,val=True,pretrained=True,nms=False)
        save(OUT/'detector-training-result.json',dict(metrics=result.results_dict,
            best=str(model.trainer.best),last=str(model.trainer.last)))
