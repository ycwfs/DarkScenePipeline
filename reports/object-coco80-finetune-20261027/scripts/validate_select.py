"""Lock candidate using site validation subject to original COCO retention guard."""
import argparse
import json
import os
from pathlib import Path
import time
os.environ['YOLO_AUTOINSTALL'] = 'false'

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from common import BASE, OUT, save, sha


def evaluate(path, domain, split='val'):
    model=YOLO(str(path))
    result=model.val(data=str(OUT/('objects.yaml' if domain=='coco' else 'site.yaml')),
        split=split,imgsz=640,batch=24,device=0,workers=4,nms=False,plots=False,
        save_json=False,project=str(OUT/'evaluation-runs'),name=f'{path.stem}-{domain}-{split}',exist_ok=True)
    classes=result.box.ap_class_index.tolist()
    aps=result.box.all_ap.tolist()
    return dict(metrics=result.results_dict,ap_by_class={str(c):float(np.mean(ap))for c,ap in zip(classes,aps)},
                ap50_by_class={str(c):float(ap[0])for c,ap in zip(classes,aps)},classes=classes)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');parser.add_argument('--test',action='store_true')
    parser.add_argument('--run',default='coco80',choices=['coco80','retention'])
    args=parser.parse_args();torch.set_num_threads(2);cv2.setNumThreads(1)
    baseline_path=OUT/'baseline-validation.json'
    if not baseline_path.exists():
        save(baseline_path,dict(path=str(BASE),sha256=sha(BASE),coco=evaluate(BASE,'coco'),site=evaluate(BASE,'site')))
    baseline=json.loads(baseline_path.read_text())
    if args.test:
        selection=json.loads((OUT/'selection.json').read_text())
        for name,path in [('original80',BASE),('candidate80',Path(selection['selected']['path']))]:
            dest=OUT/f'{name}-coco-test.json'
            if not dest.exists():save(dest,dict(path=str(path),sha256=sha(path),coco=evaluate(path,'coco','test')))
        return
    prefix='' if args.run=='coco80' else 'retention-'
    progress=OUT/f'{prefix}selection-progress.json'
    rows=json.loads(progress.read_text())if progress.exists()else[]
    while len(rows)<20:
        seen={r['path']for r in rows}
        paths=sorted((OUT/f'runs/{args.run}/weights').glob('epoch*.pt'),key=lambda p:int(p.stem[5:]))
        for path in paths:
            if str(path) in seen or time.time()-path.stat().st_mtime<5:
                continue
            coco=evaluate(path,'coco');site=evaluate(path,'site')
            eligible=coco['metrics']['metrics/mAP50-95(B)'] >= baseline['coco']['metrics']['metrics/mAP50-95(B)']-.01
            rows.append(dict(path=str(path),sha256=sha(path),epoch=int(path.stem[5:])+1,coco=coco,site=site,
                eligible=eligible,score=site['metrics']['metrics/mAP50-95(B)']))
            save(progress,rows);print('SELECTION',path.stem,'eligible',eligible,'site',rows[-1]['score'],flush=True)
        if len(rows)==20:break
        if not args.watch:return
        time.sleep(10)
    eligible=[r for r in rows if r['eligible']]
    selected=max(eligible or rows,key=lambda r:(r['score'],-r['epoch']))
    save(OUT/f'{prefix}selection.json',dict(baseline=baseline,rows=rows,selected=selected,
        generic_guard_passed=bool(eligible),criterion='Max reviewed site val mAP50-95 with COCO val >= original-.01; earliest tie',
        fallback_if_no_guard='Report candidate but retain original; no automatic release'))


if __name__=='__main__':main()
