"""Select using the deployed one-to-one head, explicitly nms=False in new Ultralytics."""
import json
from pathlib import Path
import torch
from ultralytics import YOLO
from prepare import OUT,ROOT,save


if __name__=='__main__':
    torch.set_num_threads(2)
    paths=[ROOT.parent/'yolo26m-detect.pt']+sorted((OUT/'runs/detector/weights').glob('epoch*.pt'),key=lambda p:int(p.stem[5:]))
    rows=[];best=None
    for path in paths:
        model=YOLO(str(path))
        result=model.val(data=str(OUT/'detector.yaml'),split='val',imgsz=640,batch=12,device=0,
            workers=2,nms=False,plots=False,save_json=False,project=str(OUT/'runs'),name='detector_select',exist_ok=True)
        row=dict(path=str(path),metrics=result.results_dict,head='one-to-one, nms=False')
        rows.append(row)
        if best is None or result.box.map>best['metrics']['metrics/mAP50-95(B)']:best=row
        save(OUT/'detector-selection.json',dict(rows=rows,best=best,
            rationale='same one-to-one head as deployed ONNX; no test data used'))
        print('[selected so far]',best['path'],best['metrics']['metrics/mAP50-95(B)'],flush=True)
    fine=max(rows[1:],key=lambda r:r['metrics']['metrics/mAP50-95(B)'])
    save(OUT/'detector-selection.json',dict(rows=rows,best=best,best_finetuned=fine,
        rationale='same one-to-one head as deployed ONNX; no test data used'))
    old=json.loads((OUT/'detector-training-result.json').read_text())
    old.update(best=fine['path'],metrics=fine['metrics'],recommended=best['path'],selection='detector-selection.json')
    save(OUT/'detector-training-result.json',old)
