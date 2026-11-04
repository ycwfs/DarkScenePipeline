"""Find normally lit, genuinely multi-person field takes for default-gate previews."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
from rtmlib import RTMO

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
OP = ROOT / 'platform/dark-demo-v2'
sys.path.insert(0, str(OP))
from darkpipe.stages.enhance_retinexformer import RetinexformerStage
from darkpipe.stages.downscale import DownscaleStage

torch.set_num_threads(4)
torch.cuda.set_device(5)
pose = RTMO(onnx_model=str(OP/'ckpts/rtmo-s_end2end.onnx'), model_input_size=(640,640),
            backend='onnxruntime', device='cuda:5', score_thr=.3)
enhance = RetinexformerStage(str(OP/'ckpts'),chunk=1)
enhance.load('cuda:5')
ds=DownscaleStage(720)
rows=[]
for p in sorted((ROOT/'real-actiondata-all').glob('*.mp4')):
    if not p.name.startswith(('挥手','追逐')):
        continue
    cap=cv2.VideoCapture(str(p))
    fi=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))//2
    cap.set(cv2.CAP_PROP_POS_FRAMES,fi)
    ok,f=cap.read();cap.release()
    if not ok:continue
    raw=ds([f])[0]
    luma=float((raw.astype(np.float32) @ np.array([.114,.587,.299])).mean())
    if luma<75:continue
    frame=enhance([raw])[0]
    k,s=pose(frame)
    valid=(s.mean(1)>=.5)&((s>.3).sum(1)>=10)
    if valid.sum()<2:continue
    rows.append(dict(name=p.stem,frame=fi,luma=luma,people=int(valid.sum())))
    for points,scores in zip(k[valid],s[valid]):
        q=points[scores>.3].astype(int)
        cv2.rectangle(frame,tuple(q.min(0)),tuple(q.max(0)),(0,255,0),2)
    (OUT/'candidates').mkdir(exist_ok=True)
    cv2.imwrite(str(OUT/'candidates'/f'{p.stem}.jpg'),frame)
    print(rows[-1],flush=True)
(OUT/'candidates.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
