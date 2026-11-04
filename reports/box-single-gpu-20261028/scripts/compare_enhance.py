"""Paired real-frame pixel/latency comparison of FP16 PyTorch and TensorRT NTIRE."""
import json
import os
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'NVIDIA-Box/runtime'))
from darkpipe.stages.enhance_retinexformer import RetinexformerStage
from darkpipe.stages.downscale import DownscaleStage
torch.set_num_threads(2);cv2.setNumThreads(2)
os.environ['DARKPIPE_BOX_BACKEND']='cuda'
reference=RetinexformerStage(str(ROOT/'NVIDIA-Box/weights'));reference.load('cuda:0')
os.environ['DARKPIPE_BOX_BACKEND']='tensorrt'
candidate=RetinexformerStage(str(ROOT/'NVIDIA-Box/weights'));candidate.load('cuda:0')
down=DownscaleStage(720);rows=[]
dest=OUT/'enhancement-comparison';dest.mkdir(exist_ok=True)
for item in json.loads((OUT/'inputs.json').read_text()):
    if item.get('processed'):continue
    cap=cv2.VideoCapture(str(ROOT/item['input']))
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for index in (0,count//2,max(0,count-1)):
        cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,raw=cap.read()
        assert ok,(item['id'],index)
        raw=down([raw])[0]
        start=time.perf_counter();a=reference([raw])[0];t1=time.perf_counter()-start
        start=time.perf_counter();b=candidate([raw])[0];t2=time.perf_counter()-start
        difference=a.astype(float)-b.astype(float)
        mse=float(np.mean(difference**2))
        rows.append(dict(id=item['id'],frame=index,shape=list(a.shape),
            psnr=99. if mse==0 else 10*np.log10(255**2/mse),
            mae=float(np.abs(difference).mean()),maximum_absolute_difference=float(np.abs(difference).max()),
            pytorch_ms=t1*1000,tensorrt_ms=t2*1000))
        if index==count//2:
            cv2.imwrite(str(dest/(item['id']+'.jpg')),np.hstack([a,b]))
    cap.release()
result=dict(frames=len(rows),minimum_psnr=min(r['psnr']for r in rows),maximum_mae=max(r['mae']for r in rows),rows=rows)
(OUT/'enhancement-comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print({k:v for k,v in result.items()if k!='rows'},flush=True)
candidate.close();reference.close()
