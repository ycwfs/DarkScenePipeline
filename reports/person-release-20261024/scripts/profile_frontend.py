import json
import sys
import time
from pathlib import Path
sys.path.insert(0,'/app')
import cv2
import numpy as np
import torch
from darkpipe.config import PipelineConfig
from darkpipe.person.runtime import Frontend
from darkpipe.stages.enhance_retinexformer import RetinexformerStage
from darkpipe.stages.downscale import DownscaleStage

torch.set_num_threads(2);cv2.setNumThreads(2)
cfg=PipelineConfig(ckpt_dir='/app/ckpts',device='cuda:0')
enh=RetinexformerStage(cfg.ckpt_dir,chunk=1);enh.load(cfg.device)
front=Frontend(cfg)
cap=cv2.VideoCapture('/inputs/挥手.mp4');ok,raw=cap.read();assert ok;cap.release()
raw=DownscaleStage(720)([raw])[0]
records=[]
for i in range(30):
    t=time.perf_counter();frame=enh([raw])[0];a=time.perf_counter()
    boxes,poses=front(frame,True);b=time.perf_counter()
    if i>=5:records.append(dict(enhance=a-t,vision=b-a,people=len(boxes)))
print(json.dumps(dict(means={k:float(np.mean([r[k] for r in records])) for k in records[0]},records=records)))
