"""Replay fixed reviewed test ROIs; compare action numerics without a changed detector."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);p.add_argument('--name',required=True);a=p.parse_args()
sys.path.insert(0,a.runtime)
from darkpipe.person.core import Sample
from darkpipe.person.action_engine import ActionEngine,ActionTask
dataset=ROOT/'reports/person-finetune-20261023'
rows=[r for r in json.loads((dataset/'training-data.json').read_text())['records']if r['split']=='test']
torch.set_num_threads(2)
engine=ActionEngine('cuda:0',max_batch=2,options=dict(ckpt_dir=str(ROOT/'NVIDIA-Box/weights'),
    alpha=.972,threshold=.925,lum_gate=35.,no_pose='soft',person_object_prior=False,
    wave_pose_min_frames=3,running_pose_min_speed=.35))
results=[]
try:
    for offset in range(0,len(rows),2):
        batch=rows[offset:offset+2];tasks=[]
        for index,row in enumerate(batch):
            with np.load(dataset/row['path'])as z:
                rgb,pose=z['rgb'],z['pose']
            window=tuple(Sample(row['timestamps'][i],row['source_frames'][i],rgb[i],
                               np.asarray(row['crop_boxes'][i]),pose[i],row['lumas'][i])for i in range(32))
            tasks.append(ActionTask((1,offset+index,0),window,tuple(row['crop_boxes'][-1]),.99,
                                    time.monotonic(),time.monotonic()-row['end']))
        for row,output in zip(batch,engine.predict(tasks)):
            results.append(dict(id=row['id'],truth=row['label'],source=row['source'],output=output))
finally:
    engine.close()
(OUT/(a.name+'.json')).write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
print(len(results),'fixed test windows',flush=True)
