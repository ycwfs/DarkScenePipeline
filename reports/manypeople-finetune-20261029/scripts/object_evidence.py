"""Untruncated seven-class evidence for BOTH object actions on every evaluated ROI."""
import argparse
from collections import defaultdict, OrderedDict
import sys
import time
from types import SimpleNamespace
from common import *
sys.path.insert(0,str(OP))
import cv2
import numpy as np
import torch
from darkpipe.person.object_runtime import PersonObjectPrior,EvidenceFrame,SAMPLE_POSITIONS
from darkpipe.person.action_engine import ActionTask
from darkpipe.person.runtime import content_height
from darkpipe.stages.downscale import DownscaleStage
from darkpipe.stages.enhance_retinexformer import RetinexformerStage


def run(source_rows,enhancer,prior):
    source_id=source_rows[0]['source_id'];dest=OUT/'object-evidence'/(source_id+'.json')
    if dest.exists():return
    needed=set();endings=defaultdict(list);features={}
    for r in source_rows:
        path=OUT/'features'/(r['id']+'.npz')
        if not path.exists():return  # A later pass resumes when extraction has completed.
        with np.load(path)as a:f={k:a[k]for k in a.files}
        features[r['id']]=f
        needed.update(int(f['source_frames'][i])for i in SAMPLE_POSITIONS)
        endings[int(f['source_frames'][-1])].append(r)
    first=source_rows[0];assert sha(first['source'])==first['source_sha256']
    cap=cv2.VideoCapture(first['source']);assert cap.isOpened()
    down=DownscaleStage(720);index=0;frames=OrderedDict();records=[];start=time.monotonic()
    try:
        for fi in sorted(needed):
            while index<fi:assert cap.grab();index+=1
            ok,raw=cap.read();index+=1;assert ok
            processed=first.get('processed',False)
            raw=raw[:content_height(raw.shape[0])]if processed else raw
            shape=raw.shape;small=down([raw])[0];frame=small if processed else enhancer([small])[0]
            frames[fi]=EvidenceFrame(fi,frame,shape)
            while frames and fi-next(iter(frames))>first['fps']*1.2:frames.popitem(last=False)
            for r in endings.get(fi,[]):
                f=features[r['id']]
                win=tuple(SimpleNamespace(frame_index=int(n),timestamp=float(t),pose=p,luma=None if np.isnan(l)else float(l),
                    person_box=b,binding_valid=bool(valid),frame_shape=tuple(f['shape']))
                    for n,t,p,l,b,valid in zip(f['source_frames'],f['timestamps'],f['pose'],f['lumas'],f['boxes'],f['binding']))
                chosen=dict.fromkeys(int(f['source_frames'][i])for i in SAMPLE_POSITIONS)
                assert all(n in frames for n in chosen)
                task=ActionTask((r['source_id'],r['track_id'],r['generation']),win,tuple(f['boxes'][-1]),1.,0.,0.,tuple(frames[n]for n in chosen))
                actions={}
                for label in ('drink','pick_up'):
                    row=dict(label=label,confidence=1.,decision_scores=[float(c==label)for c in LABELS],raw_luma=r['raw_luma'])
                    prior.apply(task,row);actions[label]=row['object_prior']['samples']
                records.append(dict(id=r['id'],actions=actions))
        assert len(records)==len(source_rows)
        save(dest,dict(source_id=source_id,records=records,threshold_floor=.001,lum_gate_floor=30.,
            exact_person_samples=True,model_sha256=sha(OP/'ckpts/yolo26m_objects_20261026.onnx'),seconds=time.monotonic()-start))
        print('OBJECT',source_id,len(records),round(time.monotonic()-start,1),flush=True)
    finally:cap.release();prior.cache.clear()


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',default='0/1');p.add_argument('--wait',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(1)
    plan=read(OUT/'dataset-plan.json')['records']
    rows=[r for r in plan if r['split'] in ('val','test') and r['label_kind']!='pseudo'
        and (r['scope']=='new' or r.get('current_index')is not None)]
    groups=defaultdict(list)
    for r in rows:groups[r['source_id']].append(r)
    i,n=map(int,a.shard.split('/'));sources=list(groups)[i::n]
    enhancer=RetinexformerStage(str(OP/'ckpts'),chunk=1);enhancer.load('cuda:0')
    prior=PersonObjectPrior(str(OP/'ckpts'),'cuda:0',.001,30.)
    start=time.monotonic()
    try:
        while True:
            for s in sources:run(groups[s],enhancer,prior)
            missing=[s for s in sources if not (OUT/'object-evidence'/(s+'.json')).exists()]
            if not missing:break
            if not a.wait:raise RuntimeError(f'Extraction not ready: {missing}')
            if time.monotonic()-start>7200:raise TimeoutError('Extraction incomplete after two hours')
            time.sleep(20)
    finally:enhancer.close();prior.close()
    save(OUT/f'object-{i}-complete.json',dict(complete=True,sources=sources))


if __name__=='__main__':main()
