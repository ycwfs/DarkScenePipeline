"""Exact full-source object-prior inference for repeated held-out low-score windows."""
from collections import defaultdict,OrderedDict
import time
import numpy as np
from common import *
import cv2,torch
from darkpipe.person.object_runtime import PersonObjectPrior,EvidenceFrame,SAMPLE_POSITIONS
from darkpipe.person.action_engine import ActionTask
from darkpipe.person.runtime import content_height
from darkpipe.stages.downscale import DownscaleStage
from darkpipe.stages.enhance_retinexformer import RetinexformerStage

def run_source(sid,selected,meta,enhancer,prior):
    needed=set();endings=defaultdict(list);features={}
    for r in selected:
        f=OUT/'features'/f"{r['id']}.npz"
        if not f.exists():
            f=AUDIT/'features'/f"{r['id']}.npz"
        assert f.exists(),r['id']
        with np.load(f)as a:values={k:a[k]for k in a.files}
        features[r['id']]=values
        needed.update(int(values['source_frames'][i])for i in SAMPLE_POSITIONS)
        endings[int(values['source_frames'][-1])].append(r)
    info=meta[sid];cap=cv2.VideoCapture(info['source']);assert cap.isOpened()
    fps=float(info['fps']);assert abs(cap.get(cv2.CAP_PROP_FPS)-fps)<.0001
    index=0;frames=OrderedDict();down=DownscaleStage(720);started=time.monotonic();records=[]
    try:
        for fi in sorted(needed):
            while index<fi:assert cap.grab();index+=1
            ok,raw=cap.read();index+=1;assert ok
            if info['processed']:raw=raw[:content_height(raw.shape[0])]
            small=down([raw])[0];frame=small if info['processed']else enhancer([small])[0]
            frames[fi]=EvidenceFrame(fi,frame,raw.shape)
            while frames and fi-next(iter(frames))>fps*1.2:frames.popitem(last=False)
            for r in endings.get(fi,[]):
                f=features[r['id']]
                win=tuple(type('Sample',(),dict(frame_index=int(n),timestamp=float(t),pose=p,luma=None if np.isnan(l)else float(l),
                    person_box=b,binding_valid=bool(v),frame_shape=tuple(f['shape'])))()
                    for n,t,p,l,b,v in zip(f['source_frames'],f['timestamps'],f['pose'],f['lumas'],f['boxes'],f['binding']))
                chosen=dict.fromkeys(int(f['source_frames'][i])for i in SAMPLE_POSITIONS)
                assert all(n in frames for n in chosen),(r['id'],'sample frame cache miss')
                task=ActionTask((sid,r['track_id'],r['history_generation']),win,tuple(f['boxes'][-1]),1.,0.,0.,tuple(frames[n]for n in chosen))
                label=r['raw_label'];row=dict(label=label,confidence=1.,decision_scores=[float(c==label)for c in LABELS],raw_luma=r['raw_luma'])
                prior.apply(task,row)
                records.append(dict(id=r['id'],actions={label:row['object_prior']['samples']}))
        assert len(records)==len(selected)
        for rec in records:
            save(OUT/'object-evidence'/(rec['id']+'.json'),dict(**rec,exact_features_verified=True,
                source_id=sid,provenance='exact_full_source_enhance_and_roi_detection',
                model_sha256=sha(OP/'ckpts/yolo26m_objects_20261026.onnx')))
        return dict(source_id=sid,windows=len(records),seconds=time.monotonic()-started)
    finally:cap.release();prior.cache.clear()

def main():
    torch.set_num_threads(2);cv2.setNumThreads(1)
    groups=defaultdict(list)
    for r in read(OUT/'object-needed.json'):groups[r['source_id']].append(r)
    meta=read(OUT/'source-meta.json');enhancer=RetinexformerStage(str(OP/'ckpts'),chunk=1);enhancer.load('cuda:0')
    prior=PersonObjectPrior(str(OP/'ckpts'),'cuda:0',.001,35.);started=time.monotonic();results=[]
    try:
        for sid,selected in groups.items():
            results.append(run_source(sid,selected,meta,enhancer,prior));save(OUT/'object-progress.json',dict(results=results,remaining=len(groups)-len(results),elapsed=time.monotonic()-started));print(results[-1],flush=True)
    finally:enhancer.close();prior.close()
    save(OUT/'objects-complete.json',dict(complete=True,results=results,seconds=time.monotonic()-started))

if __name__=='__main__':main()
