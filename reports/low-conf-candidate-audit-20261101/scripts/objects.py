"""Supplement missing object priors on exact sampled enhanced frames/actor ROIs."""
from collections import defaultdict,OrderedDict
import time
from common import *
import cv2,torch
from darkpipe.person.object_runtime import PersonObjectPrior,EvidenceFrame,SAMPLE_POSITIONS
from darkpipe.person.action_engine import ActionTask
from darkpipe.person.runtime import content_height
from darkpipe.stages.downscale import DownscaleStage
from darkpipe.stages.enhance_retinexformer import RetinexformerStage

def run(sid,rows,meta,enhancer,prior):
    pending=[r for r in rows if not(OUT/'object-evidence'/(r['id']+'.json')).exists()]
    ready=[r for r in pending if(OUT/'features'/(r['id']+'.npz')).exists()]
    if not ready:return 0
    windows={r['id']:samples(OUT/'features'/(r['id']+'.npz'))for r in ready}
    needed=set();ends=defaultdict(list)
    for r in ready:
        ss=windows[r['id']];needed.update(ss[j].frame_index for j in SAMPLE_POSITIONS);ends[ss[-1].frame_index].append(r)
    cap=cv2.VideoCapture(meta['source']);assert cap.isOpened();index=0;frames=OrderedDict();down=DownscaleStage(720);start=time.monotonic();written=0
    try:
        for fi in sorted(needed):
            while index<fi:assert cap.grab();index+=1
            ok,raw=cap.read();index+=1;assert ok
            if meta['processed']:raw=raw[:content_height(raw.shape[0])]
            small=down([raw])[0];frame=small if meta['processed']else enhancer([small])[0]
            frames[fi]=EvidenceFrame(fi,frame,raw.shape)
            while frames and fi-next(iter(frames))>meta['fps']*1.2:frames.popitem(last=False)
            for r in ends.get(fi,[]):
                ss=windows[r['id']];chosen=dict.fromkeys(ss[j].frame_index for j in SAMPLE_POSITIONS);assert all(n in frames for n in chosen)
                task=ActionTask((sid,r['track_id'],r['history_generation']),tuple(ss),tuple(ss[-1].person_box),1.,0.,0.,tuple(frames[n]for n in chosen))
                row=dict(label=r['raw_label'],confidence=1.,decision_scores=[float(c==r['raw_label'])for c in LABELS],raw_luma=r['raw_luma'])
                beg=time.perf_counter();prior.apply(task,row);ms=(time.perf_counter()-beg)*1000
                save(OUT/'object-evidence'/(r['id']+'.json'),dict(id=r['id'],actions={r['raw_label']:row['object_prior']['samples']},exact_features_verified=True,provenance='new_exact_frame_gpu_inference',object_ms=ms,model_sha256=sha(OP/'ckpts/yolo26m_objects_20261026.onnx')))
                written+=1
        print('OBJECT',sid,len(ready),round(time.monotonic()-start,2),flush=True)
    finally:cap.release();prior.cache.clear()
    return written

def main():
    torch.set_num_threads(2);cv2.setNumThreads(1)
    groups=defaultdict(list)
    for r in read(OUT/'object-needed.json'):groups[r['source_id']].append(r)
    meta=read(OUT/'source-meta.json');enhancer=RetinexformerStage(str(OP/'ckpts'),chunk=1);enhancer.load('cuda:0')
    prior=PersonObjectPrior(str(OP/'ckpts'),'cuda:0',.001,35.);start=time.monotonic()
    try:
        while True:
            for sid,rows in groups.items():run(sid,rows,meta[sid],enhancer,prior)
            missing=[r['id']for rr in groups.values()for r in rr if not(OUT/'object-evidence'/(r['id']+'.json')).exists()]
            save(OUT/'object-progress.json',dict(missing=missing,elapsed=time.monotonic()-start))
            if not missing:break
            if time.monotonic()-start>7200:raise TimeoutError(missing)
            time.sleep(20)
    finally:enhancer.close();prior.close()
    save(OUT/'objects-complete.json',dict(complete=True,elapsed=time.monotonic()-start))

if __name__=='__main__':main()
