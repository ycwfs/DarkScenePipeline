"""Supplement missing geometry by a complete causal frontend replay, never nearest time."""
import argparse,time
from collections import defaultdict
from common import *
import cv2,torch
from darkpipe.config import PipelineConfig
from darkpipe.person.runtime import Frontend
from darkpipe.person.core import PersonTracker,iou
from darkpipe.stages.downscale import DownscaleStage
from darkpipe.stages.enhance_retinexformer import RetinexformerStage
from darkpipe.stages.recognize import frame_luma

def main():
    a=argparse.ArgumentParser();a.add_argument('--shard',type=int,required=True);args=a.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(1)
    enhancer=RetinexformerStage(str(OP/'ckpts'),chunk=1);enhancer.load('cuda:0')
    front=Frontend(PipelineConfig(device='cuda:0',ckpt_dir=str(OP/'ckpts')));down=DownscaleStage(720)
    groups=defaultdict(list)
    for r in read(OUT/'frontend-needed.json'):groups[r['source_id']].append(r)
    meta=read(OUT/'source-meta.json')
    try:
        for sid in sorted(groups)[args.shard::6]:
            done=OUT/'frontend'/(sid+'.json')
            if done.exists():continue
            rows=groups[sid];targets=defaultdict(list);ranges=defaultdict(list)
            for r in rows:
                targets[r['source_frames'][-1]].append(r)
                ranges[(r['track_id'],r['history_generation'])].append((r['source_frames'][0]-6,r['source_frames'][-1]))
            tracker=PersonTracker();seen={};generations={};cap=cv2.VideoCapture(meta[sid]['source']);assert cap.isOpened()
            fps=meta[sid]['fps'];assert abs(cap.get(cv2.CAP_PROP_FPS)-fps)<.0001
            matched=[];start=time.monotonic();last=max(targets)
            try:
                for index in range(last+1):
                    ok,raw=cap.read();assert ok,(sid,index)
                    small=down([raw])[0];luma=frame_luma(small);frame=enhancer([small])[0]
                    boxes,poses=front(frame,luma>=35);stamp=index/fps;tracks=tracker.update(boxes,poses,frame,stamp)
                    for t in tracks:
                        if stamp-seen.get(t.tid,stamp)>.2:generations[t.tid]=generations.get(t.tid,0)+1
                        seen[t.tid]=stamp;key=(t.tid,generations.get(t.tid,0))
                        if any(lo<=index<=hi for lo,hi in ranges.get(key,[])):
                            binding=not any(iou(t.box,o.box)>.4 for o in tracks if o is not t)
                            t.append(frame,stamp,index,luma,binding_valid=binding)
                    for r in targets.get(index,[]):
                        found=[t for t in tracks if t.tid==r['track_id']];assert len(found)==1,(r['id'],'identity')
                        t=found[0];assert generations.get(t.tid,0)==r['history_generation'],r['id']
                        win=t.window(fps);assert win is not None;write_feature(r,win);matched.append(r['id'])
                    if index%500==0:
                        progress=dict(source=sid,frames=index+1,total=last+1,matched=len(matched),expected=len(rows),seconds=time.monotonic()-start)
                        save(OUT/'progress'/(sid+'.json'),progress);print(progress,flush=True)
                assert len(matched)==len(rows)
                save(done,dict(source=sid,frames=last+1,ids=matched,seconds=time.monotonic()-start,exact_track_generation_box_frames=True))
            finally:cap.release()
    finally:enhancer.close()

if __name__=='__main__':main()
