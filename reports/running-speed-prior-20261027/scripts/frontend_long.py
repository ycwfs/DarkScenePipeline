"""Recreate full sequential frontend; bind historical events by exact track+box.

No action network rerun: the saved full-video probabilities remain the paired baseline.
"""
import json
import time
import sys
from collections import deque
from common import OUT, ROOT, OP, WAVE, save
import cv2
import numpy as np
import torch
from darkpipe.config import PipelineConfig
from darkpipe.person.runtime import Frontend, prepare_stages
from darkpipe.person.core import PersonTracker
from darkpipe.stages import build_stages


def main():
    torch.set_num_threads(2);cv2.setNumThreads(2)
    events=[json.loads(s)for s in (WAVE/'long-videos/new-01/candidate-events.jsonl').read_text().splitlines()]
    selected={r['frame_index']:r for r in events if r['label']=='chase'}
    source=json.loads((WAVE/'long-videos/new-01/summary.json').read_text())['source']
    cfg=PipelineConfig(device='cuda:0',recognize='off',ckpt_dir=str(OP/'ckpts'),proc_max_side=720,enhance_chunk=1)
    stages,_=build_stages(cfg)
    stages=[s for s in stages if not s.post_recognition]
    for s in stages:s.load('cuda:0')
    front=Frontend(cfg);tracker=PersonTracker();cap=cv2.VideoCapture(source['absolute_path']);assert cap.isOpened()
    fps=cap.get(cv2.CAP_PROP_FPS);index=0;records=[];beg=time.monotonic()
    while True:
        ok,raw=cap.read()
        if not ok:break
        frame,luma=prepare_stages(stages,cfg,raw)
        boxes,poses=front(frame,luma>=35)
        tracks=tracker.update(boxes,poses,frame,index/fps)
        for t in tracks:
            t.append(frame,index/fps,index,luma)
            # Metadata added dynamically here; candidate dataclass defines it explicitly.
            t.history[-1].frame_shape=frame.shape[:2]
        if index in selected:
            e=selected[index];ts=[t for t in tracks if t.tid==e['track_id']];assert len(ts)==1,(index,e['track_id'])
            t=ts[0];assert np.allclose(t.box,e['bbox'],atol=1e-4,rtol=0),(index,t.box,e['bbox'])
            win=t.window(fps);assert [s.frame_index for s in win]==e['source_frames'],index
            samples=[dict(frame_index=s.frame_index,timestamp=s.timestamp,pose=s.pose.tolist(),luma=s.luma,
                person_box=s.person_box.tolist(),frame_shape=s.frame_shape)for s in win]
            records.append(dict(event=e,samples=samples,exact_box_and_source_frames=True))
            save(OUT/'long-running-evidence.json',records)
        index+=1
        if index%250==0:
            save(OUT/'frontend-progress.json',dict(frames=index,seconds=index/fps,elapsed=time.monotonic()-beg,matched=len(records)))
            print('FRAMES',index,'MATCHED',len(records),flush=True)
    cap.release()
    for s in stages:s.close()
    assert len(records)==len(selected) and index==source['frames']
    save(OUT/'frontend-verification.json',dict(frames=index,matched=len(records),elapsed=time.monotonic()-beg,
        method='Exact sequential enhancer/detector/pose/tracker replay. No new RGB or skeleton-action inference; historical full probabilities reused.'))

if __name__=='__main__':main()
