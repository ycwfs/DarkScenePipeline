"""Fill native frames missing from the frozen five-frame object observations."""
import argparse
from bisect import bisect_left
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

OUT=Path(__file__).resolve().parent
BASE=OUT.parent/'object-prior-finetune-20261025'
SINGLE=OUT.parent/'object-prior-single-frame-20261026'
sys.path.insert(0,str(BASE/'scripts'));sys.path.insert(0,str(SINGLE))
from evaluate_actions import (OP, DownscaleStage, RetinexformerStage, associate_poses,
    extract_objects, frame_luma, overlap, poses, session)
from replay import single_support


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def run(job: dict, obj, pose_session, enhancer) -> dict:
    target=OUT/'dense'/f'{job["id"]}.json'
    if target.exists():return json.loads(target.read_text())['summary']
    old=json.loads((BASE/'action-evaluation'/f'{job["id"]}.json').read_text())
    old_records={r['event_index']:r for r in old['records']}
    original={i:{s['frame']:s for s in r['samples']}for i,r in old_records.items()}
    requested=defaultdict(list)
    for e in job['events']:
        for fi in range(min(e['source_frames']),max(e['source_frames'])+1):
            if fi not in original[e['event_index']]:requested[fi].append(e)
    raw_data=json.loads(Path(job['cache']).read_text())
    by_id=defaultdict(list)
    for e in raw_data:by_id[(e['track_id'],e['history_generation'])].append(e)
    indexed=[]
    for identity,events in by_id.items():
        events.sort(key=lambda r:r['frame_index'])
        indexed.append((identity,[e['frame_index']for e in events],events))
    def people_at(fi: int) -> list:
        people=[]
        for (tid,generation),indices,events in indexed:
            if fi<indices[0]-job['fps']*.55 or fi>indices[-1]+job['fps']*.55:continue
            at=bisect_left(indices,fi);candidates=[i for i in (at-1,at)if 0<=i<len(events)]
            nearest=events[min(candidates,key=lambda i:(abs(indices[i]-fi),i))]
            if abs(nearest['frame_index']-fi)<=job['fps']*.55:
                people.append(dict(id=tid,generation=generation,box=np.asarray(nearest['bbox'],float),pose=None))
        return people
    cap=cv2.VideoCapture(job['source']);assert cap.isOpened()
    down=DownscaleStage(720);new_samples={};next_frame=0;started=time.perf_counter()
    try:
        for step,fi in enumerate(sorted(requested)):
            if fi-next_frame>job['fps']*3:
                cap.set(cv2.CAP_PROP_POS_FRAMES,fi);next_frame=fi
            while next_frame<fi:
                assert cap.grab(),(job['id'],next_frame);next_frame+=1
            ok,raw=cap.read();next_frame+=1
            assert ok and abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-fi-1)<.51
            small=down([raw])[0];h,w=small.shape[:2]
            scale=np.array([raw.shape[1]/w,raw.shape[0]/h]);people=people_at(fi)
            if people and frame_luma(small)>=30:
                xy,cf=poses(pose_session,enhancer([small])[0])
                ps=associate_poses(np.asarray([p['box']for p in people]),xy,cf)
                for p,pose in zip(people,ps):p['pose']=pose
            for p in people:
                p['box']*=np.tile(scale,2)
                if p['pose']is not None:
                    p['pose']=np.asarray(p['pose'],float);p['pose'][:,:2]*=scale
            for e in requested[fi]:
                key=(fi,e['track_id'],e['history_generation'],e['baseline'])
                if key in new_samples:continue
                person=next((p for p in people if p['id']==e['track_id'] and p['generation']==e['history_generation']),None)
                if person is None:
                    sample=dict(frame=fi,timestamp=fi/job['fps'],binding_valid=False,
                        person_box=(np.asarray(e['bbox'])*np.tile(scale,2)).tolist(),pose=None,detections=[])
                else:
                    others=[p for p in people if p is not person]
                    result=extract_objects(obj,raw,person['box'],person['pose'],[p['pose']for p in others],e['baseline'])
                    sample=dict(frame=fi,timestamp=fi/job['fps'],
                        binding_valid=not any(overlap(person['box'],p['box'])>.4 for p in others),
                        person_box=person['box'].tolist(),pose=None if person['pose']is None else person['pose'].tolist(),**result)
                new_samples[key]=sample
            if (step+1)%100==0:
                print(job['id'],step+1,'/',len(requested),'decoded frames',round(time.perf_counter()-started,1),'s',flush=True)
    finally:cap.release()
    records=[]
    for e in job['events']:
        previous=old_records[e['event_index']];seq=[]
        for fi in range(min(e['source_frames']),max(e['source_frames'])+1):
            if fi in original[e['event_index']]:sample=original[e['event_index']][fi]
            else:sample=new_samples[(fi,e['track_id'],e['history_generation'],e['baseline'])]
            seq.append(sample)
        sparse=single_support(previous);dense=single_support(dict(previous,samples=seq))
        if sparse['single_roi']!='other':assert dense['single_roi']==sparse['single_roi']
        records.append(dict(previous,samples=seq,five_frame_label=sparse['single_roi'],
            label=dense['single_roi'],roi_evidence=dense['roi_evidence'],
            one_frame_interaction_label=dense['single_interaction'],
            note='Dense support only; original conservative veto and strict field are the frozen five-sample values.'))
    summary=dict(source_id=job['id'],events=len(records),original_sample_observations=sum(len(r['samples'])for r in old['records']),
        dense_sample_observations=sum(len(r['samples'])for r in records),new_unique_person_frames=len(new_samples),
        new_decoded_frames=len(requested),retained=sum(r['label']!='other'for r in records),
        five_frame_retained=sum(r['five_frame_label']!='other'for r in records),seconds=time.perf_counter()-started)
    save(target,dict(job={k:v for k,v in job.items()if k!='events'},records=records,summary=summary,
        source_five_frame_sha256=hashlib.sha256((BASE/'action-evaluation'/f'{job["id"]}.json').read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()))
    print(json.dumps(summary),flush=True)
    return summary


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--numbers',type=int,nargs='+',required=True);a=p.parse_args()
    import torch
    torch.set_num_threads(2);cv2.setNumThreads(1)
    lock=json.loads((BASE/'prior-lock.json').read_text())
    model=BASE/'models/yolo26m_objects_candidate.onnx'
    assert hashlib.sha256(model.read_bytes()).hexdigest()==lock['model_sha256']
    obj=session(model);pose=session(OP/'ckpts/rtmo-s_person_20261024.onnx')
    enhancer=RetinexformerStage(str(OP/'ckpts'),chunk=1);enhancer.load('cuda:0')
    jobs=json.loads((BASE/'action-plan-new.json').read_text());summaries=[]
    try:
        for job in jobs:
            if job['number']in a.numbers:summaries.append(run(job,obj,pose,enhancer))
    finally:enhancer.close()
    save(OUT/('worker-'+ '-'.join(map(str,a.numbers))+'.json'),dict(complete=True,summaries=summaries))


if __name__=='__main__':main()
