"""Replay frozen action outputs with native-pixel, same-person object observations.

This is an offline candidate ablation, not a production streaming implementation.
Historical frame tracks/poses are exact cached inputs. New feedback has event boxes
only: nearest event boxes are bounded to .55 s and ambiguous binding abstains.
"""
import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from build_dataset import OUT, ROOT, save
from evaluate_objects import session, detect
from prior_rules import decide, overlap

OP = ROOT/'platform/dark-demo-v2'
sys.path.insert(0, str(OP))
from darkpipe.constants import BEHAVIORS
from darkpipe.person.core import fuse_person, associate_poses
from darkpipe.person.runtime import content_height
from darkpipe.stages.recognize import reject_probs, frame_luma
from darkpipe.stages.downscale import DownscaleStage
from darkpipe.stages.enhance_retinexformer import RetinexformerStage


def decision(event: dict, old=False) -> dict:
    if old:
        return fuse_person(np.asarray(event['rgb_probs'],np.float32),
            None if event['skeleton_probs'] is None else np.asarray(event['skeleton_probs'],np.float32),
            [event['raw_luma']], alpha=.95, threshold=.85, lum_gate=30., no_pose='soft')
    probs=np.asarray(event['fused_probs'],np.float32)
    d=reject_probs(probs, BEHAVIORS, .85)
    return dict(label=BEHAVIORS[int(d.argmax())],confidence=float(d.max()))


def bright(frame: np.ndarray) -> np.ndarray:
    luma=float(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).mean())
    gamma=1. if luma>=75 else .5 if luma>=15 else .32
    lut=np.round((np.arange(256)/255.)**gamma*255).astype(np.uint8)
    return cv2.LUT(frame,lut)


def crop_rect(box: np.ndarray, shape: tuple, margin=.3) -> np.ndarray:
    m=(box[2:]-box[:2])*margin
    return np.r_[np.floor(np.maximum(0,box[:2]-m)),
                 np.ceil(np.minimum([shape[1],shape[0]],box[2:]+m))].astype(int)


def poses(s, image: np.ndarray) -> tuple:
    h,w=image.shape[:2];ratio=640/max(h,w)
    canvas=np.full((640,640,3),114,np.uint8)
    scaled=cv2.resize(image,(int(w*ratio),int(h*ratio)))
    canvas[:scaled.shape[0],:scaled.shape[1]]=scaled
    dets,points=s.run(None,{s.get_inputs()[0].name:canvas.transpose(2,0,1)[None].astype(np.float32)})
    dets,points=dets[0],points[0]
    xywh=dets[:,:4].copy();xywh[:,2:]-=xywh[:,:2]
    keep=np.asarray(cv2.dnn.NMSBoxes(xywh.tolist(),dets[:,4].tolist(),.45,.45),int).reshape(-1)
    return points[keep,:,:2]/ratio,points[keep,:,2]


def extract_objects(s, raw: np.ndarray, box: np.ndarray, pose, other_poses: list, label: str) -> dict:
    rect=crop_rect(box,raw.shape)
    regions=[rect]
    if pose is not None:
        p=np.asarray(pose)
        indices=[0,1,2,3,4,9,10] if label=='drink' else [9,10,13,14]
        pts=p[indices];pts=pts[pts[:,2]>=.5,:2]
        if len(pts)>=2:
            padding=max(24.,(box[3]-box[1])*.12)
            lo=np.floor(np.maximum(rect[:2],pts.min(0)-padding)).astype(int)
            hi=np.ceil(np.minimum(rect[2:],pts.max(0)+padding)).astype(int)
            if np.all(hi-lo>=24) and np.prod(hi-lo)<np.prod(rect[2:]-rect[:2])*.8:
                regions.append(np.r_[lo,hi])
    rows=[];elapsed=[]
    for r in regions:
        x1,y1,x2,y2=r;image=bright(raw[y1:y2,x1:x2])
        started=time.perf_counter();pred=detect(s,image);elapsed.append(time.perf_counter()-started)
        pred[:,:4]+=np.tile(r[:2],2)
        rows.extend(pred[pred[:,4]>=.1].tolist())
    kept=[]
    for r in sorted(rows,key=lambda r:-r[4]):
        if any(int(r[5])==int(k[5]) and overlap(np.array(r[:4]),np.array(k[:4]))>.5 for k in kept):continue
        if int(r[5])!=6 and pose is not None:
            center=(np.array(r[:2])+r[2:4])/2
            wrist=np.asarray(pose)[[9,10]];wrist=wrist[wrist[:,2]>=.5,:2]
            own=min((np.linalg.norm(center-p) for p in wrist),default=np.inf)
            other=min((np.linalg.norm(center-p[:2]) for ps in other_poses if ps is not None
                       for p in np.asarray(ps)[[9,10]] if p[2]>=.5),default=np.inf)
            # Nearby people's objects cannot veto the selected actor.
            if other<own*1.25 and other<.15*(box[3]-box[1]):continue
        kept.append(r)
    return dict(detections=kept,person_roi=rect.tolist(),regions=[r.tolist()for r in regions],seconds=elapsed)


def jobs(kind: str) -> list:
    output=[]
    if kind=='old':
        for path in sorted((ROOT/'reports/person-release-20261024/data').glob('*/result.json')):
            data=json.loads(path.read_text());v=data['video']
            if v['split'] not in ('val','test'):continue
            selected=[]
            for idx,e in enumerate(data['events']['30']):
                pred=decision(e,True)
                if pred['label'] in ('drink','pick_up') or e['truth'] in ('drink','pick_up'):
                    selected.append(dict(e,event_index=idx,baseline=pred['label'],baseline_conf=pred['confidence']))
            if selected:output.append(dict(id=v['id'],source=str(ROOT/v['source']),fps=v['fps'],
                group=v['group'],split=v['split'],processed=v['processed'],events=selected,cache=str(path)))
    else:
        base=ROOT/'reports/manypeople-inference-20261025'
        for v in json.loads((base/'input-manifest.json').read_text()):
            path=base/'events'/f'{v["number"]:02d}_events.json'
            events=json.loads(path.read_text());selected=[]
            for idx,e in enumerate(events):
                pred=decision(e)
                if pred['label'] in ('drink','pick_up'):
                    selected.append(dict(e,event_index=idx,truth=None,baseline=pred['label'],baseline_conf=pred['confidence']))
            output.append(dict(id=f'new-{v["number"]:02d}',number=v['number'],source=v['absolute_path'],
                fps=float(Fraction(v['fps'])),group=f'new-{v["number"]:02d}',split='feedback',processed=False,
                events=selected,cache=str(path),all_event_count=len(events)))
    return output


def run(job: dict, object_session, pose_session, enhancer) -> None:
    destination=OUT/'action-evaluation'/f'{job["id"]}.json'
    if destination.exists():return
    requested=defaultdict(list)
    for e in job['events']:
        for f in np.unique(np.asarray(e['source_frames'])[np.linspace(0,31,5).round().astype(int)]):
            requested[int(f)].append(e)
    raw_data=json.loads(Path(job['cache']).read_text())
    old=job['split']!='feedback'
    by_id=defaultdict(list)
    if not old:
        for e in raw_data:by_id[(e['track_id'],e['history_generation'])].append(e)
        for rows in by_id.values():rows.sort(key=lambda r:r['frame_index'])
    def frame_people(fi):
        if old:
            meta=raw_data['frames'][fi];people=meta['tracks']['30'];res=[]
            for t in people:
                box=np.asarray(t['bbox']);candidates=[(overlap(box,np.array(b[:4])),i)for i,b in enumerate(meta['boxes'])]
                p=meta['poses'][max(candidates)[1]] if candidates and max(candidates)[0]>.5 else None
                res.append(dict(id=t['track_id'],generation=0,box=box,pose=p,valid=True))
            return res
        res=[]
        for (tid,generation),events in by_id.items():
            # Nearest stored event bounds, never extend a lost ID arbitrarily.
            nearest=min(events,key=lambda e:abs(e['frame_index']-fi))
            if abs(nearest['frame_index']-fi)<=job['fps']*.55:
                res.append(dict(id=tid,generation=generation,box=np.asarray(nearest['bbox']),pose=None,valid=True))
        return res
    cap=cv2.VideoCapture(job['source']);assert cap.isOpened()
    samples=defaultdict(list);frame_times=[];down=DownscaleStage(720)
    try:
        next_frame=0
        for fi in sorted(requested):
            # Decode forward for nearby requests; seek across long unused spans.
            if fi-next_frame>job['fps']*3:
                cap.set(cv2.CAP_PROP_POS_FRAMES,fi)
                next_frame=fi
            while next_frame<fi:
                assert cap.grab(),(job['source'],next_frame)
                next_frame+=1
            ok,raw=cap.read();next_frame+=1;assert ok,(job['source'],fi)
            assert abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-fi-1)<.51
            if job['processed']:raw=raw[:content_height(raw.shape[0])]
            small=down([raw])[0];h,w=small.shape[:2];scale=np.array([raw.shape[1]/w,raw.shape[0]/h])
            people=frame_people(fi)
            if not old and people and frame_luma(small)>=30:
                start=time.perf_counter();enhanced=enhancer([small])[0]
                xy,cf=poses(pose_session,enhanced)
                ps=associate_poses(np.array([p['box']for p in people]),xy,cf)
                frame_times.append(time.perf_counter()-start)
                for p,pose in zip(people,ps):p['pose']=pose
            for p in people:
                p['box']*=np.tile(scale,2)
                if p['pose'] is not None:
                    p['pose']=np.asarray(p['pose'],float);p['pose'][:,:2]*=scale
            for e in requested[fi]:
                person=next((p for p in people if p['id']==e['track_id'] and
                             p['generation']==e.get('history_generation',0)),None)
                key=e['event_index'];ts=fi/job['fps']
                if person is None:
                    samples[key].append(dict(frame=fi,timestamp=ts,binding_valid=False,
                        person_box=(np.array(e['bbox'])*np.tile(scale,2)).tolist(),pose=None,detections=[]))
                    continue
                box=person['box'];other=[p for p in people if p is not person]
                ambiguous=any(overlap(box,p['box'])>.4 for p in other)
                result=extract_objects(object_session,raw,box,person['pose'],[p['pose']for p in other],e['baseline'])
                samples[key].append(dict(frame=fi,timestamp=ts,binding_valid=not ambiguous,
                    person_box=box.tolist(),pose=None if person['pose'] is None else person['pose'].tolist(),**result))
    finally:cap.release()
    rows=[]
    for e in job['events']:
        seq=samples[e['event_index']]
        rows.append(dict(event_index=e['event_index'],track_id=e['track_id'],
            history_generation=e.get('history_generation',0),frame_index=e['frame_index'],timestamp=e['timestamp'],
            baseline=e['baseline'],baseline_conf=e['baseline_conf'],truth=e.get('truth'),samples=seq,
            **decide(e['baseline'],seq)))
    meta={k:v for k,v in job.items()if k!='events'}
    save(destination,dict(job=meta,records=rows,frontend_seconds=frame_times,
        evidence_schema='native source pixels; 30% body ROI plus pose-local interaction zoom; five sampled frames',
        binding='historical exact cached tracks' if old else 'nearest event bounds within .55s; >.4 actor overlap abstains; current shared pose one-to-one assignment',
        production_changed=False))
    print('[actions]',job['id'],len(rows),'veto',sum(r['conservative']!=r['baseline']for r in rows),
        'strict',sum(r['strict']!=r['baseline']for r in rows),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=['old','new'],required=True);args=parser.parse_args()
    import torch
    torch.set_num_threads(2);cv2.setNumThreads(1)
    lock=json.loads((OUT/'prior-lock.json').read_text())
    assert hashlib.sha256((OUT/'scripts/prior_rules.py').read_bytes()).hexdigest()==lock['rules_sha256']
    assert hashlib.sha256((OUT/'models/yolo26m_objects_candidate.onnx').read_bytes()).hexdigest()==lock['model_sha256']
    obj=session(OUT/'models/yolo26m_objects_candidate.onnx')
    pose=session(OP/'ckpts/rtmo-s_person_20261024.onnx') if args.kind=='new' else None
    enhancer=RetinexformerStage(str(OP/'ckpts'),chunk=1) if args.kind=='new' else None
    if enhancer:enhancer.load('cuda:0')
    plan=jobs(args.kind);save(OUT/f'action-plan-{args.kind}.json',plan)
    try:
        for j in plan:run(j,obj,pose,enhancer)
    finally:
        if enhancer:enhancer.close()
    save(OUT/f'action-complete-{args.kind}.json',dict(videos=len(plan),events=sum(len(j['events'])for j in plan),complete=True))


if __name__=='__main__':main()
