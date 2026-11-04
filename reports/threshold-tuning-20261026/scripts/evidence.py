"""Complete fixed-model evidence for threshold-only replay; never change old artifacts."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('YOLO_CONFIG_DIR','/tmp/darkpipe-yolo-20261026')
os.environ.setdefault('YOLO_AUTOINSTALL','false')
import cv2
import numpy as np
import torch

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]
OLD=ROOT/'reports/onsite-object-prior-20261026'
ABL=ROOT/'reports/object-model-ablation-20261026'
NEW=ROOT/'reports/manypeople-inference-20261025'
sys.path.insert(0,str(OLD/'scripts'))
from evaluate import (OP,CACHE,OBJECT,save,digest,baseline,actor,regions,letterbox,iou,
    DownscaleStage,RetinexformerStage,frame_luma,allows_object_prior,filter_object_evidence)
from darkpipe.constants import BEHAVIORS
from darkpipe.person.core import associate_poses


def prepare() -> None:
    if (OUT/'protocol.json').exists():raise FileExistsError('Frozen protocol already exists')
    old=json.loads((OLD/'protocol.json').read_text());rows=[];sources=[]
    reviewed=json.loads((ABL/'user-feedback-labels.json').read_text())
    binary={r['id']:r['is_drink'] for r in reviewed['labels']}
    for v in old['sources']:
        assert digest(ROOT/v['source'])==v['sha256'],v['source']
        data=json.loads((OLD/'per-video'/f'{v["id"]}.json').read_text())
        for e in data['records']:
            probs=e['baseline']['fused_probs'];idx=int(np.argmax(probs))
            rows.append(dict(id=e['id'],source_id=v['id'],scope=v['scope'],split=v['split'],group=v['group'],
                event_index=e['event_index'],track_id=e['track_id'],history_generation=0,
                timestamp=e['timestamp'],frame_index=e['frame_index'],source_frames=e['source_frames'],
                bbox=e['bbox'],raw_luma=e['raw_luma'],truth=e['truth'],is_drink=None,
                top_label=BEHAVIORS[idx],top_score=probs[idx],fused_probs=probs))
        sources.append(v)
    for v in json.loads((NEW/'input-manifest.json').read_text()):
        number=v['number'];sid=f'new-{number:02d}';path=NEW/'events'/f'{number:02d}_events.json'
        events=json.loads(path.read_text())
        assert digest(Path(v['absolute_path']))==v['sha256']
        for i,e in enumerate(events):
            probs=e['fused_probs'];idx=int(np.argmax(probs));key=f'{sid}_{i}'
            rows.append(dict(id=key,source_id=sid,scope='new_multipeople',split='feedback_calibration',group=sid,
                event_index=i,track_id=e['track_id'],history_generation=e['history_generation'],
                timestamp=e['timestamp'],frame_index=e['frame_index'],source_frames=e['source_frames'],
                bbox=e['bbox'],raw_luma=e['raw_luma'],truth=None,is_drink=binary.get(key),
                top_label=BEHAVIORS[idx],top_score=probs[idx],fused_probs=probs))
        sources.append(dict(id=sid,source=v['absolute_path'],scope='new_multipeople',split='feedback_calibration',
            fps=float(Fraction(v['fps'])),frames=v['frames'],seconds=float(v['duration']),sha256=v['sha256'],
            cache=str(path.relative_to(ROOT)),cache_sha256=digest(path),group=sid,processed=False))
    assert len(rows)==51834 and len(sources)==544
    candidates=[r for r in rows if r['top_label'] in ('drink','pick_up') and r['top_score']>=.70]
    save(OUT/'windows.json',rows)
    config=dict(action_threshold=.85,object_min_conf=.35,rgb_weight=.95,raw_luma_gate=30.,no_pose='soft')
    protocol=dict(baseline=config,fixed='All models, alpha.95, raw-luma gate30,720enhancement,30%crop,5samples,positive-single-frame rule unchanged.',
        grid=dict(action=[i/100 for i in range(70,100)],object=[i/100 for i in range(1,91)],
            refinement='Both thresholds +/-0.01 around coarse recommendation, step0.001; same frozen objective'),
        selection=dict(calibration='Old onsite val49 sources + new9 binary-reviewed drinking trigger set; old train/test excluded from selection.',
            constraints=['Old-val drink TP >= baseline8/10; pickup TP >= baseline1/4; all-positive correct >= baseline23/35.',
                'Old-val other FP <= baseline6/163; new known drink FP <= baseline49/526.',
                'No additional unreviewed new drinking triggers (baseline0); old online-val FP no worse than baseline.',
                'Action threshold must not alone reject the confirmed true drinking score0.944713056.'],
            objective='Among feasible settings minimize 0.5*(old-val other FP / old-val other N + new known drink FP /526); ties maximize old-val macro supported F1, positive correct, new true drink retained, then prefer proximity to baseline.',
            alternatives='Also show recall-oriented feasible setting (max old-val positive correct) and Pareto frontier. Do not pick by old test.',
            feedback_note='New9 are now calibration feedback at user request; not independent final accuracy. No model training.'),
        labels='Old reviewed person/time truth; new is_drink binary only527 previous0.85 drink triggers. Pickup/newly introduced windows stay unknown.',
        inference='Reuse fixed RGB/skeleton probabilities. Fresh object scores down to0.001 for all possible drink/pickup candidates at action>=0.70; preserve frozen ROIs for previous candidates.',
        binding='Old exact frame tracks; new existing frozen crops or same nearest-event<=0.55s actor + fresh same RTMO; IoU>.4 ambiguity abstains.',
        sources=sources,models=old['models'],candidates=dict(Counter(r['scope'] for r in candidates)),
        files={str(p.relative_to(ROOT)):digest(p) for p in [OUT/'windows.json',OLD/'protocol.json',
            ABL/'inputs.json',ABL/'user-feedback-labels.json',OP/'darkpipe/person/core.py',
            OP/'darkpipe/person/object_prior.py',Path(__file__)]})
    save(OUT/'protocol.json',protocol)
    print('FROZEN',len(rows),'windows',len(candidates),'object candidates',protocol['candidates'],flush=True)


@torch.inference_mode()
def infer_views(net, views: dict) -> None:
    tasks=list(views.values())
    for start in range(0,len(tasks),8):
        batch=tasks[start:start+8];xs=[];transforms=[]
        for v in batch:
            im=v.get('pixels')
            if im is None:
                path=ABL/v['path'];assert digest(path)==v['image_sha256'];im=cv2.imread(str(path));assert im is not None
            x,r,left,top=letterbox(im);xs.append(x);transforms.append((r,left,top,im.shape))
        pred=net(torch.from_numpy(np.concatenate(xs)).cuda())
        if isinstance(pred,tuple):pred=pred[0]
        pred=pred.float().cpu().numpy();assert pred.shape[1:]==(300,6)
        for v,boxes,(r,left,top,shape) in zip(batch,pred,transforms):
            boxes=boxes[boxes[:,4]>=.001].copy()
            boxes[:,[0,2]]=(boxes[:,[0,2]]-left)/r;boxes[:,[1,3]]=(boxes[:,[1,3]]-top)/r
            boxes[:,:4]=np.clip(boxes[:,:4],[0]*4,[shape[1],shape[0]]*2)
            boxes[:,:4]*=np.tile(v['scale'],2);boxes[:,:4]+=np.tile(v['origin'],2)
            v['detections']=boxes.tolist();v.pop('pixels',None)


def poses(session, frame: np.ndarray) -> tuple:
    h,w=frame.shape[:2];ratio=640/max(h,w);canvas=np.full((640,640,3),114,np.uint8)
    im=cv2.resize(frame,(int(w*ratio),int(h*ratio)));canvas[:im.shape[0],:im.shape[1]]=im
    boxes,points=session.run(None,{session.get_inputs()[0].name:canvas.transpose(2,0,1)[None].astype(np.float32)})
    boxes,points=boxes[0],points[0];xywh=boxes[:,:4].copy();xywh[:,2:]-=xywh[:,:2]
    ids=np.asarray(cv2.dnn.NMSBoxes(xywh.tolist(),boxes[:,4].tolist(),.45,.45),int).reshape(-1)
    return points[ids,:,:2]/ratio,points[ids,:,2]


def run(v: dict, rows: list, net, enhancer, pose_session, frozen: dict, fingerprint: str) -> None:
    target=OUT/'evidence'/f'{v["id"]}.json'
    if target.exists():assert json.loads(target.read_text())['protocol_sha256']==fingerprint;return
    selected=[r for r in rows if r['top_label'] in ('drink','pick_up') and r['top_score']>=.7]
    if not selected:
        save(target,dict(source_id=v['id'],records=[],views={},enhanced_frames=0,protocol_sha256=fingerprint));return
    data=json.loads((ROOT/v['cache']).read_text());assert digest(ROOT/v['cache'])==v['cache_sha256']
    is_new=v['scope']=='new_multipeople';previous={}
    if not is_new:
        previous={r['id']:r for r in json.loads((OLD/'per-video'/f'{v["id"]}.json').read_text())['records']}
    else:
        frame_ids=np.array([e['frame_index'] for e in data]);order=np.argsort(frame_ids,kind='stable');frame_ids=frame_ids[order]
    requests=defaultdict(list);records=[];views={}
    for e in selected:
        row=dict(id=e['id'],label=e['top_label'],raw_luma=e['raw_luma'],samples=[]);records.append(row)
        if not allows_object_prior(e['raw_luma']):continue
        # Exact frozen new-scene crops prevent actor-association drift in the baseline.
        if is_new and e['id'] in frozen['windows']:
            for s in frozen['windows'][e['id']]['samples']:
                sample=dict(s,available=bool(s['views']),detections=[])
                row['samples'].append(sample)
                for key in s['views']:
                    info=frozen['images'][key]['variants']['small_enhanced']
                    views.setdefault(key,dict(frame=s['frame'],origin=info['origin'],scale=info['scale'],
                        path=info['path'],image_sha256=info['sha256'],rect=frozen['images'][key]['rect']))
            continue
        old_samples={s['frame']:s for s in previous.get(e['id'],{}).get('samples',[])}
        for fi in np.unique(np.asarray(e['source_frames'])[np.linspace(0,31,5).round().astype(int)]):
            requests[int(fi)].append((e,row,old_samples.get(int(fi))))
    cap=cv2.VideoCapture(str(ROOT/v['source']));assert cap.isOpened();next_frame=0;enhanced_frames=0
    down=DownscaleStage(720)
    try:
        for fi,items in sorted(requests.items()):
            while next_frame<fi:assert cap.grab();next_frame+=1
            ok,raw=cap.read();next_frame+=1;assert ok
            small=down([raw])[0];lum=frame_luma(small);ratio=np.array([small.shape[1]/raw.shape[1],small.shape[0]/raw.shape[0]])
            enhanced=None;people=[]
            if is_new and allows_object_prior(lum):
                # Same tie order and time tolerance as the previous new-scene evaluator.
                lo=np.searchsorted(frame_ids,fi-v['fps']*.55,side='left');hi=np.searchsorted(frame_ids,fi+v['fps']*.55,side='right')
                nearest={}
                for idx in sorted(order[lo:hi]):
                    ev=data[int(idx)];key=(ev['track_id'],ev['history_generation'])
                    if key not in nearest or abs(ev['frame_index']-fi)<abs(nearest[key]['frame_index']-fi):nearest[key]=ev
                people=[dict(id=key,box=np.array(ev['bbox'],float)) for key,ev in nearest.items()]
                if people:
                    enhanced=enhancer([small])[0];enhanced_frames+=1
                    xy,cf=poses(pose_session,enhanced)
                    paired=associate_poses(np.array([p['box'] for p in people]),xy,cf)
                    for p,pose in zip(people,paired):
                        p['pose']=None if pose is None else np.array(pose,float)
                        if p['pose'] is not None:p['pose'][:,:2]/=ratio
                        p['box']/=np.tile(ratio,2)
            if not is_new:assert abs(lum-data['frames'][fi]['luma'])<1e-6
            for e,row,old_sample in items:
                sample=dict(frame=fi,raw_luma=lum,binding_valid=False,available=False,person_roi=None,views=[],detections=[])
                row['samples'].append(sample)
                if old_sample is not None:
                    sample.update(binding_valid=old_sample['binding_valid'],person_roi=old_sample['person_roi'])
                    rects=[np.array(r,int) for r in old_sample.get('regions',[])]
                else:
                    if is_new:
                        p=next((p for p in people if p['id']==(e['track_id'],e['history_generation'])),None)
                        if p is not None:p['valid']=not any(iou(p['box'],o['box'])>.4 for o in people if o is not p)
                    else:p=actor(data['frames'][fi],e['track_id'],1/ratio)
                    if p is None:continue
                    rects=regions(p['box'],p['pose'],raw.shape,e['top_label'])
                    sample.update(binding_valid=p['valid'],person_roi=rects[0].tolist())
                if not sample['binding_valid'] or not allows_object_prior(lum) or not rects:continue
                if enhanced is None:enhanced=enhancer([small])[0];enhanced_frames+=1
                for rect in rects:
                    key=f'{v["id"]}_{fi}_'+'_'.join(map(str,rect))
                    if key not in views:
                        lo=np.floor(rect[:2]*ratio).astype(int);hi=np.ceil(rect[2:]*ratio).astype(int)
                        im=enhanced[lo[1]:hi[1],lo[0]:hi[0]].copy();assert im.size
                        views[key]=dict(frame=fi,rect=rect.tolist(),origin=(lo/ratio).tolist(),scale=(1/ratio).tolist(),pixels=im)
                    sample['views'].append(key)
                sample['available']=bool(sample['views'])
    finally:cap.release()
    infer_views(net,views)
    for row in records:
        for s in row['samples']:
            boxes=[box for key in s['views'] for box in views[key]['detections']];kept=[]
            for box in sorted(boxes,key=lambda b:-b[4]):
                if not any(int(box[5])==int(k[5]) and iou(np.array(box[:4]),np.array(k[:4]))>.5 for k in kept):kept.append(box)
            s['detections']=kept
        row['at_baseline_object_threshold']=filter_object_evidence(row['label'],row['samples'],row['raw_luma'],.35)
        low=filter_object_evidence(row['label'],row['samples'],row['raw_luma'],.001)
        row['object_observed']=low['applied'];row['bypass_reason']=low['reason'] if not low['applied'] else None
        row['max_support']=max((b['confidence'] for b in low['evidence']),default=0.)
    save(target,dict(source_id=v['id'],records=records,views=views,enhanced_frames=enhanced_frames,protocol_sha256=fingerprint))
    print('[evidence]',v['id'],len(selected),'windows',len(views),'crops',flush=True)


def infer(shard: str) -> None:
    from ultralytics import YOLO
    import onnxruntime as ort
    torch.set_num_threads(2);cv2.setNumThreads(1)
    protocol=json.loads((OUT/'protocol.json').read_text());fingerprint=digest(OUT/'protocol.json')
    for spec in protocol['models'].values():assert digest(ROOT/spec['path'])==spec['sha256']
    rows=defaultdict(list)
    for r in json.loads((OUT/'windows.json').read_text()):rows[r['source_id']].append(r)
    frozen=json.loads((ABL/'inputs.json').read_text());frozen['windows']={r['id']:r for r in frozen['windows']}
    model=YOLO(str(OBJECT));net=model.model.float().eval().cuda();net.end2end=True;net.fuse(verbose=False)
    enhancer=RetinexformerStage(str(OP/'ckpts'),chunk=1);enhancer.load('cuda:0')
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    pose_session=ort.InferenceSession(str(OP/'ckpts/rtmo-s_person_20261024.onnx'),sess_options=opts,
        providers=[('CUDAExecutionProvider',{'device_id':0})]);assert pose_session.get_providers()[0]=='CUDAExecutionProvider'
    idx,n=map(int,shard.split('/'))
    try:
        for v in protocol['sources'][idx::n]:run(v,rows[v['id']],net,enhancer,pose_session,frozen,fingerprint)
    finally:enhancer.close()
    save(OUT/f'complete-{idx}.json',dict(complete=True,protocol_sha256=fingerprint))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','infer']);parser.add_argument('--shard',default='0/1')
    args=parser.parse_args()
    if args.phase=='prepare':prepare()
    else:infer(args.shard)
