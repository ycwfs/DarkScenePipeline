"""Complete untruncated object evidence over the joint alpha/gate candidate union."""
from collections import defaultdict,Counter
import importlib.util
import os

os.environ.setdefault('YOLO_CONFIG_DIR','/tmp/darkpipe-yolo-20261026')
os.environ.setdefault('YOLO_AUTOINSTALL','false')
from common import *
import cv2
import torch

spec=importlib.util.spec_from_file_location('previous_evidence',PREV/'scripts/evidence.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
from darkpipe.person.core import associate_poses


def reachable(e,g):
    ps=[fuse_person(np.array(e['rgb_probs'],np.float32),
                   None if e['skeleton_probs'] is None else np.array(e['skeleton_probs'],np.float32),
                   [e['raw_luma']],alpha=a,threshold=.7,lum_gate=g)['fused_probs'] for a in [.5,.99]]
    return [c for c in ['drink','pick_up'] if max(p[BEHAVIORS.index(c)] for p in ps)>=.7]


def candidates(v):
    raw=json.loads((ROOT/v['cache']).read_text())
    replay_path=OUT/'replay'/f'{v["id"]}.json'
    replay=json.loads(replay_path.read_text()) if replay_path.exists() else None
    new=v['scope']=='new_multipeople'; records={};refs={}
    rows=raw if new else raw['events']['30']
    for g in GATES:
        events=replay['events'][str(g)] if replay else rows
        for i,e in enumerate(events):
            eid=e.get('id',f'{v["id"]}_{i}')
            for label in reachable(e,g):
                key=eid+'|'+label
                refs.setdefault(key,[]).append(g)
                if key not in records:
                    records[key]=dict(id=key,window_id=eid,label=label,raw_luma=e['raw_luma'],
                        source_frames=e['source_frames'],track_id=e['track_id'],
                        history_generation=e.get('history_generation',0),gate_for_binding=g)
    for k,r in records.items():r['reachable_gates']=refs[k]
    return raw,replay,list(records.values())


def run(v,net,enhancer,pose_session,fingerprint):
    target=OUT/'evidence'/f'{v["id"]}.json'
    if target.exists():
        assert json.loads(target.read_text())['protocol_sha256']==fingerprint
        return
    raw,replay,records=candidates(v);new=v['scope']=='new_multipeople'
    old=json.loads((PREV/'evidence'/f'{v["id"]}.json').read_text())
    previous={r['id']:r for r in old['records']};views={};requests=defaultdict(list)
    lowpath=LOW/'data'/v['id']/'result.json'
    meta=json.loads(lowpath.read_text()) if lowpath.exists() and not new else raw
    if new:
        frame_ids=np.array([e['frame_index'] for e in raw]);order=np.argsort(frame_ids,kind='stable');frame_ids=frame_ids[order]
    for r in records:
        r['samples']=[]
        if r['raw_luma'] is None or r['raw_luma']<min(r['reachable_gates']):continue
        prev=previous.get(r['window_id'])
        if new and prev is not None and prev['label']==r['label']:
            r['samples']=prev['samples'];r['reuse']='previous frozen ROI evidence'
            continue
        for fi in np.unique(np.array(r['source_frames'])[np.linspace(0,31,5).round().astype(int)]):
            requests[int(fi)].append(r)
    cap=cv2.VideoCapture(str(ROOT/v['source']));assert cap.isOpened()
    width,height=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    down=prior.DownscaleStage(720);next_frame=0;enhanced_count=0;reused_views=0
    try:
        for fi,items in sorted(requests.items()):
            while next_frame<fi:assert cap.grab();next_frame+=1
            ok,frame=cap.read();next_frame+=1;assert ok
            small=down([frame])[0];lum=prior.frame_luma(small)
            ratio=np.array([small.shape[1]/width,small.shape[0]/height]);enhanced=None;people=[]
            if new and lum>=15:
                lo=np.searchsorted(frame_ids,fi-v['fps']*.55,'left');hi=np.searchsorted(frame_ids,fi+v['fps']*.55,'right');nearest={}
                for idx in sorted(order[lo:hi]):
                    e=raw[int(idx)];key=(e['track_id'],e['history_generation'])
                    if key not in nearest or abs(e['frame_index']-fi)<abs(nearest[key]['frame_index']-fi):nearest[key]=e
                people=[dict(id=k,box=np.array(e['bbox'],float)) for k,e in nearest.items()]
                if people:
                    enhanced=enhancer([small])[0];enhanced_count+=1
                    xy,cf=prior.poses(pose_session,enhanced)
                    paired=associate_poses(np.array([p['box'] for p in people]),xy,cf)
                    for p,pose in zip(people,paired):
                        p['pose']=None if pose is None else np.array(pose,float)
                        if p['pose'] is not None:p['pose'][:,:2]/=ratio
                        p['box']/=np.tile(ratio,2)
            for r in items:
                s=dict(frame=fi,raw_luma=lum,binding_valid=False,available=False,person_roi=None,views=[],detections=[])
                r['samples'].append(s)
                if lum<min(r['reachable_gates']):continue
                if new:
                    p=next((p for p in people if p['id']==(r['track_id'],r['history_generation'])),None)
                    if p is not None:p['valid']=not any(prior.iou(p['box'],o['box'])>.4 for o in people if o is not p)
                else:
                    f=meta['frames'][fi].copy()
                    tracks=replay['tracks'][str(r['gate_for_binding'])][fi] if replay else raw['frames'][fi]['tracks']['30']
                    f['tracks']={'30':tracks};p=prior.actor(f,r['track_id'],1/ratio)
                if p is None:continue
                rects=prior.regions(p['box'],p['pose'],frame.shape,r['label'])
                s.update(binding_valid=p['valid'],person_roi=rects[0].tolist())
                if not p['valid']:continue
                for rect in rects:
                    key=f'{v["id"]}_{fi}_'+'_'.join(map(str,rect))
                    if key not in views:
                        if key in old['views']:
                            views[key]=old['views'][key];reused_views+=1
                        else:
                            if enhanced is None:enhanced=enhancer([small])[0];enhanced_count+=1
                            lo=np.floor(rect[:2]*ratio).astype(int);hi=np.ceil(rect[2:]*ratio).astype(int)
                            im=enhanced[lo[1]:hi[1],lo[0]:hi[0]].copy();assert im.size
                            views[key]=dict(frame=fi,rect=rect.tolist(),origin=(lo/ratio).tolist(),scale=(1/ratio).tolist(),pixels=im)
                    s['views'].append(key)
                s['available']=bool(s['views'])
    finally:cap.release()
    todo={k:v for k,v in views.items() if 'detections' not in v}
    prior.infer_views(net,todo)
    for r in records:
        if r.get('reuse'):continue
        for s in r['samples']:
            boxes=[b for k in s['views'] for b in views[k]['detections']];kept=[]
            for b in sorted(boxes,key=lambda b:-b[4]):
                if not any(int(b[5])==int(k[5]) and prior.iou(np.array(b[:4]),np.array(k[:4]))>.5 for k in kept):kept.append(b)
            s['detections']=kept
    save(target,dict(source_id=v['id'],records=records,views=views,protocol_sha256=fingerprint,
        stats=dict(candidates=len(records),fresh_crops=len(todo),reused_views=reused_views,
                   reused_windows=sum(bool(r.get('reuse')) for r in records),enhanced_frames=enhanced_count)))
    print(v['id'],len(records),'candidates',len(todo),'new crops',reused_views,'reused crops',flush=True)


def main():
    import argparse
    from ultralytics import YOLO
    import onnxruntime as ort
    p=argparse.ArgumentParser();p.add_argument('--shard',default='0/1');p.add_argument('--scope',choices=['new','old','all'],default='all');a=p.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(1)
    protocol=json.loads((OUT/'protocol.json').read_text());fingerprint=digest(OUT/'protocol.json')
    affected={r['id'] for r in protocol['affected']}
    sources=[s for s in protocol['sources'] if a.scope=='all' or (s['scope']=='new_multipeople')==(a.scope=='new')]
    i,n=map(int,a.shard.split('/'));sources=sources[i::n]
    for v in sources:
        if v['id'] in affected:assert (OUT/'replay'/f'{v["id"]}.json').exists()
    for m in protocol['models'].values():assert digest(ROOT/m['path'])==m['sha256']
    model=YOLO(str(ROOT/protocol['models']['object']['path']));net=model.model.float().eval().cuda();net.end2end=True;net.fuse(verbose=False)
    enhancer=prior.RetinexformerStage(str(OP/'ckpts'),chunk=1);enhancer.load('cuda:0')
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    pose=ort.InferenceSession(str(OP/'ckpts/rtmo-s_person_20261024.onnx'),sess_options=opts,providers=[('CUDAExecutionProvider',{'device_id':0})]);assert pose.get_providers()[0]=='CUDAExecutionProvider'
    try:
        for v in sources:run(v,net,enhancer,pose,fingerprint)
    finally:enhancer.close()
    save(OUT/f'evidence-complete-{a.scope}-{i}.json',dict(complete=True,sources=len(sources),protocol_sha256=fingerprint))


if __name__=='__main__':main()
