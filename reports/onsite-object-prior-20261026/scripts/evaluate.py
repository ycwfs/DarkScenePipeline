"""Frozen action-window replay + fresh 720-enhanced object inference.

No training, threshold selection, production integration, or real-time benchmark.
All source videos stay in the inventory, including those with no action windows.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('YOLO_CONFIG_DIR', '/tmp/darkpipe-yolo-20261026')
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
import cv2
import numpy as np
import torch

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
OP = ROOT/'platform/dark-demo-v2'
CACHE = ROOT/'reports/person-release-20261024'
OBJECT = ROOT/'reports/object-prior-finetune-20261025/models/yolo26m_objects_candidate.pt'
sys.path.insert(0, str(OP))
from darkpipe.person.core import fuse_person, iou
from darkpipe.person.object_prior import allows_object_prior, filter_object_evidence
from darkpipe.stages.downscale import DownscaleStage
from darkpipe.stages.enhance_retinexformer import RetinexformerStage
from darkpipe.stages.recognize import frame_luma


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':'))+'\n')
    temp.replace(path)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def baseline(row: dict) -> dict:
    return fuse_person(np.asarray(row['rgb_probs'], np.float32),
        None if row['skeleton_probs'] is None else np.asarray(row['skeleton_probs'], np.float32),
        [row['raw_luma']], alpha=.95, threshold=.85, lum_gate=30., no_pose='soft')


def window_key(row: dict) -> tuple:
    return row['track_id'], row['frame_index'], row['crop_sha256']


def freeze() -> None:
    if (OUT/'protocol.json').exists():
        raise FileExistsError('Protocol is frozen; run infer/analyze without replacing it.')
    old = json.loads((CACHE/'protocol.json').read_text())
    deployment = json.loads((OP/'models.json').read_text())
    models = {}
    for name, spec in deployment['models'].items():
        path = OP/'ckpts'/spec['file']
        assert digest(path) == spec['sha256'] == old['models'][name]['sha256'], name
        models[name] = dict(path=str(path.relative_to(ROOT)), sha256=spec['sha256'])
    models['object'] = dict(path=str(OBJECT.relative_to(ROOT)), sha256=digest(OBJECT))
    assert models['object']['sha256'] == 'b2d4bfc6d236626a81394b015bb848597c7aca397e09155145629a945c32422e'
    windows = json.loads((CACHE/'evaluated-windows.json').read_text())
    by_video = defaultdict(list)
    for row in windows:
        by_video[row['video_id']].append(row)
    sources = []
    for v in old['videos']:
        path = CACHE/'data'/v['id']/'result.json'
        data = json.loads(path.read_text())
        assert data['protocol_sha256'] == digest(CACHE/'protocol.json')
        assert digest(ROOT/v['source']) == v['sha256'], v['source']
        lookup = {window_key(r): r for r in data['events']['30']}
        assert len(lookup) == len(by_video[v['id']]) == len(data['events']['30'])
        for row in by_video[v['id']]:
            original = lookup[window_key(row)]
            for k in ('rgb_probs', 'skeleton_probs', 'source_frames', 'raw_luma'):
                assert original[k] == row[k], (v['id'], k)
        counts = Counter(len(f['boxes']) for f in data['frames'])
        sources.append(dict(v, cache=str(path.relative_to(ROOT)), cache_sha256=digest(path),
            scope='online_supplement' if v['processed'] else 'onsite', windows=len(lookup),
            box_frames=sum(bool(f['boxes']) for f in data['frames']),
            people_mode=counts.most_common(1)[0][0], people_max=max(counts)))
    protocol = dict(version=1, action_threshold=.85, rgb_weight=.95, raw_luma_gate=30.,
        no_pose='soft', object_min_conf=.35, proc_max_side=720, models=models,
        action_cache='Frozen original RGB/ST-GCN probabilities; production fuse_person recomputed. No new tracking/action forward.',
        fresh_inference='Source decoded forward without seeking; Retinexformer FP16 full720 then same-person crops; YOLO26m FP32 batch8 one-to-one head640.',
        sampling='Five unique evenly-spaced positions in the existing32-frame window; one compatible valid bright detection is enough.',
        regions='30% person bbox margin plus pose-local interaction region if reliable; exact original/720 floor-ceil transform.',
        binding='Exact cached per-frame track; bboxIoU>.4 with another track abstains; object center in actor expanded ROI.',
        classes=['bottle', 'cup', 'phone', 'camera', 'paper', 'portable_object', 'robot_or_exhibit'],
        compatible=dict(drink=[0,1], pick_up=[0,1,2,3,4,5]),
        unaffected='Other action classes unchanged. Unknown/below30 window luma or no usable bright samples abstain.',
        comparison='Without object prior vs fixed chosen object prior on identical windows; count false positives removed AND new false negatives.',
        labels='Existing reviewed person/time labels, including prior review overlay; uncertain labels stay null.',
        limitations=['Training-fit and validation/test reported separately; historical initialization saw some holdout sources.',
            'Label transfer may leave detected actors unknown; absent detection/window is NOT counted correct.',
            'Window metrics are not true-action episode recall or saved-clip counts.',
            'Cached-window candidate experiment, not a production streaming/latency validation.',
            'Object presence does not prove interaction; five samples can miss short or occluded objects.'],
        files={str(p.relative_to(ROOT)): digest(p) for p in (
            CACHE/'protocol.json', CACHE/'evaluated-windows.json', OP/'models.json',
            OP/'darkpipe/person/core.py', OP/'darkpipe/person/object_prior.py',
            ROOT/'reports/person-finetune-20261023/annotations.json',
            ROOT/'reports/person-finetune-20261023/splits.json', Path(__file__))}, sources=sources)
    save(OUT/'protocol.json', protocol)
    print('FROZEN', len(sources), 'sources', Counter(v['scope'] for v in sources),
          'windows', len(windows), flush=True)


def letterbox(image: np.ndarray) -> tuple:
    h, w = image.shape[:2]
    ratio = min(640/w, 640/h)
    nw, nh = round(w*ratio), round(h*ratio)
    left, top = (640-nw)//2, (640-nh)//2
    canvas = np.full((640,640,3),114,np.uint8)
    canvas[top:top+nh,left:left+nw] = cv2.resize(image,(nw,nh))
    x = np.ascontiguousarray(canvas[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255
    return x, ratio, left, top


def regions(box: np.ndarray, pose: np.ndarray | None, shape: tuple, action: str) -> list:
    margin = (box[2:]-box[:2])*.3
    rect = np.r_[np.floor(np.maximum(0,box[:2]-margin)),
        np.ceil(np.minimum([shape[1],shape[0]],box[2:]+margin))].astype(int)
    output = [rect]
    if pose is not None:
        indices = [0,1,2,3,4,9,10] if action == 'drink' else [9,10,13,14]
        points = pose[indices]; points = points[points[:,2]>=.5,:2]
        if len(points)>=2:
            padding = max(24., (box[3]-box[1])*.12)
            lo = np.floor(np.maximum(rect[:2],points.min(0)-padding)).astype(int)
            hi = np.ceil(np.minimum(rect[2:],points.max(0)+padding)).astype(int)
            if np.all(hi-lo>=24) and np.prod(hi-lo)<np.prod(rect[2:]-rect[:2])*.8:
                output.append(np.r_[lo,hi])
    return output


def actor(meta: dict, tid: int, scale: np.ndarray) -> dict | None:
    tracks = meta['tracks']['30']
    t = next((t for t in tracks if t['track_id']==tid), None)
    if t is None:
        return None
    box = np.asarray(t['bbox'],float)
    ambiguous = any(iou(box,np.asarray(o['bbox']))>.4 for o in tracks if o is not t)
    matches = [(iou(box,np.asarray(b[:4])),i) for i,b in enumerate(meta['boxes'])]
    pose = meta['poses'][max(matches)[1]] if matches and max(matches)[0]>.5 else None
    if pose is not None:
        pose = np.array(pose,float); pose[:,:2] *= scale
    return dict(box=box*np.tile(scale,2),pose=pose,valid=not ambiguous)


@torch.inference_mode()
def infer_views(net, views: dict) -> None:
    tasks = list(views.values())
    for start in range(0,len(tasks),8):
        batch = tasks[start:start+8]; xs=[]; transforms=[]
        for view in batch:
            x,r,left,top = letterbox(view['pixels'])
            xs.append(x); transforms.append((r,left,top,view['pixels'].shape))
        prediction = net(torch.from_numpy(np.concatenate(xs)).to('cuda:0'))
        if isinstance(prediction,tuple):prediction=prediction[0]
        prediction = prediction.float().cpu().numpy()
        assert prediction.shape[1:] == (300,6), prediction.shape
        for view,rows,(r,left,top,shape) in zip(batch,prediction,transforms):
            rows = rows[rows[:,4]>=.1].copy()
            rows[:,[0,2]]=(rows[:,[0,2]]-left)/r
            rows[:,[1,3]]=(rows[:,[1,3]]-top)/r
            rows[:,:4]=np.clip(rows[:,:4],[0]*4,[shape[1],shape[0]]*2)
            rows[:,:4]*=np.tile(view['scale'],2)
            rows[:,:4]+=np.tile(view['origin'],2)
            view['detections']=rows.tolist()
            del view['pixels']


def run_video(v: dict, windows: list, net, enhancer, fingerprint: str) -> None:
    destination=OUT/'per-video'/f'{v["id"]}.json'
    if destination.exists():
        assert json.loads(destination.read_text())['protocol_sha256']==fingerprint
        return
    start=time.monotonic()
    data=json.loads((ROOT/v['cache']).read_text())
    assert digest(ROOT/v['cache'])==v['cache_sha256']
    requests=defaultdict(list); records=[]
    keys={window_key(r):i for i,r in enumerate(data['events']['30'])}
    for e in windows:
        row={k:e[k] for k in ('track_id','frame_index','timestamp','crop_sha256','source_frames',
            'bbox','truth','truth_provenance','people','raw_luma','pose_valid')}
        row.update(id=f'{v["id"]}_{keys[window_key(e)]}',event_index=keys[window_key(e)],
            baseline=baseline(e),samples=[])
        records.append(row)
        if row['baseline']['label'] in ('drink','pick_up') and allows_object_prior(e['raw_luma']):
            assert not v['processed'], 'Processed source must have unknown raw brightness'
            for fi in np.unique(np.asarray(e['source_frames'])[np.linspace(0,31,5).round().astype(int)]):
                requests[int(fi)].append(row)
    down=DownscaleStage(720); views={}; enhanced_count=0; read_frames=0
    if requests:
        cap=cv2.VideoCapture(str(ROOT/v['source']));assert cap.isOpened()
        try:
            for fi,rows in sorted(requests.items()):
                while read_frames<fi:
                    assert cap.grab(), (v['source'],read_frames)
                    read_frames+=1
                ok,raw=cap.read();read_frames+=1;assert ok, (v['source'],fi)
                small=down([raw])[0];lum=frame_luma(small)
                assert abs(lum-data['frames'][fi]['luma'])<1e-6, (v['id'],fi,lum)
                ratio=np.array([small.shape[1]/raw.shape[1],small.shape[0]/raw.shape[0]])
                enhanced=None
                for row in rows:
                    p=actor(data['frames'][fi],row['track_id'],1/ratio)
                    sample=dict(frame=fi,timestamp=fi/v['fps'],raw_luma=lum,binding_valid=False,
                        available=False,person_roi=None,views=[],detections=[])
                    row['samples'].append(sample)
                    if p is None:continue
                    rects=regions(p['box'],p['pose'],raw.shape,row['baseline']['label'])
                    sample.update(binding_valid=p['valid'],person_roi=rects[0].tolist(),
                        person_box=p['box'].tolist(),regions=[r.tolist() for r in rects])
                    if not p['valid'] or not allows_object_prior(lum):continue
                    if enhanced is None:
                        enhanced=enhancer([small])[0];enhanced_count+=1
                    for rect in rects:
                        key=f'{fi}_'+ '_'.join(map(str,rect))
                        if key not in views:
                            lo=np.floor(rect[:2]*ratio).astype(int)
                            hi=np.ceil(rect[2:]*ratio).astype(int)
                            image=enhanced[lo[1]:hi[1],lo[0]:hi[0]].copy()
                            assert image.size, (v['id'],fi,rect)
                            views[key]=dict(frame=fi,rect=rect.tolist(),origin=(lo/ratio).tolist(),
                                scale=(1/ratio).tolist(),shape=list(image.shape),pixels=image,
                                pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest())
                        sample['views'].append(key)
                    sample['available']=bool(sample['views'])
        finally:cap.release()
        infer_views(net,views)
    for row in records:
        for sample in row['samples']:
            kept=[]
            boxes=[b for key in sample['views'] for b in views[key]['detections']]
            for box in sorted(boxes,key=lambda b:-b[4]):
                if not any(int(box[5])==int(k[5]) and iou(np.asarray(box[:4]),np.asarray(k[:4]))>.5 for k in kept):
                    kept.append(box)
            sample['detections']=kept
        row['prior']=filter_object_evidence(row['baseline']['label'],row['samples'],row['raw_luma'])
        row['after']=row['prior']['label']
    save(destination,dict(source=v,protocol_sha256=fingerprint,records=records,views=views,
        fresh_enhanced_frames=enhanced_count,decoded_through_frame=read_frames-1,
        seconds=time.monotonic()-start,complete=True))
    print('[video]',v['source'],'windows',len(records),'views',len(views),
        'filtered',sum(r['after']!=r['baseline']['label'] for r in records),flush=True)


def infer(shard: str) -> None:
    from ultralytics import YOLO
    torch.set_num_threads(2);cv2.setNumThreads(1)
    protocol=json.loads((OUT/'protocol.json').read_text()); fingerprint=digest(OUT/'protocol.json')
    for name,spec in protocol['models'].items():assert digest(ROOT/spec['path'])==spec['sha256'],name
    windows=defaultdict(list)
    assert digest(CACHE/'evaluated-windows.json')==protocol['files'][str((CACHE/'evaluated-windows.json').relative_to(ROOT))]
    for e in json.loads((CACHE/'evaluated-windows.json').read_text()):windows[e['video_id']].append(e)
    model=YOLO(str(OBJECT));assert list(model.names.values())==protocol['classes'],model.names
    net=model.model.float().eval().to('cuda:0');net.end2end=True;net.fuse(verbose=False)
    enhancer=RetinexformerStage(str(OP/'ckpts'),chunk=1);enhancer.load('cuda:0')
    index,total=map(int,shard.split('/')); assert 0<=index<total
    sources=protocol['sources'][index::total]
    try:
        for v in sources:run_video(v,windows[v['id']],net,enhancer,fingerprint)
    finally:enhancer.close()
    save(OUT/f'complete-{index}.json',dict(protocol_sha256=fingerprint,sources=len(sources),complete=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','infer'])
    parser.add_argument('--shard',default='0/1');args=parser.parse_args()
    if args.phase=='freeze':freeze()
    else:infer(args.shard)
