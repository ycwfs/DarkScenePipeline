"""Frozen n/s/m detection and input-resolution ablation; never trains on feedback."""
import argparse
from collections import defaultdict
from fractions import Fraction
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
BASE = ROOT/'reports/object-prior-finetune-20261025'
OP = ROOT/'platform/dark-demo-v2'
sys.path[:0] = [str(OP), str(BASE/'scripts')]
from darkpipe.person.object_prior import allows_object_prior, filter_object_evidence
from darkpipe.stages.downscale import DownscaleStage
from darkpipe.stages.enhance_retinexformer import RetinexformerStage
from darkpipe.stages.recognize import frame_luma
from darkpipe.person.runtime import content_height
from select_export import letterbox, iou
from evaluate_objects import counts

VARIANTS = ('native_raw', 'native_gamma', 'small_raw', 'small_enhanced')
MODELS = dict(n=ROOT.parent/'yolo26n-detect.pt', s=ROOT.parent/'yolo26s-detect.pt',
              m=ROOT.parent/'yolo26m-detect.pt',
              m_finetuned=BASE/'models/yolo26m_objects_candidate.pt')
MAPPING = {39: 0, 41: 1, 67: 2}  # common COCO classes only; no invented camera/paper class


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gamma(image: np.ndarray, value: float | None = None) -> np.ndarray:
    if value is None:
        lum = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean()
        value = 1. if lum >= 75 else .5 if lum >= 15 else .32
    return cv2.LUT(image, np.round((np.arange(256)/255.)**value*255).astype(np.uint8))


def prepare() -> None:
    torch.set_num_threads(2); cv2.setNumThreads(1)
    if (OUT/'inputs.json').exists():
        print('Prepared input manifest already exists; reusing frozen files.', flush=True)
        return
    save(OUT/'protocol.json', dict(models={k:dict(path=str(v),sha256=digest(v)) for k,v in MODELS.items()},
        variants=VARIANTS, confidence=.35, iou=.5, lum_gate=30., action_threshold=.85,
        detector='640 letterbox, FP32, end-to-end one-to-one head, batch8',
        object_truth='78 previously reviewed val/test ROIs; auxiliary visual labels',
        actions='all619 frozen candidate windows (55old,564new), five cached sample frames each',
        gate='raw window L<30 or unknown: skip object inference/filter; sample L<30 also skipped',
        limits=['No training, no model selection on feedback', 'COCO n/s/m lack camera/paper/exhibit classes',
                'New feedback has no complete action GT', 'Same-scene history, not independent new-scene test',
                'Cached actor binding, positive ROI support only; no phone/exhibit veto',
                'No streaming speed claim; separately processed source crops'],
        input_manifest_sha256=digest(BASE/'dataset-manifest.json'),
        policy_sha256=digest(OP/'darkpipe/person/object_prior.py')))
    enhancer = RetinexformerStage(str(OP/'ckpts'), chunk=1); enhancer.load('cuda:0')
    down = DownscaleStage(720)
    images, windows = {}, []
    def views(identity, full, rect, small, enhanced, metadata, gamma_value=None):
        x1,y1,x2,y2 = map(int, rect)
        native = full[y1:y2,x1:x2]
        assert native.size, (identity,rect)
        ratio = np.array([small.shape[1]/full.shape[1],small.shape[0]/full.shape[0]])
        lo = np.floor(np.array([x1,y1])*ratio).astype(int)
        hi = np.ceil(np.array([x2,y2])*ratio).astype(int)
        crops = dict(native_raw=native, native_gamma=gamma(native,gamma_value),
                     small_raw=small[lo[1]:hi[1],lo[0]:hi[0]],
                     small_enhanced=enhanced[lo[1]:hi[1],lo[0]:hi[0]])
        paths = {}
        for variant, image in crops.items():
            path=OUT/'inputs'/variant/(identity+'.png'); path.parent.mkdir(parents=True,exist_ok=True)
            assert cv2.imwrite(str(path),image,[cv2.IMWRITE_PNG_COMPRESSION,2])
            # Exact crop transform (including floor/ceil), used to recover full-source coordinates.
            scale=[1.,1.] if variant.startswith('native') else (1/ratio).tolist()
            origin=[x1,y1] if variant.startswith('native') else (lo/ratio).tolist()
            paths[variant]=dict(path=str(path.relative_to(OUT)),width=image.shape[1],height=image.shape[0],
                                origin=origin,scale=scale,sha256=digest(path))
        images[identity]=dict(id=identity,rect=list(map(int,rect)),variants=paths,**metadata)
    sources={r['id']:r for r in json.loads((BASE/'source-inventory.json').read_text())['records']}
    holdout=[r for r in json.loads((BASE/'dataset-manifest.json').read_text())['records'] if r['split'] in ('val','test')]
    for r in holdout:
        full=cv2.imread(str(Path(r['image_path']).parent/r['raw_file']));assert full is not None
        small=down([full])[0]; processed=sources[r['video_id']]['processed']
        enhanced=small if processed else enhancer([small])[0]
        views('h_'+r['id'],full,r['reviewed_crop_xyxy_raw'],small,enhanced,
              dict(kind='holdout',split=r['split'],group=r['group'],boxes=r['boxes'],
                   source=r['source'],processed=processed,raw_luma=None if processed else frame_luma(small)),
              r['brightness_gamma'])
    print('[prepare] holdout',len(holdout),flush=True)
    for file in sorted((BASE/'action-evaluation').glob('*.json')):
        data=json.loads(file.read_text());job=data['job']; requested=defaultdict(list)
        raw_cache=json.loads(Path(job['cache']).read_text())
        events=raw_cache if job['split']=='feedback' else raw_cache['events']['30']
        for record in data['records']:
            event=events[record['event_index']]
            row={k:record[k] for k in ('event_index','track_id','history_generation','timestamp','baseline','truth')}
            row.update(id=f"{job['id']}_{record['event_index']}",source_id=job['id'],split=job['split'],
                       group=job['group'],processed=job['processed'],raw_luma=event['raw_luma'],samples=[])
            windows.append(row)
            if not allows_object_prior(row['raw_luma']):
                continue
            for sample in record['samples']:
                new=dict(frame=sample['frame'],binding_valid=sample['binding_valid'],
                         person_roi=sample.get('person_roi'),views=[],raw_luma=None)
                row['samples'].append(new)
                if sample['binding_valid'] and sample.get('regions'):
                    requested[sample['frame']].append((sample,new))
        cap=cv2.VideoCapture(job['source']);assert cap.isOpened(),job['source']
        next_frame=0
        for fi,requests in sorted(requested.items()):
            if fi-next_frame>job['fps']*3:
                cap.set(cv2.CAP_PROP_POS_FRAMES,fi);next_frame=fi
            while next_frame<fi:
                assert cap.grab();next_frame+=1
            ok,full=cap.read();next_frame+=1;assert ok
            if job['processed']:full=full[:content_height(full.shape[0])]
            small=down([full])[0];lum=None if job['processed'] else frame_luma(small)
            for _,new in requests:new['raw_luma']=lum
            if not allows_object_prior(lum):continue
            enhanced=enhancer([small])[0]
            for sample,new in requests:
                for rect in sample['regions']:
                    key=f"a_{job['id']}_{fi}_"+'_'.join(str(int(v))for v in rect)
                    if key not in images:
                        views(key,full,rect,small,enhanced,dict(kind='action',split=job['split'],group=job['group'],
                            source=job['source'],processed=job['processed'],raw_luma=lum,frame=fi))
                    new['views'].append(key)
        cap.release()
        print('[prepare]',job['id'],'frames',len(requested),'views_total',len(images),flush=True)
    enhancer.close()
    save(OUT/'inputs.json',dict(images=images,windows=windows,protocol_sha256=digest(OUT/'protocol.json')))
    print('[prepared]',len(images),'views',len(windows),'windows',flush=True)


def infer(name: str) -> None:
    from ultralytics import YOLO
    torch.set_num_threads(2);cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=True
    manifest=json.loads((OUT/'inputs.json').read_text())
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert digest(MODELS[name])==protocol['models'][name]['sha256']
    model=YOLO(str(MODELS[name]));names=model.names
    net=model.model.float().eval().to('cuda:0');net.end2end=True
    net.fuse(verbose=False)
    tasks=[(key,variant,info)for key,r in manifest['images'].items()for variant,info in r['variants'].items()]
    output={};times=[]
    with torch.inference_mode():
        for _ in range(3):net(torch.zeros((8,3,640,640),device='cuda:0'))
        for start in range(0,len(tasks),8):
            batch=tasks[start:start+8];xs=[];transforms=[]
            for key,variant,info in batch:
                image=cv2.imread(str(OUT/info['path']));assert image is not None
                x,r,left,top=letterbox(image);xs.append(x);transforms.append((r,left,top,image.shape))
            tensor=torch.from_numpy(np.concatenate(xs)).to('cuda:0')
            torch.cuda.synchronize();begin=time.perf_counter();pred=net(tensor)
            if isinstance(pred,tuple):pred=pred[0]
            torch.cuda.synchronize();times.append((time.perf_counter()-begin)/len(batch))
            pred=pred.cpu().numpy();assert pred.shape[1:]==(300,6),pred.shape
            for (key,variant,info),rows,(r,left,top,shape) in zip(batch,pred,transforms):
                rows=rows[rows[:,4]>=.1].copy()
                rows[:,[0,2]]=(rows[:,[0,2]]-left)/r;rows[:,[1,3]]=(rows[:,[1,3]]-top)/r
                rows[:,:4]=np.clip(rows[:,:4],[0]*4,[shape[1],shape[0]]*2)
                output.setdefault(key,{})[variant]=rows.tolist()
            if start%1600==0:print('[infer]',name,start,'/',len(tasks),flush=True)
    save(OUT/'predictions'/f'{name}.json',dict(model=name,names=names,predictions=output,
        model_sha256=digest(MODELS[name]),inputs_sha256=digest(OUT/'inputs.json'),seconds_per_image=times,
        timing_note='GPU forward only, batch8 FP32, excludes decode/crop/transfers; not stream FPS'))
    print('[inferred]',name,len(tasks),flush=True)


def map_rows(rows: list, name: str) -> np.ndarray:
    arr=np.asarray(rows,dtype=float).reshape(-1,6)
    if name!='m_finetuned':
        arr=arr[np.isin(arr[:,5],list(MAPPING))]
        for row in arr:row[5]=MAPPING[int(row[5])]
    return arr


def analyze() -> None:
    inputs=json.loads((OUT/'inputs.json').read_text());results={};action_results=[];objects=[]
    for name in MODELS:
        data=json.loads((OUT/'predictions'/f'{name}.json').read_text())
        assert data['inputs_sha256']==digest(OUT/'inputs.json')
        for variant in VARIANTS:
            for key,image in inputs['images'].items():
                if image['kind']!='holdout':continue
                info=image['variants'][variant];pred=map_rows(data['predictions'][key][variant],name)
                pred[:,:4]*=np.tile(info['scale'],2)
                pred[:,:4]+=np.tile(np.array(info['origin'])-np.array(image['rect'][:2]),2)
                size=np.array(image['rect'][2:])-np.array(image['rect'][:2])
                pred[:,:4]=np.clip(pred[:,:4],[0]*4,np.tile(size,2))
                objects.append(dict(id=key,split=image['split'],group=image['group'],processed=image['processed'],
                    model=name,variant=variant,counts=counts(pred,image['boxes']),detections=pred.tolist()))
            for window in inputs['windows']:
                samples=[]
                for sample in window['samples']:
                    dets=[]
                    for key in sample['views']:
                        info=inputs['images'][key]['variants'][variant]
                        pred=map_rows(data['predictions'][key][variant],name)
                        pred[:,:4]*=np.tile(info['scale'],2);pred[:,:4]+=np.tile(info['origin'],2)
                        dets+=pred.tolist()
                    # Same-class cross-ROI duplicate suppression; no person/model score mixing.
                    kept=[]
                    for box in sorted(dets,key=lambda b:-b[4]):
                        if not any(int(box[5])==int(k[5]) and iou(np.array(box[:4]),np.array(k[:4]))>.5 for k in kept):
                            kept.append(box)
                    samples.append(dict(sample,detections=kept,available=bool(sample['views'])))
                decision=filter_object_evidence(window['baseline'],samples,window['raw_luma'])
                common=filter_object_evidence(window['baseline'],
                    [dict(s,detections=[d for d in s['detections'] if int(d[5]) in (0,1,2)])for s in samples],window['raw_luma'])
                action_results.append({k:v for k,v in window.items()if k!='samples'}|dict(model=name,variant=variant,
                    decision=decision,common_decision=common))
        times=data['seconds_per_image']
        results[name]=dict(images=sum(len(r) for r in data['predictions'].values()),forward_ms_median=float(np.median(times)*1000),
                           forward_ms_p95=float(np.quantile(times,.95)*1000))
    save(OUT/'object-results.json',objects);save(OUT/'action-results.json',action_results)
    save(OUT/'timing.json',results)
    print('[analysis]',len(objects),'object rows',len(action_results),'action rows',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','infer','analyze'])
    parser.add_argument('--model',choices=list(MODELS));args=parser.parse_args()
    if args.phase=='prepare':prepare()
    elif args.phase=='infer':infer(args.model)
    else:analyze()
