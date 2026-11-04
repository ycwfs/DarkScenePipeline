"""Offline multi-view person/COCO17 drafts, never ground truth or training-ready labels."""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort
import torch
from rtmlib import RTMO, RTMPose
from scipy.optimize import linear_sum_assignment

from scan import OUT, ROOT, downscale, save, digest

OP = ROOT/'platform/dark-demo-v2'
sys.path.insert(0, str(OP))
from darkpipe.stages.enhance_retinexformer import RetinexformerStage
from darkpipe.stages.object_prior import YoloOnnxDetector

MODELS = {
    'yolo26m': OP/'ckpts/yolo26m_detect.onnx',
    'rtmo_s': OP/'ckpts/rtmo-s_end2end.onnx',
    'rtmpose_s': Path('/ssd1/wfs/.cache/rtmlib/hub/checkpoints/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.onnx'),
    'retinexformer': OP/'ckpts/NTIRE.pth',
}
EDGES = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
JOINTS = ['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder',
          'left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee',
          'right_knee','left_ankle','right_ankle']
PALETTE = [(100,235,60),(255,200,40),(200,120,255),(60,150,255),(230,220,70),(150,70,245)]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.maximum(0, np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    union = np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-inter
    return float(inter/max(union,1e-6))


def same_box(a: dict, b: dict) -> bool:
    if iou(a['box'],b['box']) >= .48:
        return True
    lo=np.maximum(a['box'][:2],b['box'][:2]);hi=np.minimum(a['box'][2:],b['box'][2:])
    areas=[np.maximum(0,x['box'][2:]-x['box'][:2]).prod() for x in (a,b)]
    containment=np.maximum(0,hi-lo).prod()/max(min(areas),1)
    # Manual boxes include outstretched arms. Merge contained torso-only proposals.
    manual=any('visual' in x['source'] or 'user_confirmed' in x['source'] for x in (a,b))
    return bool(containment>.85 and (manual or min(areas)/max(areas)>.3))


def gamma(image: np.ndarray) -> np.ndarray:
    # Display transformation, no generated content or geometry changes.
    lut = np.rint((np.arange(256)/255.)**.45*255).astype(np.uint8)
    return cv2.LUT(image,lut)


def session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    result = ort.InferenceSession(str(path), sess_options=options,
        providers=[('CUDAExecutionProvider', {'device_id':0})])
    if result.get_providers()[0] != 'CUDAExecutionProvider':
        raise RuntimeError('CUDA unavailable; refusing silent CPU fallback')
    return result


def pose_tool(cls: type, path: Path, shape: tuple, mean=None, std=None):
    # Construct using the installed rtmlib preprocessing, but with bounded ORT thread pools.
    obj = cls.__new__(cls)
    obj.session=session(path)
    obj.model_input_size=shape
    obj.mean=mean
    obj.std=std
    obj.backend='onnxruntime'
    obj.to_openpose=False
    obj.nms_thr=.45
    obj.score_thr=.45
    return obj


class Proposer:
    def __init__(self):
        self.det = YoloOnnxDetector(str(OP/'ckpts'),'cuda:0')
        self.det.sess = session(MODELS['yolo26m'])
        self.rtmo = pose_tool(RTMO, MODELS['rtmo_s'], (640,640))
        self.topdown = pose_tool(RTMPose, MODELS['rtmpose_s'], (192,256),
            (123.675,116.28,103.53),(58.395,57.12,57.375))
        self.enhancer=RetinexformerStage(str(OP/'ckpts'),chunk=1)
        self.enhancer.load('cuda:0')

    def candidates(self, views: dict, old: dict) -> list:
        rows = []
        for name, image in views.items():
            for x1,y1,x2,y2,score,cls in self.det._frames_to_dets([image])[0]:
                if cls == 0 and score >= .25 and x2-x1 > 8 and y2-y1 > 20:
                    rows.append(dict(box=np.array([x1,y1,x2,y2]), score=float(score),
                                     source='yolo26m/'+name, pose=None))
            prepared, ratio = self.rtmo.preprocess(image)
            det, pose = self.rtmo.inference(prepared)
            det, pose = det[0], pose[0].copy()
            boxes=det[:,:4]/ratio
            xywh=boxes.copy()
            xywh[:,2:]-=xywh[:,:2]
            keep=cv2.dnn.NMSBoxes(xywh.tolist(),det[:,4].tolist(),.45,.45)
            for idx in np.asarray(keep,dtype=int).reshape(-1):
                box=boxes[idx].copy()
                box=np.clip(box,[0,0,0,0],[image.shape[1],image.shape[0]]*2)
                if box[2]-box[0] <= 8 or box[3]-box[1] <= 20:
                    continue
                points=pose[idx].copy()
                points[:,:2]/=ratio
                rows.append(dict(box=box,score=float(det[idx,4]),source='rtmo_s/'+name,pose=points))
        preview=np.array(next(iter(views.values())).shape[1::-1])
        for prior in old['previous_boxes'] or []:
            box=np.array(prior['bbox'])*np.tile(preview/np.array(old['previous_work_size']),2)
            rows.append(dict(box=box,score=None,source='previous_yolo26m/retinexformer720',pose=None))
        return rows

    def annotate(self, raw: np.ndarray, old: dict, video: dict, anchors: list) -> tuple:
        bright=gamma(raw)
        enhanced=self.enhancer([downscale(raw,720)])[0]
        enhanced=cv2.resize(enhanced,raw.shape[1::-1])
        candidates=self.candidates(dict(raw=raw,gamma=bright,retinexformer=enhanced),old)
        scale=np.array(raw.shape[1::-1])/np.array([video['width'],video['height']])
        refs=[r for r in anchors if r['bbox_xyxy'] is not None]
        exact=next((r for r in refs if r['frame_index_zero_based']==old['frame']),None)
        manual=None
        if exact:
            manual=dict(box=np.array(exact['bbox_xyxy'])*np.tile(scale,2),score=None,
                source='user_confirmed_box' if (Path(video['source']).stem,old['frame']) in
                {('捡东西1',32),('挥手18',30)} else 'assistant_visual_box',pose=None)
        elif refs and refs[0]['frame_index_zero_based'] < old['frame'] < refs[-1]['frame_index_zero_based']:
            left=max((r for r in refs if r['frame_index_zero_based']<old['frame']),key=lambda r:r['frame_index_zero_based'])
            right=min((r for r in refs if r['frame_index_zero_based']>old['frame']),key=lambda r:r['frame_index_zero_based'])
            t=(old['frame']-left['frame_index_zero_based'])/(right['frame_index_zero_based']-left['frame_index_zero_based'])
            box=(1-t)*np.array(left['bbox_xyxy'])+t*np.array(right['bbox_xyxy'])
            manual=dict(box=box*np.tile(scale,2),score=None,source='interpolated_visual_anchors_unreviewed',pose=None)
        candidates.sort(key=lambda r: r['score'] or .3,reverse=True)
        if manual:
            candidates.insert(0,manual)
        groups=[]
        for candidate in candidates:
            group=next((g for g in groups if same_box(g[0],candidate)),None)
            if group is None:
                groups.append([candidate])
            else:
                group.append(candidate)
        people=[]
        for group in groups:
            first=group[0]
            box=first['box']
            pose_candidates=[(r['source'],r['pose']) for r in group if r['pose'] is not None]
            for name,image in [('gamma',bright),('retinexformer',enhanced)]:
                xy,cf=self.topdown(image,bboxes=[box.tolist()])
                pose_candidates.append(('rtmpose_s/'+name,np.column_stack([xy[0],cf[0]])))
            # Confidence ranks draft alternatives, never COCO ground-truth visibility.
            def quality(item):
                xy=item[1][:,:2]; sc=item[1][:,2]
                inside=((xy>=box[:2]-8)&(xy<=box[2:]+8)).all(1)
                return float(np.minimum(sc,.8)[inside].sum())
            pose_candidates.sort(key=quality,reverse=True)
            pose_source, points=pose_candidates[0]
            points=points.copy()
            inside=((points[:,:2]>=0)&(points[:,:2]<np.array(raw.shape[1::-1]))).all(1)
            usable=(points[:,2]>=.3)&inside
            disagree=[]
            for _,alternative in pose_candidates[1:]:
                common=usable&(alternative[:,2]>=.3)
                if common.sum()>=4:
                    disagree.append(float(np.linalg.norm(points[common,:2]-alternative[common,:2],axis=1).mean()/max(np.linalg.norm(box[2:]-box[:2]),1)))
            risks=[]
            if len({r['source'].split('/')[0] for r in group})<2:
                risks.append('single_model_box')
            if usable.sum()<10:
                risks.append('sparse_pose')
            if max(disagree,default=0)>.15:
                risks.append('pose_disagreement')
            if 'interpolated' in first['source']:
                risks.append('interpolated_box')
            if old['luma']<15:
                risks.append('very_dark')
            if any(iou(first['box'],r['box'])<.48 for r in group[1:]):
                risks.append('nested_box_merge_check')
            native_points=points.copy();native_points[:,:2]/=scale
            people.append(dict(person_id=None,bbox_xyxy=(box/np.tile(scale,2)).round(3).tolist(),
                bbox_source=first['source'],bbox_score=first['score'],
                box_review='user_confirmed' if first['source']=='user_confirmed_box' else 'pending',
                keypoints=native_points.round(4).tolist(),keypoint_visibility=[None]*17,
                keypoint_usable=usable.tolist(),pose_source=pose_source,pose_review='pending',
                proposal_support=[r['source'] for r in group],risks=risks,
                alternative_boxes=[dict(source=r['source'],bbox_xyxy=(r['box']/np.tile(scale,2)).round(3).tolist()) for r in group[1:]],
                max_pose_disagreement=max(disagree,default=0),include_in_training=False))
        return people,enhanced


def assign_ids(people: list, history: list, next_id: int, index: int) -> tuple:
    history=[r for r in history if index-r['last_frame']<=8]
    cost=np.ones((len(history),len(people)))
    for i,old in enumerate(history):
        for j,new in enumerate(people):
            cost[i,j]=1-iou(np.array(old['box']),np.array(new['bbox_xyxy']))
    matched=set()
    if cost.size:
        rr,cc=linear_sum_assignment(cost)
        for i,j in zip(rr,cc):
            if cost[i,j]<.8:
                people[j]['person_id']=history[i]['person_id']; matched.add(i)
    for person in people:
        if person['person_id'] is None:
            person['person_id']=f'P{next_id:03d}';next_id+=1
    history=[r for i,r in enumerate(history) if i not in matched]
    history.extend(dict(box=p['bbox_xyxy'],person_id=p['person_id'],last_frame=index) for p in people)
    return history,next_id


def draw(image: np.ndarray, record: dict, video: dict) -> np.ndarray:
    out=image.copy();scale=np.array(image.shape[1::-1])/np.array([video['width'],video['height']])
    for p in record['people']:
        color=PALETTE[(int(p['person_id'][1:])-1)%len(PALETTE)]
        x1,y1,x2,y2=np.rint(np.array(p['bbox_xyxy'])*np.tile(scale,2)).astype(int)
        cv2.rectangle(out,(x1,y1),(x2,y2),color,2)
        cv2.putText(out,p['person_id']+' ?',(x1,max(20,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
        points=np.array(p['keypoints']);points[:,:2]*=scale
        valid=p['keypoint_usable']
        for a,b in EDGES:
            if valid[a] and valid[b]:
                cv2.line(out,tuple(np.rint(points[a,:2]).astype(int)),tuple(np.rint(points[b,:2]).astype(int)),color,2,cv2.LINE_AA)
        for j,point in enumerate(points):
            if valid[j]:
                cv2.circle(out,tuple(np.rint(point[:2]).astype(int)),3,color,-1,cv2.LINE_AA)
    cv2.rectangle(out,(0,0),(out.shape[1],27),(18,20,22),-1)
    text=f'f{record["frame"]:06d} | {record["timestamp"]:.2f}s | L={record["luma"]:.2f} | DRAFT / REVIEW REQUIRED'
    cv2.putText(out,text,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.47,(235,235,235),1,cv2.LINE_AA)
    if not record['people']:
        cv2.putText(out,'NO RELIABLE PROPOSAL - NOT A NEGATIVE LABEL',(10,out.shape[0]-18),cv2.FONT_HERSHEY_SIMPLEX,.5,(40,180,255),1)
    return out


def process(scan: dict, proposer: Proposer, anchors: list, sample=False) -> dict:
    video=scan['video'];destination=OUT/'data'/video['id'];path=destination/('smoke.json' if sample else 'annotations.json')
    if path.exists():
        return json.loads(path.read_text())
    frames=[f for f in scan['frames'] if f['eligible']]
    if sample:
        wanted=set(np.linspace(0,len(frames)-1,min(len(frames),4),dtype=int))
        frames=[f for i,f in enumerate(frames) if i in wanted or f['frame'] in {r['frame_index_zero_based'] for r in anchors}]
    history=[];next_id=1;records=[];started=time.monotonic()
    for frame in frames:
        raw=cv2.imread(str(OUT/frame['image']))
        people,enhanced=proposer.annotate(raw,frame,video,anchors)
        history,next_id=assign_ids(people,history,next_id,frame['frame'])
        record={k:v for k,v in frame.items() if k not in {'previous_boxes','previous_work_size','eligible'}}
        record.update(people=people,review_status='pending',include_in_training=False,
            exhaustive_person_annotation=False,no_box_is_negative=False,
            risks=sorted({r for p in people for r in p['risks']}) if people else ['missing_person_or_unresolvable'],
            previous_person_count=len(frame['previous_boxes']) if frame['previous_boxes'] is not None else None)
        enhanced_path=destination/'enhanced'/f'{frame["frame"]:06d}.jpg'
        enhanced_path.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(enhanced_path),enhanced,[cv2.IMWRITE_JPEG_QUALITY,92]):
            raise IOError(enhanced_path)
        record['enhanced_image']=enhanced_path.relative_to(OUT).as_posix()
        if sample:
            preview=draw(gamma(raw),record,video)
            cv2.imwrite(str(destination/f'smoke_{frame["frame"]:06d}.jpg'),preview)
        records.append(record)
    result=dict(schema='dark_person_pose_review_v1',video=video,frames=records,
        coordinate_system='native source pixels; bbox xyxy; COCO17 xy and model score; visibility null until reviewed',
        keypoint_names=JOINTS,skeleton_edges=EDGES,include_in_training=False,
        exhaustive_person_annotation=False,source_luma='original before enhancement, resized longest side 720',
        proposal_method='YOLO26m + RTMO-s, raw/gamma/Retinexformer views; RTMPose-s top-down alternatives; sparse visual anchors',
        user_confirmed_boxes=sum(p['box_review']=='user_confirmed' for f in records for p in f['people']),
        user_confirmed_poses=0,elapsed=time.monotonic()-started)
    save(path,result)
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard',default='0/1')
    parser.add_argument('--ids',default='')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(1)
    inventory=json.loads((OUT/'inventory.json').read_text())
    selected=[]
    wanted=set(args.ids.split(',')) if args.ids else None
    for video in inventory['raw_videos']:
        path=OUT/'data'/video['id']/'scan.json'
        if not path.exists():
            if wanted and video['id'] not in wanted:
                continue
            raise RuntimeError(f'Scan not finished: {path}')
        scan=json.loads(path.read_text())
        if scan['dark_frames'] and (wanted is None or video['id'] in wanted):
            selected.append(scan)
    shard,total=map(int,args.shard.split('/'))
    selected=selected[shard::total]
    references=json.loads((ROOT/'reports/dark-person-label-audit-20261023/annotations-draft.json').read_text())
    corrections=json.loads((OUT/'visual-anchor-additions.json').read_text())
    references['frames'].extend(corrections['frames'])
    proposer=Proposer()
    save(OUT/f'models-shard-{shard}.json',{name:dict(path=str(path),sha256=digest(path)) for name,path in MODELS.items()})
    for index,scan in enumerate(selected,1):
        anchors=sorted([r for r in references['frames'] if references['sources'][r['video']]['sha256']==scan['video']['sha256']],
                       key=lambda r:r['frame_index_zero_based'])
        result=process(scan,proposer,anchors,args.smoke)
        print(f'[propose] {index}/{len(selected)} {scan["video"]["source"]} frames={len(result["frames"])} '
              f'person_boxes={sum(len(f["people"]) for f in result["frames"])} elapsed={result["elapsed"]:.1f}',flush=True)


if __name__=='__main__':
    main()
