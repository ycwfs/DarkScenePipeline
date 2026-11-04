"""Filter auxiliary dark annotations without changing their original evidence.

User approved aggregate quality on 2026-10-24, not exhaustive human ground truth.
Unknown people/frames are excluded, never converted into background negatives.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
LABELS = ROOT / 'reports/dark-annotations-20261023'
EDGES = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),
         (11,13),(13,15),(12,14),(14,16)]
RULES = dict(min_track_seconds=.4, max_gap_seconds=.12, min_joint_seconds=.16,
             min_pose_seconds=.4, min_pose_joints=8, train_pose_joints=10,
             keypoint_conf=.4, max_disagreement=.15, sample_hz=5,
             max_frames_per_video=40, max_train_frames_per_group=400,
             smoothing='centered 3-frame 0.2/0.6/0.2, contiguous valid points only',
             unknown_person_policy='exclude whole image from detector training',
             visibility='pseudo supervision mask, not measured COCO visibility')


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':'))+'\n')
    temp.replace(path)


def iou(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    inter = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
    return float(inter / max(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter, 1))


def runs(mask: np.ndarray, times: np.ndarray, max_gap: float = .12) -> list:
    groups, current = [], []
    for i, yes in enumerate(mask):
        if not yes or (current and times[i]-times[current[-1]] > max_gap):
            if current: groups.append(current)
            current = []
        if yes: current.append(i)
    if current: groups.append(current)
    return groups


def filter_pose(points: np.ndarray, usable: np.ndarray, boxes: np.ndarray,
                times: np.ndarray, fps: float) -> tuple:
    """Offline labels: reject spikes/short runs; never interpolate missing joints."""
    xy = points[:, :, :2].copy()
    valid = usable.copy()
    diag = np.maximum(np.linalg.norm(boxes[:, 2:]-boxes[:, :2], axis=1), 1)
    center = (boxes[:, 2:]+boxes[:, :2])/2
    relative = (xy-center[:, None, :])/diag[:, None, None]
    reasons = Counter()
    # Reject an isolated excursion only when both neighbours agree on the joint.
    for i in range(1, len(times)-1):
        if max(times[i]-times[i-1], times[i+1]-times[i]) > .12: continue
        common = valid[i-1] & valid[i] & valid[i+1]
        neighbour = np.linalg.norm(relative[i-1]-relative[i+1], axis=1)
        spike = np.linalg.norm(relative[i]-(relative[i-1]+relative[i+1])/2, axis=1)
        bad = common & (neighbour < .06) & (spike > .12)
        valid[i, bad] = False
        reasons['isolated_spike_joints'] += int(bad.sum())
    # Segment-specific bone consistency; no global assumption of an upright body.
    for a, b in EDGES:
        common = valid[:, a] & valid[:, b]
        length = np.linalg.norm(relative[:, a]-relative[:, b], axis=1)
        if common.sum() < 8: continue
        median = np.median(length[common])
        if median < .015: continue
        bad = common & ((length < .35*median) | (length > 2.5*median))
        for i in np.flatnonzero(bad):
            joint = a if points[i, a, 2] < points[i, b, 2] else b
            valid[i, joint] = False
            reasons['bone_outlier_joints'] += 1
    for j in range(17):
        for run in runs(valid[:, j], times):
            if times[run[-1]]-times[run[0]]+1/fps < RULES['min_joint_seconds']:
                valid[run, j] = False
                reasons['short_joint_run_points'] += len(run)
    pose_ok = np.zeros(len(times), dtype=bool)
    for run in runs(valid.sum(1) >= RULES['min_pose_joints'], times):
        if times[run[-1]]-times[run[0]]+1/fps >= RULES['min_pose_seconds']:
            pose_ok[run] = True
    valid[~pose_ok] = False
    smooth = xy.copy()
    for i in range(1, len(times)-1):
        if max(times[i]-times[i-1], times[i+1]-times[i]) > .12: continue
        common = valid[i-1] & valid[i] & valid[i+1]
        # A small symmetric local filter does not introduce phase lag in labels.
        smooth[i, common] = .2*xy[i-1, common]+.6*xy[i, common]+.2*xy[i+1, common]
    return smooth, valid, dict(reasons)


def build() -> None:
    count, videos, all_samples = Counter(), [], []
    old_splits = json.loads((ROOT/'reports/person-finetune-20261023/splits.json').read_text())
    legacy_other = next(r for r in old_splits['records']
                        if r['source']=='real-actiondata-all/other/other1.mp4')
    for path in sorted((LABELS/'data').glob('*/annotations.json')):
        data = json.loads(path.read_text()); video = data['video']; fps = video['fps']
        split = dict(video['frozen_split'])
        if split['split']=='unassigned':
            # All six newly inventoried AVIs belong to the existing legacy-other family.
            assert video['source'].startswith('real-actiondata-0909/other/')
            split.update(split=legacy_other['split'], group=legacy_other['group'],
                         reason='entire legacy other recording family remains train-only')
        video['effective_split'] = split
        count['source_frames'] += len(data['frames'])
        segments, active = [], {}
        accepted = defaultdict(list)
        for frame in data['frames']:
            for person in frame['people']:
                count['source_boxes'] += 1
                key = person['person_id']; box = np.array(person['bbox_xyxy'])
                prior = active.get(key)
                if prior:
                    dt = frame['timestamp']-prior[-1][0]['timestamp']
                    old = np.array(prior[-1][1]['bbox_xyxy'])
                    area_ratio = np.prod(box[2:]-box[:2])/max(np.prod(old[2:]-old[:2]),1)
                    if dt <= 0 or dt > .12 or iou(box,old)<.3 or not .4<area_ratio<2.5:
                        prior = None
                if prior is None:
                    prior=[]; segments.append(prior); active[key]=prior
                prior.append((frame,person))
        audit=[]
        for sid, seq in enumerate(segments):
            times=np.array([f['timestamp'] for f,p in seq])
            duration=times[-1]-times[0]+1/fps
            if duration<RULES['min_track_seconds']:
                count['short_track_boxes'] += len(seq);continue
            boxes=np.array([p['bbox_xyxy'] for f,p in seq])
            points=np.array([p['keypoints'] for f,p in seq])
            usable=np.array([p['keypoint_usable'] for f,p in seq],bool)
            usable &= points[:,:,2]>=RULES['keypoint_conf']
            reliable=[]
            for i,(frame,p) in enumerate(seq):
                families={x.split('/')[0] for x in p['proposal_support']}
                manual='visual' in p['bbox_source'] or 'user_confirmed' in p['bbox_source']
                score=p['bbox_score'] or 0
                ok=(len(families)>=2 or manual or score>=.75)
                ok &= 'resembles_visually_rejected_background' not in p['risks']
                reliable.append(ok)
                if (p.get('max_pose_disagreement') or 0) > RULES['max_disagreement']:
                    usable[i]=False
                if not ok: usable[i]=False
                if p['pose_source']=='assistant_visual_body_joint_draft':
                    # Preserve the coordinate evidence, but an isolated visual pose
                    # does not bypass the minimum-duration rule.
                    usable[i]=np.array(p['keypoint_usable'],bool)
            reliable=np.array(reliable,bool)
            for run in runs(reliable,times):
                if times[run[-1]]-times[run[0]]+1/fps < RULES['min_track_seconds']:
                    reliable[run]=False
            usable[~reliable]=False
            smooth,valid,reasons=filter_pose(points,usable,boxes,times,fps)
            count.update(reasons)
            for i,(frame,p) in enumerate(seq):
                row=dict(person_id=p['person_id'],segment=sid,bbox_xyxy=p['bbox_xyxy'],
                    bbox_accepted=bool(reliable[i]),keypoints_xy=smooth[i].round(4).tolist(),
                    keypoint_mask=valid[i].tolist(),pose_accepted=bool(valid[i].sum()>=10),
                    bbox_source=p['bbox_source'],pose_source=p['pose_source'],
                    duration=duration,quality='user broadly accepted filtered auxiliary label')
                audit.append(dict(frame=frame['frame'],**row))
                if reliable[i]: accepted[frame['frame']].append(row)
        selected=[];last=-1e9
        for f in data['frames']:
            people=accepted[f['frame']]
            # No empty/partially rejected image is silently treated as exhaustive GT.
            if not people or len(people)!=len(f['people']):
                count['unknown_or_partial_frames_excluded']+=1;continue
            if f['timestamp']-last < 1/RULES['sample_hz']-1e-6: continue
            last=f['timestamp']
            selected.append(dict(video_id=video['id'],source=video['source'],
                frame=f['frame'],timestamp=f['timestamp'],luma=f['luma'],
                width=video['width'],height=video['height'],split=split['split'],
                group=split['group'],people=people,annotation_exhaustive=False))
        limit=RULES['max_frames_per_video']
        if len(selected)>limit:
            selected=[selected[i] for i in np.linspace(0,len(selected)-1,limit).round().astype(int)]
        all_samples.extend(selected)
        videos.append(dict(video=video,selected_frames=len(selected),segments=len(segments)))
        save(OUT/'data'/video['id']/'filtered.json',dict(video=video,people=audit))
    grouped=defaultdict(list)
    for sample in all_samples:grouped[sample['group']].append(sample)
    final=[]
    for group,items in sorted(grouped.items()):
        assert len({r['split'] for r in items})==1,'source group leaks between splits'
        limit=RULES['max_train_frames_per_group']
        if items[0]['split']=='train' and len(items)>limit:
            items=[items[i] for i in np.linspace(0,len(items)-1,limit).round().astype(int)]
        final.extend(items)
    for i,s in enumerate(final):s['id']=i+1
    summary=dict(counts=dict(count),videos=len(videos),selected_images=len(final),
        splits={sp:dict(images=sum(x['split']==sp for x in final),
            boxes=sum(len(x['people']) for x in final if x['split']==sp),
            pose_people=sum(p['pose_accepted'] for x in final if x['split']==sp for p in x['people']),
            videos=len({x['video_id'] for x in final if x['split']==sp}),
            groups=len({x['group'] for x in final if x['split']==sp})) for sp in ['train','val','test']})
    save(OUT/'protocol.json',dict(rules=RULES,user_authorized='2026-10-24 aggregate review and training request',
        split_sha256=hashlib.sha256((ROOT/'reports/person-finetune-20261023/splits.json').read_bytes()).hexdigest(),
        input_label_status='filtered auxiliary annotations, not independent exhaustive human GT',
        sample_views='enhanced720 and raw720 for train; enhanced720 for validation/test',
        selection='validation only; test after checkpoint selection; original model eligible',
        frozen_models=['Retinexformer','VideoMamba','ST-GCN'],production_modified=False))
    save(OUT/'manifest.json',dict(samples=final,videos=videos))
    save(OUT/'filter-summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def export_images(shard: int, shards: int) -> None:
    import torch
    sys.path.insert(0,str(ROOT/'platform/dark-demo-v2'))
    from darkpipe.stages.enhance_retinexformer import RetinexformerStage
    torch.set_num_threads(2);cv2.setNumThreads(1)
    enhancer=RetinexformerStage(str(ROOT/'platform/dark-demo-v2/ckpts'),chunk=1)
    enhancer.load('cuda:0')
    samples=json.loads((OUT/'manifest.json').read_text())['samples']
    grouped=defaultdict(list)
    for s in samples:grouped[s['video_id']].append(s)
    for n,(uid,items) in enumerate(sorted(grouped.items())):
        if n%shards!=shard:continue
        pending={s['frame']:s for s in items}
        cap=cv2.VideoCapture(str(ROOT/items[0]['source']));index=-1
        while pending:
            ok,native=cap.read();index+=1
            if not ok:raise RuntimeError(f'decode stopped {uid} {index}')
            if index not in pending:continue
            s=pending.pop(index)
            h,w=native.shape[:2];scale=min(720/max(h,w),1)
            raw=cv2.resize(native,(round(w*scale)//2*2,round(h*scale)//2*2),interpolation=cv2.INTER_AREA)
            dest=OUT/'dataset/images'/s['split'];dest.mkdir(parents=True,exist_ok=True)
            targets={'enhanced':dest/f'{uid}_{index:06d}_enhanced.png'}
            if s['split']=='train':targets['raw']=dest/f'{uid}_{index:06d}_raw.png'
            if all(p.exists() for p in targets.values()):continue
            enhanced=enhancer([raw])[0]
            for view,p in targets.items():
                if not cv2.imwrite(str(p),enhanced if view=='enhanced' else raw):raise IOError(p)
        cap.release()
        print(f'[export {shard}] {n+1}/{len(grouped)} {items[0]["source"]} {len(items)}',flush=True)
    enhancer.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true')
    parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=1)
    args=parser.parse_args()
    export_images(args.shard,args.shards) if args.export else build()
