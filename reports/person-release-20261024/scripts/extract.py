"""Gate30 dataset extraction with frozen person/time labels; train/val/test kept separate."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'reports/person-finetune-20261023'
sys.path[:0] = [str(ROOT/'reports/person-action-20261023/scripts'),
                str(ROOT/'reports/dark-finetune-20261024/scripts'), str(DATA/'scripts')]
from person_core import (OP, BEHAVIORS, PersonTracker, associate_poses, person_boxes,
                         fuse_person, render_people, iou)
from demo import content_height
from inference import Frontend
from assemble import assign_label
from darkpipe.stages.recognize import BehaviorRecognizer, frame_luma, gate_luma, reject_probs
from darkpipe.stages.enhance_retinexformer import RetinexformerStage
from darkpipe.stages.downscale import DownscaleStage
from darkpipe.stages.skeleton_aux import STGCN, normalize_window, completeness

GATES = (30,)
MODEL_PATHS = dict(
    detector=ROOT/'reports/detector-clean-finetune-20261024/models/yolo26m_clean_candidate.onnx',
    pose=ROOT/'reports/dark-finetune-20261024/models/rtmo-s_dark_candidate.onnx',
    rgb=DATA/'models/videomamba_t_behavior_32f_person_roi_20261023.pth',
    skeleton=OP/'ckpts/stgcn_behavior_32f_realactall_full.pth', enhancer=OP/'ckpts/NTIRE.pth')


def save(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, separators=(',', ':'))+'\n')
    tmp.replace(path)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> None:
    splits = {r['video_id']: r for r in json.loads((DATA/'splits.json').read_text())['records']}
    atlas = json.loads((ROOT/'reports/luma-samples-20261024/selected-sources.json').read_text())['records']
    atlas_ids = {v['id'] for v in atlas}
    videos = []
    for v in json.loads((DATA/'source-inventory.json').read_text())['videos']:
        split = splits[v['id']]
        old = json.loads((DATA/'data'/v['id']/'manifest.json').read_text())
        ls = [r['luma'] for r in old['frames'] if r['luma'] is not None]
        if True:
            videos.append(dict(**v, split=split['split'], group=split['group'],
                raw_mean_luma=float(np.mean(ls)) if ls else None,
                demo=False))
    videos.sort(key=lambda v: (v['split']!='val', v['split']!='test', v['source']))
    protocol = dict(gates=GATES, alpha=.7, threshold=.75, smoothing=False,
        selection='All535 deduplicated onsite and online negative sources;417 train,60 val,58 test frozen source groups.',
        models={k:dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for k,p in MODEL_PATHS.items()},
        source_splits_sha256=sha(DATA/'splits.json'), annotations_sha256=sha(DATA/'annotations.json'),
        tracking='Independent identical PersonTracker per gate, including gate-specific pose association. No temporal smoothing.',
        gating='Per-frame raw L enables pose extraction; sampled32-frame mean enables fusion. Mask poses from below-threshold frames.',
        labels='Original reviewed person/time labels transferred by per-frame bbox IoU>=.5 with ambiguity margin. Unmatched positive actors unknown. Source other inherited; no detections not counted correct.',
        inference='Shared original decoded/enhanced/detected pixels; RGB evaluated once per exact crop SHA, shared only when bytes and sampled indices match. Skeleton separately gate-specific.',
        units='person windows; paired exact crop+source-frame windows; confidence intervals clustered by original source group',
        evaluation_split_caveat='Historical initialization saw many sources; test inspected previously. Train only on train; select checkpoints and alpha/threshold on val, then lock before test scoring.',
        tuning=dict(seed=24, skeleton_epochs=20, skeleton_lr=0.0001, alpha_grid=[i/40 for i in range(41)], threshold_grid=[i/100 for i in range(50,96)], no_pose=['soft','rgb'], criterion='maximum validation macro F1 over supported classes subject to other false positives no higher than gate30 old-STGCN alpha.7 threshold.75 soft baseline; ties: fewer false positives, higher positive recall, closest previous settings'),
        production_modified=False, candidate_only=False, videos=videos)
    save(OUT/'protocol.json', protocol)
    print('Frozen',len(videos),'sources',sum(v['frames'] for v in videos),'frames',Counter(v['split']for v in videos),flush=True)


class Models:
    def __init__(self) -> None:
        self.front = Frontend(MODEL_PATHS['detector'], MODEL_PATHS['pose'])
        self.enhancer = RetinexformerStage(str(OP/'ckpts'), chunk=1)
        self.enhancer.load('cuda:0')
        self.rgb = BehaviorRecognizer(str(OP/'ckpts'), skel_fusion_alpha=1.)
        self.rgb.ckpt = str(MODEL_PATHS['rgb'])
        self.rgb.load('cuda:0')
        self.skeleton = STGCN(num_classes=10)
        state = torch.load(MODEL_PATHS['skeleton'], map_location='cpu', weights_only=False)
        self.skeleton.load_state_dict(state.get('model', state))
        self.skeleton = self.skeleton.cuda().eval()
        self.mean = torch.as_tensor(self.rgb.cfg['mean'],device='cuda:0')
        self.std = torch.as_tensor(self.rgb.cfg['std'],device='cuda:0')

    @torch.inference_mode()
    def predict(self, requests: list, rgb_cache: dict) -> list:
        arrays, keys = {}, []
        for gate, track, window in requests:
            pixels = np.stack([s.crop for s in window])
            key = hashlib.sha256(pixels.tobytes()).hexdigest()
            keys.append(key)
            if key not in rgb_cache: arrays[key] = pixels
        unique = list(arrays)
        for start in range(0,len(unique),4):
            ks=unique[start:start+4]
            pixels=torch.from_numpy(np.stack([arrays[k]for k in ks])).cuda().float()
            x=pixels.div_(255).sub_(self.mean).div_(self.std).permute(0,4,1,2,3).contiguous()
            with torch.autocast('cuda',dtype=torch.float16):
                probs=self.rgb.net(x).softmax(1).float().cpu().numpy()
            rgb_cache.update(zip(ks,probs))
        results=[]
        for (gate, track, window),key in zip(requests,keys):
            sk=np.stack([s.pose for s in window])[:,None]
            normal,presence=normalize_window(sk)
            visible=(sk[:,0,:,2]>.3).sum(1)>=10
            eligible=bool(presence[0]>=.5 and completeness(sk)[0]>=.5 and visible.mean()>=.5
                          and sk[-1,0,:,2].mean()>=.5 and visible[-1])
            lumas=[s.luma for s in window]
            valid=eligible and gate_luma(lumas,gate)is None
            skp=None
            if valid:
                skp=self.skeleton(torch.from_numpy(normal[:1]).cuda()).softmax(1).float().cpu().numpy()[0]
            p=rgb_cache[key]
            row=fuse_person(p,skp,lumas,lum_gate=gate)
            row.update(gate=gate, track_id=track.tid, frame_index=window[-1].frame_index,
                timestamp=window[-1].timestamp, source_frames=[s.frame_index for s in window],
                timestamps=[s.timestamp for s in window], crop_boxes=[s.crop_box.tolist()for s in window],
                crop_sha256=key, bbox=track.box.tolist(), pose_valid=valid, pose_eligible=eligible,
                pose_visible_fraction=float(visible.mean()), pose_present_fraction=float(presence[0]),
                normalized_pose=normal[0].tolist(),
                rgb_label=BEHAVIORS[int(reject_probs(p,BEHAVIORS,.75).argmax())])
            results.append(row)
        return results


def assign_truth(row: dict, video: dict, old: dict, annotations: dict) -> None:
    if video['video_label']=='other':
        row.update(truth='other', truth_provenance=video['label_provenance'])
        return
    votes=[]
    for fi, box in zip(row['source_frames'],row['crop_boxes']):
        # The old crop_boxes include15% context; compare to old boxes expanded by the same rule.
        candidates=[]
        for t in old['frames'][fi]['tracks']:
            b=np.array(t['bbox']);sz=b[2:]-b[:2]
            expanded=np.r_[np.maximum(0,np.floor(b[:2]-.15*sz)),np.minimum([720,404],np.ceil(b[2:]+.15*sz))]
            candidates.append((iou(np.array(box),expanded),t['track_id']))
        candidates.sort(reverse=True)
        good=bool(candidates and candidates[0][0]>=.5 and (len(candidates)==1 or candidates[0][0]-candidates[1][0]>=.1))
        votes.append(candidates[0][1]if good else None)
    counts=Counter(v for v in votes if v is not None)
    tid,n=counts.most_common(1)[0]if counts else (None,0)
    ann=annotations.get((video['id'],tid))
    label=None;coverage={}
    if n>=len(votes)*.8 and all(v in (tid,None)for v in votes) and ann and ann['status']=='reviewed':
        label,coverage=assign_label(row['timestamps'],ann['segments'])
    row.update(truth=label,truth_provenance='original reviewed person/time; spatial transfer' if label else 'unknown, excluded',
               original_track_id=tid, original_track_match_fraction=n/len(votes), annotation_coverage=coverage)


FONT=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',21)


def mosaic(frame: np.ndarray, tracks: dict, timestamp: float, luma: float|None, source: str) -> np.ndarray:
    panes=[]
    for g in GATES:
        shown=render_people(frame,tracks[g],frame.shape,timestamp)
        shown=cv2.resize(shown,(720,404))
        panes.append(shown)
    canvas=np.full((488,1440,3),24,np.uint8)
    canvas[40:444]=np.hstack(panes)
    pil=Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(pil)
    for i,g in enumerate(GATES):d.text((i*720+12,6),f'门限 {g} | 同一组最新模型，无骨架平滑',font=FONT,fill='white')
    d.text((12,451),f'{Path(source).name} | t={timestamp:.2f}s | 原图L={luma:.2f}'if luma is not None else f'{Path(source).name} | 回传原始L未知',font=FONT,fill='white')
    return cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)


def run(video: dict, models: Models, annotations: dict, protocol_sha: str) -> dict:
    dest=OUT/'data'/video['id'];dest.mkdir(parents=True,exist_ok=True)
    result=dest/'result.json'
    if result.exists():
        saved=json.loads(result.read_text());assert saved['protocol_sha256']==protocol_sha
        return saved['summary']
    assert sha(ROOT/video['source'])==video['sha256']
    cap=cv2.VideoCapture(str(ROOT/video['source']));assert cap.isOpened()
    old=json.loads((DATA/'data'/video['id']/'manifest.json').read_text())
    previous_path=ROOT/'reports/luma-gate15-20261024/data'/video['id']/'result.json'
    previous=json.loads(previous_path.read_text()) if previous_path.exists() else None
    trackers={g:PersonTracker()for g in GATES};events={g:[]for g in GATES};frames=[]
    down=DownscaleStage(720);cache={};index=0;started=time.monotonic();encoder=None;review={}
    if video['demo']:
        vp=OUT/'videos'/f'{video["id"]}.mp4';vp.parent.mkdir(exist_ok=True)
        encoder=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1440x488',
            '-r',str(video['fps']),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',
            '-threads','2','-movflags','+faststart',str(vp)],stdin=subprocess.PIPE)
    try:
        while True:
            ok,raw=cap.read()
            if not ok:break
            ts=index/video['fps']
            if video['processed']:
                frame=down([raw[:content_height(raw.shape[0])]])[0];luma=None
            else:
                raw=down([raw])[0];luma=frame_luma(raw);frame=models.enhancer([raw])[0]
            assert luma is None or abs(luma-old['frames'][index]['luma'])<1e-7
            if previous is not None:
                cached=previous['frames'][index]
                assert cached['luma']==luma
                boxes=np.array(cached['boxes'],np.float32).reshape(-1,6)
                pose=[None if p is None else np.array(p,np.float32) for p in cached['poses']]
            else:
                boxes=person_boxes(models.front.detector._frames_to_dets([frame])[0])
                pose=[None]*len(boxes)
                if len(boxes) and (luma is None or luma>=min(GATES)):
                    _,xy,cf,_=models.front.poses(frame);pose=associate_poses(boxes,xy,cf)
            current={};requests=[]
            for g in GATES:
                poses=pose if luma is None or luma>=g else [None]*len(boxes)
                current[g]=trackers[g].update(boxes,poses,frame,ts)
                for t in current[g]:
                    t.append(frame,ts,index,luma)
                    if ts-t.last_inference>1.:t.prediction=None
                    w=t.window(video['fps'])
                    if w is not None and ts-t.last_inference>=.5-1e-6:requests.append((g,t,w))
                    if g==15 and index%max(1,round(video['fps']*.25))==0:
                        review.setdefault(t.tid,[]).append((ts,cv2.cvtColor(t.history[-1].crop,cv2.COLOR_RGB2BGR)))
            for (g,t,w),row in zip(requests,models.predict(requests,cache)):
                assign_truth(row,video,old,annotations)
                events[g].append(row);t.prediction=row;t.last_inference=ts
            frames.append(dict(frame=index,timestamp=ts,luma=luma,boxes=boxes.tolist(),
                poses=[None if p is None else p.tolist()for p in poses],
                tracks={str(g):[dict(track_id=t.tid,bbox=t.box.tolist(),label=t.prediction['label']if t.prediction else None,
                    mode=t.prediction['mode']if t.prediction else None)for t in current[g]]for g in GATES}))
            if encoder:encoder.stdin.write(mosaic(frame,current,ts,luma,video['source']).tobytes())
            if index==min(video['frames']-1,round(video['fps']*1.6)) and video['demo']:
                (OUT/'posters').mkdir(exist_ok=True);cv2.imwrite(str(OUT/'posters'/f'{video["id"]}.jpg'),mosaic(frame,current,ts,luma,video['source']))
            index+=1
    finally:
        cap.release()
        if encoder:
            encoder.stdin.close()
            if encoder.wait(timeout=60):raise RuntimeError('Video encoding failed')
    assert index==video['frames']==len(old['frames'])
    # Prediction-free contact sheets support review of newly detected positive actors.
    if video['video_label']!='other' and (video['raw_mean_luma'] or 999)<75:
        for tid,tiles in review.items():
            chosen=np.linspace(0,len(tiles)-1,min(12,len(tiles))).round().astype(int)
            canvas=np.full((3*142,4*250,3),24,np.uint8)
            for j,k in enumerate(chosen):
                ts,crop=tiles[k];im=cv2.resize(crop,(250,112));y=(j//4)*142;x=(j%4)*250
                canvas[y:y+112,x:x+250]=im
                cv2.putText(canvas,f'T{tid:02d} t={ts:.2f}',(x+5,y+132),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            rp=OUT/'review'/f'{video["id"]}_T{tid:02d}.jpg';rp.parent.mkdir(exist_ok=True);cv2.imwrite(str(rp),canvas)
    summary=dict(id=video['id'],source=video['source'],split=video['split'],frames=index,
        windows={g:len(events[g])for g in GATES},known={g:sum(e['truth']is not None for e in events[g])for g in GATES},
        elapsed=time.monotonic()-started,video=f'videos/{video["id"]}.mp4'if encoder else None)
    rows=events[30]
    np.savez_compressed(dest/'skeleton-inputs.npz', normalized=np.asarray([r.pop('normalized_pose')for r in rows],np.float32).reshape(-1,3,32,17))
    save(result,dict(protocol_sha256=protocol_sha,video=video,summary=summary,frames=frames,events=events))
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return summary


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--shard',default='0/1');p.add_argument('--limit',type=int,default=0)
    args=p.parse_args()
    if args.freeze:freeze();return
    cv2.setNumThreads(1);torch.set_num_threads(2)
    protocol=json.loads((OUT/'protocol.json').read_text());fingerprint=sha(OUT/'protocol.json')
    for name,path in MODEL_PATHS.items():assert sha(path)==protocol['models'][name]['sha256']
    assert sha(DATA/'annotations.json')==protocol['annotations_sha256']
    annotations={(a['video_id'],a['track_id']):a for a in json.loads((DATA/'annotations.json').read_text())['records']}
    shard,total=map(int,args.shard.split('/'));videos=protocol['videos'][shard::total]
    if args.limit:videos=videos[:args.limit]
    models=Models()
    for i,v in enumerate(videos):
        run(v,models,annotations,fingerprint)
        print(f'[shard {shard}/{total}] {i+1}/{len(videos)} complete',flush=True)
    save(OUT/f'complete-{shard}.json',dict(protocol_sha256=fingerprint,videos=len(videos),passed=True))


if __name__=='__main__':main()
