"""Extract original enhanced person pixels and exact same-person COCO17 sequences.

New recordings replay the complete frontend; frame/track/box correspondence is
verified against the reviewed run. Old crops use frozen per-frame boxes. No text
overlay previews or XCLIP outputs enter either model input.
"""
import argparse
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import sys
import time
from types import SimpleNamespace
from common import *

sys.path.insert(0, str(OP))
import cv2
import numpy as np
import torch
from darkpipe.config import PipelineConfig
from darkpipe.person.runtime import Frontend, content_height
from darkpipe.person.core import PersonTracker, crop_person, iou
from darkpipe.stages.downscale import DownscaleStage
from darkpipe.stages.enhance_retinexformer import RetinexformerStage


def arrays(window):
    return dict(pose=np.stack([s.pose for s in window]),
        boxes=np.stack([s.person_box for s in window]),
        lumas=np.array([np.nan if s.luma is None else s.luma for s in window]),
        binding=np.array([s.binding_valid for s in window]),
        shape=np.array(window[-1].frame_shape),
        source_frames=np.array([s.frame_index for s in window]),
        timestamps=np.array([s.timestamp for s in window]))


def write_npz(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.tmp').open('wb') as stream:
        np.savez_compressed(stream, **values)
    path.with_suffix('.tmp').replace(path)


def save_window(row, window, executor):
    pixels = np.stack([s.crop for s in window])
    crop_hash = hashlib.sha256(pixels.tobytes()).hexdigest()
    assert pixels.shape == (32,224,400,3) and pixels.dtype == np.uint8
    values = arrays(window)
    rgb_path = Path(row['path'])
    futures = []
    if not rgb_path.exists():
        futures.append(executor.submit(write_npz, rgb_path, dict(rgb=pixels, pose=values['pose'])))
    elif 'crop_sha256' in row:
        # Existing tensors are reused only when the pixel identity is proven below.
        pass
    feature_path = OUT/'features'/(row['id']+'.npz')
    futures.append(executor.submit(write_npz, feature_path, values))
    return dict(id=row['id'], crop_sha256=crop_hash, expected_crop_sha256=row.get('crop_sha256'),
        exact_crop_hash=crop_hash == row.get('crop_sha256', crop_hash),
        feature_path=str(feature_path), rgb_path=str(rgb_path)), futures


def new_source(source, rows, enhancer, front, executor):
    done = OUT/'extraction'/(rows[0]['source_id']+'.json')
    if done.exists(): return
    assert sha(source['absolute_path']) == source['sha256']
    targets = defaultdict(list)
    ranges = defaultdict(list)
    for r in rows:
        targets[r['source_frames'][-1]].append(r)
        ranges[(r['track_id'], r['generation'])].append((r['source_frames'][0]-6, r['source_frames'][-1]))
    tracker = PersonTracker()
    generations, seen = {}, {}
    cap = cv2.VideoCapture(source['absolute_path']); assert cap.isOpened()
    fps = float(Fraction(source['fps']))
    assert abs(cap.get(cv2.CAP_PROP_FPS)-fps) < .0001
    down = DownscaleStage(720)
    from darkpipe.stages.recognize import frame_luma
    pending, records, index, start = deque(), [], 0, time.monotonic()
    try:
        while True:
            ok, raw = cap.read()
            if not ok: break
            small = down([raw])[0]; luma = frame_luma(small)
            frame = enhancer([small])[0]
            boxes, poses = front(frame, luma >= 30.)
            stamp = index/fps
            tracks = tracker.update(boxes, poses, frame, stamp)
            by_tid = {t.tid:t for t in tracks}
            for t in tracks:
                if stamp-seen.get(t.tid, stamp) > .2:
                    generations[t.tid] = generations.get(t.tid, 0)+1
                seen[t.tid] = stamp
                key = (t.tid, generations.get(t.tid, 0))
                if any(a <= index <= b for a,b in ranges.get(key, [])):
                    binding = not any(iou(t.box,o.box)>.4 for o in tracks if o is not t)
                    t.append(frame,stamp,index,luma,binding_valid=binding)
            for row in targets.get(index, []):
                t = by_tid.get(row['track_id'])
                assert t is not None and generations.get(t.tid,0)==row['generation'], (row['id'],'identity mismatch')
                assert np.allclose(t.box,row['bbox'],atol=.02,rtol=0), (row['id'],'box mismatch',t.box,row['bbox'])
                win = t.window(fps)
                assert win is not None and [s.frame_index for s in win]==row['source_frames'], (row['id'],'source frame mismatch')
                result, jobs = save_window(row,win,executor)
                records.append(result);pending.extend(jobs)
                while len(pending)>6:pending.popleft().result()
            index += 1
            if index%250==0:
                progress=dict(source=rows[0]['source_id'],frames=index,total=source['frames'],windows=len(records),expected=len(rows),seconds=time.monotonic()-start)
                save(OUT/'progress'/(rows[0]['source_id']+'.json'),progress)
                print(progress,flush=True)
        assert index==source['frames'] and len(records)==len(rows)
        while pending:pending.popleft().result()
        save(done,dict(source_id=rows[0]['source_id'],frames=index,rows=records,seconds=time.monotonic()-start,
            exact_original_track_box_and_frames=True))
    finally:cap.release()


def old_source(video, rows, enhancer, executor):
    done=OUT/'extraction'/('old-'+video['id']+'.json')
    if done.exists():return
    source=ROOT/video['source'];assert sha(source)==video['sha256']
    current=read(CURRENT/'data'/video['id']/'result.json')
    events=read(CURRENT/'evaluated-windows.json')
    needed=defaultdict(list)
    for r in rows:
        e=events[r['current_index']]
        for j,fi in enumerate(e['source_frames']):needed[fi].append((r,j,e))
    if not needed:
        save(done,dict(source_id=video['id'],rows=[]));return
    cap=cv2.VideoCapture(str(source));assert cap.isOpened()
    down=DownscaleStage(720);windows={r['id']:[None]*32 for r in rows};index=0;records=[];pending=deque();start=time.monotonic()
    try:
        for fi, requests in sorted(needed.items()):
            while index<fi: assert cap.grab();index+=1
            ok, raw=cap.read();index+=1;assert ok
            if video['processed']:
                frame=down([raw[:content_height(raw.shape[0])]])[0]
            else:frame=enhancer([down([raw])[0]])[0]
            meta=current['frames'][fi]
            tracks={t['track_id']:t for t in meta['tracks']['30']}
            crops={}
            for r,j,e in requests:
                box=np.array(tracks[e['track_id']]['bbox'],np.float32)
                positions=[k for k,b in enumerate(meta['boxes']) if np.allclose(box,b[:4],atol=.001,rtol=0)]
                assert len(positions)==1
                pose=meta['poses'][positions[0]]
                pose=np.zeros((17,3),np.float32) if pose is None else np.array(pose,np.float32)
                pose[:,:2]/=frame.shape[1::-1]
                rect=tuple(e['crop_boxes'][j])
                if rect not in crops:crops[rect]=crop_person(frame,np.array(rect),margin=0)[0]
                binding=not any(iou(box,np.array(t['bbox']))>.4 for tid,t in tracks.items()if tid!=e['track_id'])
                sample=SimpleNamespace(crop=crops[rect],pose=pose,person_box=box,luma=meta['luma'],
                    binding_valid=binding,frame_shape=frame.shape[:2],frame_index=fi,timestamp=fi/video['fps'])
                windows[r['id']][j]=sample
                if j==31:
                    win=windows.pop(r['id']);assert all(s is not None for s in win)
                    result,jobs=save_window(r,win,executor)
                    if not result['exact_crop_hash']:
                        raise RuntimeError(f'Historical RGB pixel mismatch: {r["id"]}; do not reuse incompatible probabilities')
                    records.append(result);pending.extend(jobs)
                    while len(pending)>6:pending.popleft().result()
        while pending:pending.popleft().result()
        assert len(records)==len(rows)
        save(done,dict(source_id=video['id'],rows=records,seconds=time.monotonic()-start,exact_pixels=True))
        print('OLD',video['id'],len(records),round(time.monotonic()-start,1),flush=True)
    finally:cap.release()


def main():
    p=argparse.ArgumentParser();p.add_argument('--scope',choices=['old','new'],required=True);p.add_argument('--shard',default='0/1');a=p.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(1)
    rows=read(OUT/'dataset-plan.json')['records'];i,n=map(int,a.shard.split('/'))
    enhancer=RetinexformerStage(str(OP/'ckpts'),chunk=1);enhancer.load('cuda:0')
    cfg=PipelineConfig(device='cuda:0',ckpt_dir=str(OP/'ckpts'))
    with ThreadPoolExecutor(max_workers=2) as executor:
        try:
            if a.scope=='new':
                front=Frontend(cfg)
                for source in read(REVIEW/'sources.json')[i::n]:
                    selected=[r for r in rows if r['scope']=='new' and r['number']==source['number']]
                    if selected:new_source(source,selected,enhancer,front,executor)
            else:
                for video in read(CURRENT/'protocol.json')['videos'][i::n]:
                    selected=[r for r in rows if r['scope']=='old' and r['source_id']==video['id'] and r['current_index'] is not None]
                    old_source(video,selected,enhancer,executor)
        finally:enhancer.close()
    save(OUT/f'extract-{a.scope}-{i}-complete.json',dict(complete=True,protocol_sha256=sha(OUT/'protocol.json')))


if __name__=='__main__':main()
