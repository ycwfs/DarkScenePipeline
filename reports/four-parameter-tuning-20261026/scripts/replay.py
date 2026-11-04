"""Replay independent trackers and gate-specific skeleton sequences on old raw videos."""
from collections import Counter
import copy
import importlib.util
import time

from common import *
import cv2
import torch

spec=importlib.util.spec_from_file_location('release_extract',CACHE/'scripts/extract.py')
release=importlib.util.module_from_spec(spec);spec.loader.exec_module(release)
from darkpipe.person import core


def exact(row):
    return row['crop_sha256'],tuple(row['source_frames'])


def run(v,models,annotations,reviewed,fingerprint):
    target=OUT/'replay'/f'{v["id"]}.json'
    if target.exists():
        assert json.loads(target.read_text())['protocol_sha256']==fingerprint
        return
    cached=json.loads((ROOT/v['cache']).read_text())
    low=json.loads((LOW/'data'/v['id']/'result.json').read_text())
    for f,l in zip(cached['frames'],low['frames']):
        assert f['luma']==l['luma'] and f['boxes']==l['boxes']
    original=json.loads((release.DATA/'data'/v['id']/'manifest.json').read_text())
    mapping={exact(r):(i,r) for i,r in enumerate(cached['events']['30'])}
    truth={exact(r):r for r in reviewed if r['video_id']==v['id']}
    rgb_cache={r['crop_sha256']:np.array(r['rgb_probs'],np.float32)
               for rs in [*low['events'].values(),cached['events']['30']] for r in rs}
    initial_keys=set(rgb_cache)
    trackers={g:core.PersonTracker() for g in GATES};events={str(g):[] for g in GATES}
    frame_tracks={str(g):[] for g in GATES};stats=Counter();started=time.monotonic()
    cap=cv2.VideoCapture(str(ROOT/v['source']));assert cap.isOpened()
    down=release.DownscaleStage(720)
    old_appearance,old_crop=core.appearance,core.crop_person
    appearance_cache={};crop_cache={}
    def appearance(frame,box):
        key=box.tobytes()
        if key not in appearance_cache:appearance_cache[key]=old_appearance(frame,box)
        return appearance_cache[key]
    def crop(frame,box):
        key=box.tobytes()
        if key not in crop_cache:crop_cache[key]=old_crop(frame,box)
        return crop_cache[key]
    core.appearance,core.crop_person=appearance,crop
    try:
        for fi,meta in enumerate(low['frames']):
            ok,raw=cap.read();assert ok
            small=down([raw])[0];lum=release.frame_luma(small)
            assert abs(lum-meta['luma'])<1e-7
            boxes=np.array(meta['boxes'],np.float32).reshape(-1,6)
            poses=[None if p is None else np.array(p,np.float32) for p in meta['poses']]
            if len(boxes):frame=models.enhancer([small])[0];stats['enhanced_frames']+=1
            else:frame=small
            appearance_cache.clear();crop_cache.clear();requests=[]
            for g,tkr in trackers.items():
                current=tkr.update(boxes,poses if lum>=g else [None]*len(boxes),frame,fi/v['fps'])
                frame_tracks[str(g)].append([dict(track_id=t.tid,bbox=t.box.tolist()) for t in current])
                for t in current:
                    t.append(frame,fi/v['fps'],fi,lum)
                    window=t.window(v['fps'])
                    if window is not None and fi/v['fps']-t.last_inference>=.5-1e-6:
                        requests.append((g,t,window))
            for (g,t,w),row in zip(requests,models.predict(requests,rgb_cache)):
                release.assign_truth(row,v,original,annotations)
                match=mapping.get(exact(row));known=truth.get(exact(row))
                if known is not None:
                    row['truth']=known['truth'];row['truth_provenance']=known['truth_provenance']
                row.update(id=f'{v["id"]}_{match[0]}' if match else f'{v["id"]}_g{g}_{len(events[str(g)])}',
                           baseline_id=f'{v["id"]}_{match[0]}' if match else None,
                           source_id=v['id'],scope=v['scope'],split=v['split'],group=v['group'],is_drink=None,
                           history_generation=0,event_index=len(events[str(g)]))
                row.pop('normalized_pose',None)
                if g==30:
                    assert match is not None,('gate30 crop/frames changed',v['id'],row['id'])
                    old=match[1]
                    assert row['track_id']==old['track_id']
                    assert np.allclose(row['rgb_probs'],old['rgb_probs'],atol=1e-6)
                    assert (row['skeleton_probs'] is None)==(old['skeleton_probs'] is None)
                    if row['skeleton_probs'] is not None:
                        assert np.allclose(row['skeleton_probs'],old['skeleton_probs'],atol=1e-6)
                    stats['gate30_windows_reproduced']+=1
                t.last_inference=fi/v['fps'];events[str(g)].append(row)
    finally:
        cap.release();core.appearance,core.crop_person=old_appearance,old_crop
    assert len(events['30'])==len(cached['events']['30'])
    for g in GATES:
        stats[f'unpaired_g{g}']=sum(e['baseline_id'] is None for e in events[str(g)])
        stats[f'changed_track_frames_g{g}']=sum(
            [(t['track_id'],t['bbox']) for t in a]!=[(t['track_id'],t['bbox']) for t in b['tracks']['30']]
            for a,b in zip(frame_tracks[str(g)],cached['frames']))
    stats['new_rgb_crops']=len(set(rgb_cache)-initial_keys)
    save(target,dict(protocol_sha256=fingerprint,source_id=v['id'],events=events,tracks=frame_tracks,
                     stats=dict(stats),elapsed=time.monotonic()-started))
    print(v['id'],Path(v['source']).name,dict(stats),'sec',round(time.monotonic()-started,2),flush=True)


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--shard',default='0/1');p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    cv2.setNumThreads(1);torch.set_num_threads(2)
    protocol=json.loads((OUT/'protocol.json').read_text());fingerprint=digest(OUT/'protocol.json')
    for k,m in protocol['models'].items():assert digest(ROOT/m['path'])==m['sha256']
    for k,m in json.loads((LOW/'protocol.json').read_text())['models'].items():assert protocol['models'][k]['sha256']==m['sha256']
    ids={r['id'] for r in protocol['affected']};videos=[v for v in protocol['sources'] if v['id'] in ids]
    annotations={(a['video_id'],a['track_id']):a for a in json.loads((release.DATA/'annotations.json').read_text())['records']}
    reviewed=json.loads((CACHE/'evaluated-windows.json').read_text())
    i,n=map(int,a.shard.split('/'));videos=videos[i::n]
    if a.limit:videos=videos[:a.limit]
    models=release.Models()
    for v in videos:run(v,models,annotations,reviewed,fingerprint)
    save(OUT/f'replay-complete-{i}.json',dict(complete=True,sources=len(videos),protocol_sha256=fingerprint))


if __name__=='__main__':main()
