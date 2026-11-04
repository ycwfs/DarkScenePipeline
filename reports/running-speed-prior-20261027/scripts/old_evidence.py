"""Exact historical pose replay; test requires a selection lock."""
import argparse
from collections import defaultdict, Counter
import json
from types import SimpleNamespace
import numpy as np
from common import OUT, ROOT, WAVE, save
from running_prior import speed_evidence


def shape(v):
    h,w=v['height'],v['width']
    if v['processed']:
        candidates=[x for x in range(h-180,h) if x%2==0 and x+max(48,round(.08*x))+(max(48,round(.08*x))&1)==h]
        assert len(candidates)==1
        h=candidates[0]
    if max(h,w)>720:
        ratio=720/max(h,w);h,w=round(h*ratio)//2*2,round(w*ratio)//2*2
    return h,w


def geometry(v, rows):
    prior=ROOT/'reports/four-parameter-tuning-20261026'
    cache=json.loads((ROOT/v['cache']).read_text())
    replay_file=prior/'replay'/f'{v["id"]}.json'
    replay=json.loads(replay_file.read_text()) if replay_file.exists() else None
    frames=json.loads((ROOT/'reports/luma-gate15-20261024/data'/v['id']/'result.json').read_text())['frames'] if replay else cache['frames']
    tracks=replay['tracks']['35'] if replay else [f['tracks']['30'] for f in frames]
    events=replay['events']['35'] if replay else cache['events']['30']
    h,w=shape(v)
    for row in rows:
        e=events[row['event_index']];assert e['source_frames']==row['source_frames']
        samples=[]
        for fi in e['source_frames']:
            meta=frames[fi];actors=[t for t in tracks[fi] if t['track_id']==e['track_id']]
            assert len(actors)==1
            found=[i for i,b in enumerate(meta['boxes']) if np.allclose(b[:4],actors[0]['bbox'],atol=1e-4,rtol=0)]
            assert len(found)==1
            pose=meta['poses'][found[0]]
            p=np.zeros((17,3),np.float32) if pose is None else np.array(pose,np.float32)
            p[:,:2]/=[w,h]
            if meta['luma'] is not None and meta['luma']<35:p[:]=0
            samples.append(SimpleNamespace(frame_index=fi,timestamp=meta['timestamp'],pose=p,luma=meta['luma'],
                person_box=actors[0]['bbox'],frame_shape=(h,w)))
        yield row,samples


def main():
    p=argparse.ArgumentParser();p.add_argument('--test',action='store_true');a=p.parse_args()
    if a.test:assert (OUT/'selection.json').exists(),'Test report only after threshold lock'
    rows=json.loads((ROOT/'reports/threshold-tuning-20261026/windows.json').read_text())
    pred=json.loads((ROOT/'reports/four-parameter-tuning-20261026/predictions.json').read_text())
    assert [r['id'] for r in rows]==pred['ids']
    sources={v['id']:v for v in json.loads((WAVE/'protocol.json').read_text())['old_sources'] if v['scope']!='new_multipeople'}
    grouped=defaultdict(list)
    for r,i in zip(rows,pred['predictions']['recommended']):
        if r['source_id'] not in sources or (r['split']=='test')!=a.test:continue
        label=pred['labels'][i]
        if r['truth']=='chase' or label=='chase':grouped[r['source_id']].append(dict(r,baseline=label))
    records=[]
    for sid,rr in grouped.items():
        for r,samples in geometry(sources[sid],rr):
            e=speed_evidence(samples,r['raw_luma'],0,35)
            records.append(dict(id=r['id'],source_id=sid,source=sources[sid]['source'],split=r['split'],group=r['group'],
                truth=r['truth'],baseline=r['baseline'],raw_luma=r['raw_luma'],track_id=r['track_id'],timestamp=r['timestamp'],evidence=e))
    dest=OUT/('old-test-evidence.json' if a.test else 'old-calibration-evidence.json');save(dest,records)
    for split in sorted({r['split']for r in records}):
        print(split,dict(Counter((r['truth'],r['baseline'],r['evidence']['speed']is not None)for r in records if r['split']==split)))

if __name__=='__main__':main()
