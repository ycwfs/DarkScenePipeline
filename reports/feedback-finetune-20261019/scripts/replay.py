"""Paired deployed-model replay; cache unchanged input/pose evidence for checkpoint comparison."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'reports/feedback-finetune-20261019'
OP = ROOT/'platform/dark-demo-v2'
sys.path.insert(0, str(OP))
from darkpipe.constants import BEHAVIORS, IN_MEAN, IN_STD
from darkpipe.stages.recognize import BehaviorRecognizer, reject_probs, frame_luma
from darkpipe.stages.downscale import DownscaleStage
from darkpipe.stages.enhance_retinexformer import RetinexformerStage
from darkpipe.stages.skeleton_aux import SkeletonAux
from prepare_data import windows


def predict(p: np.ndarray, tau: float = .75) -> str:
    return BEHAVIORS[int(reject_probs(p, BEHAVIORS, tau).argmax())]


def summarize(rows: list) -> dict:
    result = {}
    for suite in sorted({r['suite'] for r in rows}):
        sub = [r for r in rows if r['suite']==suite]
        cm = np.zeros((10,10), np.int64)
        for r in sub:
            cm[BEHAVIORS.index(r['expect']), BEHAVIORS.index(r['label'])] += 1
        tp=np.diag(cm)
        f1=2*tp/np.maximum(cm.sum(0)+cm.sum(1),1)
        videos=defaultdict(list)
        for r in sub:
            videos[r['source']].append(r)
        result[suite] = dict(windows=len(sub), videos=len(videos), correct=int(tp.sum()),
                            top1=float(tp.sum()/len(sub)), macro_f1=float(f1[cm.sum(1)>0].mean()),
                            fired=sum(r['label']!='other' for r in sub),
                            videos_fired=sum(any(r['label']!='other' for r in v) for v in videos.values()),
                            rgb_fired=sum(r['rgb_label']!='other' for r in sub),
                            confusion=cm.tolist(), incorrect=[r['name'] for r in sub if r['label']!=r['expect']])
    return result


def plan(group: str) -> dict:
    planned=defaultdict(list)
    src=ROOT/'real-actiondata-0909'
    if group=='regression':
        val=ROOT.parent/'compare/behavior/cache/realact_rf_v10_full/val'
        for p in sorted(val.glob('*/*.npy')):
            stem,start=p.stem.rsplit('_',1)
            path=src/(stem+'.mp4')
            if not path.exists(): path=src/'other'/(stem+'.mp4')
            planned[path].append(dict(suite='A',name=p.stem,expect=p.parent.name,
                                      lo=int(start),hi=int(start)+32,processed=False))
        groups=[('B',p,'other') for p in sorted((src/'other').glob('other*.mp4'))]
        groups += [('C',src/(s+'.mp4'),l) for s,l in
                   [('挥手22','wave'),('捡东西2','pick_up'),('追逐16','chase'),('喝水16','drink')]]
        groups += [('P',p,'other') for p in sorted((ROOT/'online-negative-sample/20261012_083305_78').glob('*/*.mp4'))]
        for suite,path,expect in groups:
            cap=cv2.VideoCapture(str(path))
            assert cap.isOpened(),path
            n,fps=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            for tag,lo,hi in windows(n,fps):
                planned[path].append(dict(suite=suite,name=f'{path.stem}@{tag}',expect=expect,
                                          lo=lo,hi=hi,processed=suite=='P'))
    m=json.loads((OUT/'data-manifest.json').read_text())
    target='train' if group=='regression' else 'holdout'
    for row in m['windows']:
        if row['split']!=target: continue
        path=OUT/'data/videos'/row['source']
        planned[path].append(dict(suite='Qtrain' if target=='train' else 'Qholdout',
                                  name=f'{path.stem}@{row["tag"]}',expect='other',lo=row['lo'],hi=row['hi'],
                                  processed=True,session=row['session']))
    return planned


def build(group: str) -> None:
    planned=plan(group)
    counts=Counter(r['suite'] for rows in planned.values() for r in rows)
    print('[plan]',dict(counts),'videos',len(planned),flush=True)
    if group=='regression':
        assert all(counts[s]==n for s,n in dict(A=177,B=365,C=18,P=460,Qtrain=175).items()),counts
    else: assert counts['Qholdout']==68
    ds=DownscaleStage(720)
    enhancer=RetinexformerStage(str(OP/'ckpts'),chunk=1)
    enhancer.load('cuda:0')
    rec=BehaviorRecognizer(str(OP/'ckpts'),stride=32,reject_tau=.75,skel_fusion_alpha=.7,lum_gate=75)
    rec.ckpt=str(OUT/'before/platform/dark-demo-v2/ckpts/videomamba_t_behavior_32f_realactall_full_pn.pth')
    rec.load('cuda:0')
    assert rec.skel is not None
    rec.warmup()
    captured={}
    original_infer,original_fuse=rec._infer,rec.skel.fuse
    def infer(arr):
        p=original_infer(arr)
        captured['rgb']=p.copy()
        captured['input']=arr.copy()
        return p
    def fuse(pr,ps,pres):
        captured.update(skeleton=ps.copy(),presence=float(pres))
        return original_fuse(pr,ps,pres)
    rec._infer,rec.skel.fuse=infer,fuse
    dest=OUT/'evaluation'/group
    dest.mkdir(parents=True,exist_ok=True)
    cache=dest/'inputs'
    cache.mkdir(exist_ok=True)
    records=[]
    started=time.monotonic()
    with (dest/'baseline.jsonl').open('w') as stream:
        for num,(path,items) in enumerate(planned.items(),1):
            selected=sorted({int(i) for r in items for i in np.linspace(r['lo'],r['hi']-1,32).round().astype(int)})
            wanted=set(selected)
            raw={}
            cap=cv2.VideoCapture(str(path))
            assert cap.isOpened(),path
            for index in range(selected[-1]+1):
                ok,frame=cap.read()
                assert ok,(path,index)
                if index not in wanted: continue
                if items[0]['processed']:
                    h=frame.shape[0]
                    content=[ch for ch in range(h-180,h) if ch%2==0 and
                             ch+max(48,round(.08*ch))+(max(48,round(.08*ch))&1)==h]
                    assert len(content)==1,(path,h,content)
                    frame=frame[:content[0]]
                raw[index]=ds([frame])[0]
            cap.release()
            processed=items[0]['processed']
            enhanced=raw if processed else dict(zip(selected,enhancer([raw[i] for i in selected])))
            for item in items:
                indices=np.linspace(item['lo'],item['hi']-1,32).round().astype(int)
                rec.reset()
                captured.clear()
                for i,index in enumerate(indices):
                    ev=rec.push(enhanced[int(index)],i,i/32,
                                raw_frame=None if processed else raw[int(index)])
                assert ev is not None
                pixels=np.stack([cv2.cvtColor(cv2.resize(enhanced[int(i)],(400,224),interpolation=cv2.INTER_AREA),
                                              cv2.COLOR_BGR2RGB) for i in indices])
                normalized=(pixels.astype(np.float32)/255.-IN_MEAN)/IN_STD
                assert np.array_equal(normalized,captured['input']),item['name']
                p=captured['rgb']
                if 'skeleton' in captured:
                    p=original_fuse(p,captured['skeleton'],captured['presence'])
                assert rec.display(predict(p))==ev.label
                cachepath=cache/f'{len(records):05d}.npy'
                np.save(cachepath,pixels)
                row=dict(item,source=str(path.relative_to(ROOT)),input=str(cachepath),
                         rgb=captured['rgb'].tolist(),skeleton=captured.get('skeleton',np.array([])).tolist(),
                         presence=captured.get('presence'),decision=p.tolist(),label=predict(p),
                         rgb_label=predict(captured['rgb']),
                         raw_luma=None if processed else float(np.mean([frame_luma(raw[int(i)]) for i in indices])),
                         note='Processed replay: label bar removed, no re-enhancement; raw-scene gate unavailable' if processed else rec.prior_note)
                records.append(row)
                stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                stream.flush()
            print(f'[build] {num}/{len(planned)} videos {len(records)} windows {time.monotonic()-started:.0f}s',flush=True)
    summary=summarize(records)
    (dest/'baseline-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print('[summary]',json.dumps(summary,ensure_ascii=False),flush=True)


def evaluate(checkpoint: Path, group: str) -> None:
    dest=OUT/'evaluation'/group
    rows=[json.loads(line) for line in (dest/'baseline.jsonl').read_text().splitlines()]
    rec=BehaviorRecognizer(str(OP/'ckpts'),skel_fusion_alpha=1.)
    rec.ckpt=str(checkpoint)
    rec.load('cuda:0')
    fuser=SkeletonAux(str(OP/'ckpts'),'cuda:0',.7,'soft')
    results=[]
    for i,row in enumerate(rows):
        pixels=np.load(row['input'])
        arr=(pixels.astype(np.float32)/255.-IN_MEAN)/IN_STD
        with torch.inference_mode():
            pr=rec._infer(arr)
        p=fuser.fuse(pr,np.array(row['skeleton'],np.float32),row['presence']) if row['skeleton'] else pr
        results.append(dict(row,rgb=pr.tolist(),decision=p.tolist(),label=predict(p),rgb_label=predict(pr)))
        if (i+1)%200==0: print('[eval]',checkpoint.name,i+1,len(rows),flush=True)
    name=checkpoint.stem
    with (dest/(name+'.jsonl')).open('w') as stream:
        for row in results: stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    summary=summarize(results)
    (dest/(name+'-summary.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print('[summary]',name,json.dumps(summary,ensure_ascii=False),flush=True)


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--build',action='store_true')
    ap.add_argument('--group',choices=['regression','holdout'],default='regression')
    ap.add_argument('--checkpoint',type=Path)
    args=ap.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    if args.build: build(args.group)
    else:
        assert args.checkpoint is not None
        evaluate(args.checkpoint,args.group)


if __name__=='__main__': main()
