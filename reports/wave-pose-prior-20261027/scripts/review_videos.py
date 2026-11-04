"""All wave groups, including retained predictions; raw context and actual joint evidence."""
from collections import defaultdict
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from common import OUT, ROOT, save


def read_rows(path):
    rows=[]
    for line in path.read_text().splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:continue
    return rows


def source_for(directory):
    if (directory/'summary.json').exists():return json.loads((directory/'summary.json').read_text())['source']
    n=int(directory.name[-2:])
    return next(v for v in json.loads((ROOT/'reports/manypeople-inference-20261025/input-manifest.json').read_text())if v['number']==n)


def main():
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');args=p.parse_args()
    view=OUT/'wave-review';view.mkdir(exist_ok=True);cases=[]
    for scope in ['long-videos','short-regression','wave-shorts','old-videos']:
        for directory in sorted((OUT/scope).glob('*')):
            if not (directory/'summary.json').exists()and (not args.partial or scope!='long-videos'):continue
            if not (directory/'candidate-events.jsonl').exists():continue
            source=source_for(directory)
            baseline=read_rows(directory/'baseline-events.jsonl');candidate=read_rows(directory/'candidate-events.jsonl')
            if not args.partial:assert len(baseline)==len(candidate)
            indices=defaultdict(list)
            for i,(a,b)in enumerate(zip(baseline,candidate)):
                assert (a['frame_index'],a['track_id'])==(b['frame_index'],b['track_id'])
                if a['label']=='wave':indices[(a['track_id'],a['history_generation'])].append(i)
            groups=[]
            for identity,items in indices.items():
                group=[]
                for i in items:
                    if group and baseline[i]['timestamp']-baseline[group[-1]]['timestamp']>1.2:
                        groups.append((identity,group));group=[]
                    group.append(i)
                if group:groups.append((identity,group))
            cap=cv2.VideoCapture(source['absolute_path']);assert cap.isOpened()
            width,height=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            for identity,items in sorted(groups,key=lambda group:baseline[group[1][0]]['timestamp']):
                first=baseline[items[0]];rep=items[0];row=candidate[rep];ev=row['wave_prior']
                case_id=f'{scope}-{directory.name}-T{identity[0]}-G{identity[1]}-F{first["frame_index"]}'
                sheet=Image.new('RGB',(1800,750),'#181c22');draw=ImageDraw.Draw(sheet)
                after=sum(candidate[i]['label']=='wave'for i in items)
                draw.text((10,5),f'{case_id} | wave windows {len(items)} -> {after} | {first["timestamp"]:.2f}s',fill='white')
                draw.text((10,23),f'FIRST WINDOW: left/right raised={ev.get("counts")} reliable={ev.get("reliable_counts")} {ev["reason"]}',fill='white')
                entries=ev.get('samples',[])
                if not entries:
                    # Dark-gate bypass deliberately has no claimed keypoint evidence.
                    entries=[dict(frame=f,timestamp=f/source['fps'],shoulders_wrists=[])for f in sorted(set(row['source_frames']))]
                picked=np.linspace(0,len(entries)-1,min(6,len(entries))).round().astype(int)
                scene=None
                for col,j in enumerate(picked):
                    e=entries[j];cap.set(cv2.CAP_PROP_POS_FRAMES,e['frame']);ok,frame=cap.read();assert ok
                    if scene is None:scene=frame.copy()
                    ratio=720/max(width,height)
                    mw,mh=int(round(width*ratio))//2*2,int(round(height*ratio))//2*2
                    bb=np.array(row['bbox'])*[width/mw,height/mh,width/mw,height/mh]
                    size=bb[2:]-bb[:2];lo=np.maximum(0,np.floor(bb[:2]-.25*size)).astype(int);hi=np.minimum([width,height],np.ceil(bb[2:]+.25*size)).astype(int)
                    points=e['shoulders_wrists']
                    for k,point in enumerate(points):
                        if any(v is None for v in point)or point[2]<.5:continue
                        xy=np.rint(np.array(point[:2])*[width,height]).astype(int)
                        cv2.circle(frame,tuple(xy),5,(0,255,255)if k<2 else (255,150,0)if k==2 else(20,80,255),-1)
                    crop=frame[lo[1]:hi[1],lo[0]:hi[0]]
                    image=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB));image.thumbnail((296,430))
                    sheet.paste(image,(col*300,78))
                    draw.text((col*300+3,47),f'{e["timestamp"]:.2f}s F{e["frame"]}',fill='white')
                    draw.text((col*300+3,61),f'above L/R {e.get("above_both", "gate bypass")}',fill='white')
                if scene is not None:
                    cv2.rectangle(scene,tuple(lo),tuple(hi),(0,255,255),3)
                    context=Image.fromarray(cv2.cvtColor(scene,cv2.COLOR_BGR2RGB));context.thumbnail((700,230));sheet.paste(context,(15,515))
                draw.text((730,530),'Yellow: both shoulders; cyan: left wrist; red: right wrist.',fill='white')
                draw.text((730,550),'Raw source pixels shown; joints from actual enhanced-frame inference.',fill='white')
                draw.text((730,570),'Raised hand alone does not establish waving. Review full context.',fill='white')
                sheet.save(view/f'{case_id}.jpg',quality=95)
                cases.append(dict(id=case_id,scope=scope,directory=str(directory.relative_to(OUT)),source=source,
                    track_id=identity[0],history_generation=identity[1],start=first['timestamp'],end=baseline[items[-1]]['timestamp'],
                    baseline_windows=len(items),candidate_windows=after,indices=items,representative=rep,
                    counts=ev.get('counts'),reliable=ev.get('reliable_counts'),reason=ev['reason'],
                    variants={setting:sum(candidate[i].get('wave_sweep',{}).get(setting,candidate[i]['label'])=='wave'for i in items)
                              for setting in ['1_total','3_total','5_total','8_total','3_consecutive','5_consecutive','8_consecutive']},
                    image=str((view/f'{case_id}.jpg').relative_to(OUT))))
            cap.release()
    save(OUT/('partial-wave-cases.json'if args.partial else 'wave-cases.json'),cases)
    print([(c['id'],c['baseline_windows'],c['candidate_windows'],c['counts'])for c in cases])


if __name__=='__main__':main()
