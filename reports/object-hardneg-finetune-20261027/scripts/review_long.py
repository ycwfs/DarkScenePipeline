"""Review every retained/changed interaction group, not just favorable examples."""
from collections import defaultdict
import argparse
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw
from build_dataset import OUT, ROOT, NAMES, save


def read_rows(path):
    return [json.loads(line)for line in path.read_text().splitlines()if line.endswith('}')]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--short',action='store_true');args=parser.parse_args()
    folder='short-regression' if args.short else 'long-videos'
    review_folder='short-review' if args.short else 'long-review'
    manifest=json.loads((ROOT/'reports/manypeople-inference-20261025/input-manifest.json').read_text())
    cases=[]
    for number in [1,7,9]:
        dest=OUT/folder/f'new-{number:02}'
        if not (dest/'summary.json').exists():continue
        source=json.loads((dest/'summary.json').read_text())['source']
        baseline=read_rows(dest/'baseline-events.jsonl');candidate=read_rows(dest/'candidate-events.jsonl')
        assert len(baseline)==len(candidate)
        grouped=defaultdict(list)
        for i,(a,b) in enumerate(zip(baseline,candidate)):
            assert (a['frame_index'],a['track_id'],a['history_generation'])==(b['frame_index'],b['track_id'],b['history_generation'])
            if a['label']in ('drink','pick_up')or b['label']in ('drink','pick_up'):
                grouped[(a['track_id'],a['history_generation'],a['action_label'])].append(i)
        groups=[]
        for identity,indices in grouped.items():
            seq=[]
            for i in indices:
                if seq and baseline[i]['timestamp']-baseline[seq[-1]]['timestamp']>1.2:
                    groups.append((identity,seq));seq=[]
                seq.append(i)
            if seq:groups.append((identity,seq))
        for identity,indices in sorted(groups,key=lambda x:baseline[x[1][0]]['timestamp']):
            prefix='short'if args.short else 'new'
            case=dict(id=f'{prefix}-{number:02}-{len(cases):03}',number=number,track_id=identity[0],generation=identity[1],
                action=identity[2],start=baseline[indices[0]]['timestamp'],end=baseline[indices[-1]]['timestamp'],
                baseline_windows=sum(baseline[i]['label']==identity[2]for i in indices),
                candidate_windows=sum(candidate[i]['label']==identity[2]for i in indices),indices=indices,truth=None)
            # Maximum retained candidate evidence, else maximum baseline; expose the hardest remaining error.
            representative=max(indices,key=lambda i:max([e['confidence']for e in candidate[i]['object_prior']['evidence']]+[0])+
                .001*max([e['confidence']for e in baseline[i]['object_prior']['evidence']]+[0]))
            case['representative']=representative
            ar,br=baseline[representative],candidate[representative]
            case['source']=source['absolute_path']
            case['source_offset']=source.get('source_offset',0)
            view=OUT/review_folder;view.mkdir(exist_ok=True)
            cap=cv2.VideoCapture(source['absolute_path']);sheet=Image.new('RGB',(1800,850),(24,24,24));draw=ImageDraw.Draw(sheet)
            draw.text((10,5),f'{case["id"]} T{identity[0]} {identity[2]} {case["start"]:.2f}-{case["end"]:.2f}s | windows {case["baseline_windows"]} -> {case["candidate_windows"]}',fill='white')
            draw.text((10,22),'Raw source crops for visual GT; boxes came from enhanced720 detector. Upper baseline / lower candidate.',fill='white')
            samples=ar['object_prior']['samples']
            if not samples:
                samples=[dict(frame=ar['frame_index'],person_roi=[int(v*1920/720)for v in ar['bbox']],detections=[])]
            for col,s in enumerate(samples):
                cap.set(cv2.CAP_PROP_POS_FRAMES,s['frame']);ok,frame=cap.read();assert ok
                rect=s.get('person_roi')
                if rect is None:continue
                x1,y1,x2,y2=map(int,rect);crop=frame[y1:y2,x1:x2]
                if not crop.size:continue
                im=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB));scale=min(350/im.width,370/im.height)
                im=im.resize((round(im.width*scale),round(im.height*scale)))
                for row,record in enumerate([ar,br]):
                    tile=im.copy();d=ImageDraw.Draw(tile)
                    matched=next((v for v in record['object_prior']['samples']if v['frame']==s['frame']),{})
                    for b in matched.get('detections',[]):
                        bbox=[(b[0]-x1)*scale,(b[1]-y1)*scale,(b[2]-x1)*scale,(b[3]-y1)*scale]
                        color='#ff6855'if int(b[5])in (0,1)else '#33ddff'
                        d.rectangle(bbox,outline=color,width=2);d.text((bbox[0],max(0,bbox[1]-12)),f'{NAMES[int(b[5])]} {b[4]:.3f}',fill=color,stroke_width=1,stroke_fill='black')
                    sheet.paste(tile,(col*360,65+row*390));draw.text((col*360+3,48+row*390),f'frame{s["frame"]} {record["label"]}',fill='white')
            cap.release();sheet.save(view/f'{case["id"]}.jpg',quality=95)
            case['image']=str((view/f'{case["id"]}.jpg').relative_to(OUT));cases.append(case)
    save(OUT/f'{review_folder}-cases.json',cases)
    print([(c['id'],c['start'],c['end'],c['baseline_windows'],c['candidate_windows'])for c in cases])


if __name__=='__main__':main()
