"""Generate resumable object-box drafts on ALL historical onsite sources.

Grounding DINO proposals are explicitly unreviewed, not training labels. Raw source
pixels are cropped before resizing. Brightness correction is only an annotation aid.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'reports/person-finetune-20261023'
sys.path.insert(0, str(ROOT / 'platform/dark-demo-v2'))
from darkpipe.person.runtime import content_height

PROMPT = 'bottle. cup. phone. camera. paper. robot. person.'
NAMES = ['bottle', 'cup', 'phone', 'camera', 'paper', 'portable_object', 'robot_or_exhibit']


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def category(label):
    if 'robot' in label:
        return 6
    for key, value in [('bottle',0),('cup',1),('phone',2),('camera',3),('paper',4)]:
        if key in label:
            return value
    return None


def bright(image):
    # Display/proposal aid only; no generated structures, no training pixels altered.
    value = float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())
    gamma = 1.0 if value >= 75 else .5 if value >= 15 else .32
    lut = np.round((np.arange(256)/255.)**gamma*255).astype(np.uint8)
    return cv2.LUT(image, lut), gamma


def region(box, image_shape, model_shape, margin=.3):
    h,w = image_shape[:2]; mh,mw = model_shape
    b = np.asarray(box) * [w/mw,h/mh,w/mw,h/mh]
    m = (b[2:]-b[:2])*margin
    lo = np.floor(np.maximum(0,b[:2]-m)).astype(int)
    hi = np.ceil(np.minimum([w,h],b[2:]+m)).astype(int)
    return [*map(int,lo),*map(int,hi)]


@torch.inference_mode()
def detect(model, processor, image):
    rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    batch = processor(images=rgb, text=PROMPT, return_tensors='pt').to('cuda:0')
    outputs = model(**batch)
    # Person is an explicit competing class, not a positive target. Do not map a
    # concatenated phrase containing both "person" and "robot" to robot.
    tokenized = processor.tokenizer(PROMPT, return_offsets_mapping=True)
    offset = tokenized['offset_mapping']
    words = ['bottle','cup','phone','camera','paper','robot','person']
    scores = outputs.logits[0].sigmoid()
    per_class = []
    for word in words:
        start = PROMPT.index(word); end = start+len(word)
        positions = [i for i,(a,b) in enumerate(offset) if a < end and b > start]
        per_class.append(scores[:, positions].mean(dim=1))
    best, labels = torch.stack(per_class, dim=1).max(dim=1)
    boxes = outputs.pred_boxes[0].detach().cpu().numpy()
    xyxy = np.concatenate([boxes[:,:2]-boxes[:,2:]/2, boxes[:,:2]+boxes[:,2:]/2],axis=1)
    xyxy *= [image.shape[1],image.shape[0]]*2
    rows = []
    for label_index, score, box in zip(labels.tolist(),best.tolist(),xyxy):
        label=words[label_index]
        if score < .22 or label == 'person': continue
        cls = category(label)
        if cls is not None:
            b = np.clip(box, [0]*4, [image.shape[1],image.shape[0]]*2)
            if np.all(b[2:]-b[:2] >= 2):
                rows.append(dict(class_id=cls, class_name=NAMES[cls], text_label=label,
                                 confidence=float(score), bbox=b.tolist(), status='unreviewed'))
    keep=[]
    for cls in range(len(NAMES)):
        subset=[r for r in rows if r['class_id']==cls]
        if not subset: continue
        xywh=np.array([r['bbox'] for r in subset]);xywh[:,2:]-=xywh[:,:2]
        indices=cv2.dnn.NMSBoxes(xywh.tolist(),[r['confidence']for r in subset],.22,.5)
        keep += [subset[i] for i in np.asarray(indices,dtype=int).reshape(-1)]
    return keep


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--only-id')
    args=parser.parse_args()
    torch.set_num_threads(2); cv2.setNumThreads(1)
    inventory=json.loads((OLD/'source-inventory.json').read_text())['videos']
    splits={r['video_id']:r for r in json.loads((OLD/'splits.json').read_text())['records']}
    inventory.sort(key=lambda r:(r['video_label'] not in ('drink','pick_up'),r['video_label'],r['source']))
    if args.only_id: inventory=[r for r in inventory if r['id']==args.only_id]
    if args.limit: inventory=inventory[:args.limit]
    processor=AutoProcessor.from_pretrained(OUT/'proposal-model',local_files_only=True)
    model=AutoModelForZeroShotObjectDetection.from_pretrained(OUT/'proposal-model',local_files_only=True).to('cuda:0').eval()
    started=time.monotonic()
    for index,source in enumerate(inventory,1):
        dest=OUT/'drafts'/source['id']; target=dest/'proposals.json'
        if target.exists():
            continue
        dest.mkdir(parents=True,exist_ok=True)
        cache=json.loads((OLD/'data'/source['id']/'manifest.json').read_text())
        all_meta={r['frame']:r for r in cache['frames']}
        selected=sorted(set(int(round((source['frames']-1)*f)) for f in (.2,.5,.8)))
        cap=cv2.VideoCapture(str(ROOT/source['source'])); records=[]
        for fi in selected:
            cap.set(cv2.CAP_PROP_POS_FRAMES,fi);ok,native=cap.read()
            if not ok: raise RuntimeError((source['source'],fi))
            if source['processed']: native=native[:content_height(native.shape[0])]
            h,w=native.shape[:2]; scale=min(1.,720/max(h,w))
            mh,mw=(int(round(h*scale))//2)*2,(int(round(w*scale))//2)*2
            meta=all_meta[fi]
            cv2.imwrite(str(dest/f'f{fi:06d}_raw.png'),native)
            views=[('full', [0,0,w,h], None)]
            views += [(f'T{t["track_id"]:03d}',region(t['bbox'],native.shape,(mh,mw)),t['track_id'])
                      for t in meta['tracks']]
            for name,bounds,tid in views:
                if name=='full' and fi!=selected[len(selected)//2]: continue
                x1,y1,x2,y2=bounds
                crop=native[y1:y2,x1:x2]
                if not crop.size: continue
                adjusted,gamma=bright(crop)
                filename=f'f{fi:06d}_{name}.png'
                cv2.imwrite(str(dest/filename),adjusted)
                proposals=detect(model,processor,adjusted)
                records.append(dict(id=f'{source["id"]}_{fi:06d}_{name}',frame=fi,view=name,
                    timestamp=fi/source['fps'],track_id=tid,raw_file=f'f{fi:06d}_raw.png',
                    image=filename,crop_xyxy_raw=bounds,width=crop.shape[1],height=crop.shape[0],
                    brightness_gamma=gamma,luma=meta['luma'],proposals=proposals,
                    status='unreviewed; not object ground truth'))
        cap.release()
        save(target,dict(video=source,split=splits[source['id']]['split'],
                         group=splits[source['id']]['group'],records=records,
                         proposal_model='IDEA-Research/grounding-dino-tiny',prompt=PROMPT,
                         status='assistant must review before training',training_allowed=False))
        save(OUT/'proposal-progress.json',dict(last_video=source['source'],index=index,
             total=len(inventory),completed_sources=len(list((OUT/'drafts').glob('*/proposals.json'))),
             elapsed_seconds=time.monotonic()-started))
        print(f'[{index}/{len(inventory)}] {source["source"]}: {len(records)} views, '
              f'{sum(len(r["proposals"]) for r in records)} proposals',flush=True)


if __name__=='__main__':main()
