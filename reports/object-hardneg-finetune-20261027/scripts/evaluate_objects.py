"""Paired fixed-threshold evaluation in reviewed heldout regions."""
from collections import Counter
import csv
import json
import sys

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
from build_dataset import OUT, ROOT, OLD, NAMES, save
from select_export import letterbox, iou


def session(path):
    opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1
    s=ort.InferenceSession(str(path),sess_options=opt,providers=[('CUDAExecutionProvider',{'device_id':0})])
    assert s.get_providers()[0]=='CUDAExecutionProvider'
    return s


def predict(s,frame):
    x,ratio,left,top=letterbox(frame)
    pred=s.run(None,{s.get_inputs()[0].name:x})[0][0].copy()
    pred[:,[0,2]]=(pred[:,[0,2]]-left)/ratio
    pred[:,[1,3]]=(pred[:,[1,3]]-top)/ratio
    pred[:,:4]=np.clip(pred[:,:4],0,[frame.shape[1],frame.shape[0]]*2)
    return pred[pred[:,4]>=.01]


def counts(pred,boxes):
    result={}
    for c,name in enumerate(NAMES):
        gt=[b['bbox']for b in boxes if b['class_id']==c]
        pp=sorted([p for p in pred if int(p[5])==c and p[4]>=.581],key=lambda p:-p[4])
        used=set()
        for p in pp:
            candidates=[(iou(p[:4],np.array(g)),i)for i,g in enumerate(gt)if i not in used]
            if candidates and max(candidates)[0]>=.5:used.add(max(candidates)[1])
        result[name]=dict(tp=len(used),fp=len(pp)-len(used),fn=len(gt)-len(used))
    confusion=[]
    for b in boxes:
        if b['class_id'] not in (2,3,7):continue
        overlap=[p for p in pred if int(p[5])in (0,1)and iou(p[:4],np.array(b['bbox']))>=.3]
        confusion.append(dict(truth=NAMES[b['class_id']],max_container_conf=max([float(p[4])for p in overlap]+[0.]),
            mistaken_container=any(p[4]>=.581 for p in overlap)))
    return result,confusion


def main():
    cv2.setNumThreads(1)
    sessions=dict(baseline=session(OLD/'models/yolo26m_objects_candidate.onnx'),
                  candidate=session(OUT/'models/yolo26m_objects_candidate.onnx'))
    records=json.loads((OUT/'dataset-manifest.json').read_text())['records']
    rows=[]
    for r in records:
        if r['split']not in ('val','test'):continue
        frame=cv2.imread(r['dataset_image']);variants={}
        for name,s in sessions.items():
            pred=predict(s,frame);ct,confusion=counts(pred,r['boxes'])
            variants[name]=dict(predictions=pred.tolist(),counts=ct,confusion=confusion)
        rows.append(dict(r,variants=variants))
    summary={}
    for split in ('val','test'):
        for cohort in ('old','new'):
            subset=[r for r in rows if r['split']==split and r['cohort']==cohort]
            info=dict(images=len(subset),groups=len({r['group']for r in subset}),models={})
            for name in sessions:
                info['models'][name]={c:{m:sum(r['variants'][name]['counts'][c][m]for r in subset)for m in ('tp','fp','fn')}for c in NAMES}
                info['models'][name]['confused_as_container']={c:dict(
                    errors=sum(b['mistaken_container']for r in subset for b in r['variants'][name]['confusion']if b['truth']==c),
                    total=sum(1 for r in subset for b in r['variants'][name]['confusion']if b['truth']==c))for c in ('phone','camera','microphone')}
            summary[f'{cohort}_{split}']=info
    save(OUT/'object-evaluation.json',dict(records=rows,summary=summary,confidence=.581,iou=.5,
        confusion_iou=.3,limitations='Mined auxiliary visually reviewed crop GT, not an unbiased full-scene sample. One seed.'))
    selected=[r for r in rows if r['split']=='test'and r['cohort']=='new']
    visual=OUT/'object-comparisons';visual.mkdir(exist_ok=True)
    for r in selected:
        raw=Image.open(r['dataset_image']).convert('RGB');w,h=raw.size;factor=min(500/w,450/h)
        size=(round(w*factor),round(h*factor));canvas=Image.new('RGB',(1000,500),(24,24,24));d=ImageDraw.Draw(canvas)
        for j,mode in enumerate(sessions):
            im=raw.resize(size);draw=ImageDraw.Draw(im)
            for box in r['boxes']:
                b=[v*factor for v in box['bbox']];draw.rectangle(b,outline='white',width=1)
            for pred in r['variants'][mode]['predictions']:
                if pred[4]<.581:continue
                b=[v*factor for v in pred[:4]];c=int(pred[5]);color='#ff7555'if c in (0,1)else '#36e0ff'
                draw.rectangle(b,outline=color,width=2);draw.text((b[0],max(0,b[1]-12)),f'{NAMES[c]} {pred[4]:.2f}',fill=color,stroke_width=1,stroke_fill='black')
            canvas.paste(im,(j*500+(500-im.width)//2,40));d.text((j*500+10,8),f'{mode} | {r["id"]} | {r["group"]}',fill='white')
        canvas.save(visual/f'{r["id"]}.jpg',quality=95)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
