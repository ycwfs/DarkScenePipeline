"""Matched site crops, deployment context and original-class confusion diagnostics."""
from collections import Counter
import json

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

from common import MAPPING, OUT, PREVIOUS, ROOT, save
from export import letterbox, iou


def session(path):
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    s=ort.InferenceSession(str(path),sess_options=opts,providers=[('CUDAExecutionProvider',{'device_id':0})])
    assert s.get_providers()[0]=='CUDAExecutionProvider'
    return s


def predict(s,image):
    x,scale,left,top=letterbox(image)
    p=s.run(None,{s.get_inputs()[0].name:x})[0][0].copy()
    p[:,[0,2]]=(p[:,[0,2]]-left)/scale;p[:,[1,3]]=(p[:,[1,3]]-top)/scale
    p[:,:4]=np.clip(p[:,:4],0,[image.shape[1],image.shape[0]]*2)
    return p[p[:,4]>=.01]


def score(pred,boxes):
    counts={}
    for c in [39,41,67,73]:
        gt=[b['bbox']for b in boxes if MAPPING[b['class_id']]==c]
        pp=sorted([p for p in pred if int(p[5])==c and p[4]>=.581],key=lambda p:-p[4]);used=set()
        for p in pp:
            candidates=[(iou(p[:4],np.array(b)),j)for j,b in enumerate(gt)if j not in used]
            if candidates and max(candidates)[0]>=.5:used.add(max(candidates)[1])
        counts[str(c)]=dict(tp=len(used),fp=len(pp)-len(used),fn=len(gt)-len(used))
    confusion=[]
    for b in boxes:
        if b['class_id']not in [2,3,6,7]:continue
        confused=[p for p in pred if int(p[5])in [39,41]and iou(p[:4],np.array(b['bbox']))>=.3]
        person=[p for p in pred if int(p[5])==0 and iou(p[:4],np.array(b['bbox']))>=.4]
        confusion.append(dict(source_class=b['class_id'],container_max=max([float(p[4])for p in confused]+[0.]),
            as_container=any(p[4]>=.581 for p in confused),person_max=max([float(p[4])for p in person]+[0.]),
            as_person=any(p[4]>=.35 for p in person)))
    return dict(counts=counts,confusion=confusion,predictions=pred.tolist())


def legacy_to_coco(p):
    p=p.copy()
    for b in p:b[5]=MAPPING.get(int(b[5])) if MAPPING.get(int(b[5])) is not None else 99
    return p


def main():
    cv2.setNumThreads(1)
    names=json.loads((OUT/'classes.json').read_text())
    sessions=dict(original80=session(OUT/'models/yolo26m_coco80_original.onnx'),
        current7=session(ROOT/'reports/object-prior-finetune-20261025/models/yolo26m_objects_candidate.onnx'),
        candidate80=session(OUT/'models/yolo26m_coco80_candidate.onnx'))
    records=[]
    original=json.loads((PREVIOUS/'dataset-manifest.json').read_text())['records']
    for r in original:
        if r['split']not in ['val','test']:continue
        records.append(dict(id=r['id'],split=r['split'],source=r['source'],group=r['group'],
            cohort=r['cohort'],context='review_crop',image=r['dataset_image'],boxes=r['boxes']))
    cache=ROOT/'reports/object-model-ablation-20261026'
    annotations=json.loads((PREVIOUS/'reviewed-labels.json').read_text())['records']
    for r in annotations:
        if not r.get('included')or'small_enhanced'not in r['variants']:continue
        v=r['variants']['small_enhanced'];origin=np.array(v['origin']*2);scale=np.array(v['scale']*2)
        boxes=[dict(class_id=b['class_id'],bbox=((np.array(b['bbox'])-origin)/scale).tolist())for b in r['reviewed_global_boxes']]
        records.append(dict(id=f'context_{r["index"]:03}',index=r['index'],split=r['split'],source=r['source'],
            group=f'new-{r["number"]:02}',cohort='new',context='deployment_roi',image=str(cache/v['path']),
            boxes=boxes,valid_region=((np.array(r['reviewed_global_region'])-origin)/scale).tolist()))
    for r in records:
        image=cv2.imread(r['image']);assert image is not None;r['variants']={}
        for name,s in sessions.items():
            p=predict(s,image)
            if name=='current7':p=legacy_to_coco(p)
            if 'valid_region'in r:
                region=np.array(r['valid_region']);centers=(p[:,:2]+p[:,2:4])/2
                p=p[((centers>=region[:2])&(centers<=region[2:])).all(1)]
            r['variants'][name]=score(p,r['boxes'])
    summary={}
    for context in ['review_crop','deployment_roi']:
        for split in ['train','val','test']:
            rs=[r for r in records if r['context']==context and r['split']==split]
            if not rs:continue
            summary[f'{context}_{split}']=dict(images=len(rs),sources=len({r['source']for r in rs}),models={})
            for name in sessions:
                ct={names[c]:{k:sum(r['variants'][name]['counts'][str(c)][k]for r in rs)for k in ['tp','fp','fn']}for c in [39,41,67,73]}
                errors={str(c):dict(total=sum(x['source_class']==c for r in rs for x in r['variants'][name]['confusion']),
                    container_errors=sum(x['source_class']==c and x['as_container']for r in rs for x in r['variants'][name]['confusion']),
                    person_errors=sum(x['source_class']==c and x['as_person']for r in rs for x in r['variants'][name]['confusion']))for c in [2,3,6,7]}
                summary[f'{context}_{split}']['models'][name]=dict(classes=ct,confusion=errors)
    save(OUT/'object-evaluation.json',dict(records=records,summary=summary,threshold=.581,
        warning='Reviewed target categories only. Do not score unlabeled generic objects as false positives. Remapped camera/mic/paper labels are project semantics. Train is fitting diagnostic.'))
    directory=OUT/'comparisons';directory.mkdir(exist_ok=True)
    for r in records:
        if r['context']!='deployment_roi'or r['split']!='test':continue
        raw=Image.open(r['image']).convert('RGB');scale=min(450/raw.width,400/raw.height)
        canvas=Image.new('RGB',(1350,450),'#181818');draw=ImageDraw.Draw(canvas)
        for j,name in enumerate(sessions):
            im=raw.resize((round(raw.width*scale),round(raw.height*scale)));d=ImageDraw.Draw(im)
            for p in r['variants'][name]['predictions']:
                if p[4]<.581:continue
                c=int(p[5]);box=[v*scale for v in p[:4]];color='#ff7255'if c in [39,41]else'#35dfff'
                d.rectangle(box,outline=color,width=2);d.text((box[0],max(0,box[1]-12)),f'{names[c] if c<80 else "legacy-other"} {p[4]:.2f}',fill=color,stroke_width=1,stroke_fill='black')
            canvas.paste(im,(j*450,35));draw.text((j*450+6,8),f'{name} | {r["id"]}',fill='white')
        canvas.save(directory/f'{r["id"]}.jpg',quality=94)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
