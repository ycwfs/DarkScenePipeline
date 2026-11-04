"""Use original deployment crop context; score only visually reviewed valid regions."""
import json
from collections import Counter

import cv2
import numpy as np
from build_dataset import OUT, CACHE, OLD, NAMES, save
from evaluate_objects import session, predict, counts


def main():
    cv2.setNumThreads(1)
    sessions=dict(baseline=session(OLD/'models/yolo26m_objects_candidate.onnx'),
        candidate=session(OUT/'models/yolo26m_objects_candidate.onnx'))
    annotations=json.loads((OUT/'reviewed-labels.json').read_text())['records']
    rows=[]
    for r in annotations:
        if not r.get('included')or 'small_enhanced'not in r['variants']:continue
        v=r['variants']['small_enhanced'];frame=cv2.imread(str(CACHE/v['path']));assert frame is not None
        origin,scale=np.array(v['origin']*2),np.array(v['scale']*2)
        roi=(np.array(r['reviewed_global_region'])-origin)/scale
        boxes=[dict(class_id=b['class_id'],bbox=((np.array(b['bbox'])-origin)/scale).tolist())for b in r['reviewed_global_boxes']]
        variants={}
        for name,s in sessions.items():
            pred=predict(s,frame);center=(pred[:,:2]+pred[:,2:4])/2
            pred=pred[((center>=roi[:2])&(center<=roi[2:])).all(1)]
            ct,confusion=counts(pred,boxes)
            variants[name]=dict(predictions=pred.tolist(),counts=ct,confusion=confusion)
        rows.append(dict(index=r['index'],number=r['number'],split=r['split'],frame=r['frame'],
            key=r['key'],source=r['source'],image=str(CACHE/v['path']),valid_region=roi.tolist(),boxes=boxes,variants=variants))
    summary={}
    for split in ('train','val','test'):
        subset=[r for r in rows if r['split']==split]
        summary[split]=dict(images=len(subset),sources=len({r['source']for r in subset}),models={})
        for name in sessions:
            summary[split]['models'][name]={c:{m:sum(r['variants'][name]['counts'][c][m]for r in subset)for m in ('tp','fp','fn')}for c in NAMES}
            summary[split]['models'][name]['confused_as_container']={c:dict(
                errors=sum(b['mistaken_container']for r in subset for b in r['variants'][name]['confusion']if b['truth']==c),
                total=sum(1 for r in subset for b in r['variants'][name]['confusion']if b['truth']==c))for c in ('phone','camera','microphone')}
    save(OUT/'object-context-evaluation.json',dict(records=rows,summary=summary,confidence=.581,iou=.5,confusion_iou=.3,
        note='Original cached deployment ROI pixels, no extra crop around GT. Only centers inside reviewed region count. '
             'Train rows are fitting diagnostics; test original sources excluded from this fine-tune. Mined views, auxiliary GT.'))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
