"""Post-selection bottle threshold description; never selects a deployment threshold."""
import json
import numpy as np
from common import OUT, save
from export import iou


def main():
    data=json.loads((OUT/'object-evaluation.json').read_text());rows=[]
    for context in ['review_crop','deployment_roi']:
        records=[r for r in data['records']if r['split']=='test'and r['context']==context]
        for name in ['original80','current7','candidate80']:
            for tau in [.05,.1,.2,.25,.35,.5,.581,.7,.8,.9]:
                tp=fp=fn=0
                for r in records:
                    gt=[b['bbox']for b in r['boxes']if b['class_id']==0];used=set()
                    pred=sorted([b for b in r['variants'][name]['predictions']if b[4]>=tau and int(b[5])==39],key=lambda b:-b[4])
                    for b in pred:
                        matches=[(iou(np.array(b[:4]),np.array(g)),j)for j,g in enumerate(gt)if j not in used]
                        if matches and max(matches)[0]>=.5:used.add(max(matches)[1])
                    tp+=len(used);fp+=len(pred)-len(used);fn+=len(gt)-len(used)
                rows.append(dict(context=context,model=name,threshold=tau,tp=tp,fp=fp,fn=fn))
    save(OUT/'bottle-threshold-diagnostic.json',dict(records=rows,
        note='Descriptive post-selection diagnostic only; no model/default/threshold selection. Same heldout cropped images, not action-window inference.'))


if __name__=='__main__':main()
