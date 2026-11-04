"""Remove only exact visually checked false boxes, preserving evidence and unknowns."""
import json

import numpy as np

from scan import OUT, save


def overlap(a, b):
    a=np.array(a);b=np.array(b)
    inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return float(inter/max(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter,1))


def main():
    spec=json.loads((OUT/'visual-rejections.json').read_text());results=[]
    by_source={}
    for p in (OUT/'data').glob('*/annotations.json'):
        d=json.loads(p.read_text());by_source[d['video']['source']]=(p,d)
    for entry in spec['entries']:
        path,data=by_source[entry['source']]
        f=next(r for r in data['frames'] if r['frame']==entry['frame'])
        if f.get('assistant_rejected_proposals'):
            rejected=f['assistant_rejected_proposals']
        else:
            rejected=[p for p in f['people'] if entry['person_ids']=='all' or p['person_id'] in entry['person_ids']]
            f['assistant_rejected_proposals']=rejected
            f['people']=[p for p in f['people'] if p not in rejected]
        f['assistant_note']=entry['reason'];f['assistant_visual_status']='exact_false_proposals_removed_pending_user'
        f['risks']=sorted({r for p in f['people'] for r in p['risks']}|{'visual_correction_needs_user_review'})
        flagged=0
        for other in data['frames']:
            if other is f:continue
            for p in other['people']:
                if any(overlap(p['bbox_xyxy'],bad['bbox_xyxy'])>.6 for bad in rejected):
                    p['risks']=sorted(set(p['risks'])|{'resembles_visually_rejected_background'})
                    other['risks']=sorted(set(other['risks'])|{'resembles_visually_rejected_background'});flagged+=1
        save(path,data);(path.parent/'video-check.json').unlink(missing_ok=True)
        results.append(dict(source=entry['source'],frame=entry['frame'],removed=len(rejected),
                            adjacent_proposals_flagged_not_removed=flagged,reason=entry['reason']))
    save(OUT/'visual-rejections-summary.json',dict(records=results,removed_proposals=sum(r['removed'] for r in results),
        user_confirmed=False,include_in_training=False))
    print(json.dumps(results,ensure_ascii=False))


if __name__=='__main__':main()
