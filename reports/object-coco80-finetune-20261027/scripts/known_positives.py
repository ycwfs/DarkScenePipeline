"""Locate known-positive visual anchors in the new complete-video timeline."""
import json
import cv2
import numpy as np
from common import OUT, PREVIOUS, save
from export import iou
from review_long import read_rows


def main():
    views=json.loads((PREVIOUS/'label-views.json').read_text());audits=[]
    (OUT/'review').mkdir(exist_ok=True)
    scale=np.array([1920/720,1080/404]*2)
    for number,idx,stamp in [(1,1,149.92),(9,112,54.8)]:
        v=views[idx];rect=np.array(v['rect']);modes={}
        for name in ['baseline','candidate']:
            rows=read_rows(OUT/'long-videos'/f'new-{number:02}'/f'{name}-events.jsonl')
            local=[r for r in rows if abs(r['timestamp']-stamp)<.6]
            scores=[(iou(np.array(r['bbox'])*scale,rect),r)for r in local]
            scores.sort(key=lambda p:-p[0]);identity=scores[0][1]['track_id']
            best=min([v for v in scores if v[1]['track_id']==identity],key=lambda v:abs(v[1]['timestamp']-stamp))
            modes[name]=dict(iou_with_review_roi=best[0],event=best[1],
                adjacent_track_events=[r for r in local if r['track_id']==identity])
            print(number,name,[(round(x,3),r['timestamp'],r['track_id'],r['action_label'],r['label'],r['action_confidence'],r['object_prior']['reason'])for x,r in scores[:4]])
        audits.append(dict(number=number,timestamp=stamp,review_frame=v['frame'],review_roi=rect.tolist(),
            truth_basis='source01 user-confirmed drink; source09 assistant-visible bottle near mouth, not independent temporal action GT',variants=modes))
        cap=cv2.VideoCapture(v['source']);cap.set(cv2.CAP_PROP_POS_FRAMES,v['frame']);ok,frame=cap.read();assert ok
        for name,color in [('baseline',(0,0,255)),('candidate',(0,255,0))]:
            r=modes[name]['event'];b=np.round(np.array(r['bbox'])*scale).astype(int);cv2.rectangle(frame,tuple(b[:2]),tuple(b[2:]),color,2)
        cv2.imwrite(str(OUT/'review'/f'known-positive-{number:02}.jpg'),frame);cap.release()
    save(OUT/'known-positive-audit.json',audits)


if __name__=='__main__':main()
