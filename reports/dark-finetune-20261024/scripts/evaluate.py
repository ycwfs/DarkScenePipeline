"""Paired inference at unchanged deployment thresholds; auxiliary-label metrics."""
import argparse
import json
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from inference import Frontend, OUT, ROOT
from prepare import iou,save


def match(truth,pred):
    if not len(truth) or not len(pred):return {}
    matrix=np.array([[iou(a,b) for b in pred] for a in truth])
    ii,jj=linear_sum_assignment(1-matrix)
    return {int(i):int(j) for i,j in zip(ii,jj) if matrix[i,j]>=.5}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['val','test'],default='test')
    args=parser.parse_args();cv2.setNumThreads(1)
    op=ROOT/'platform/dark-demo-v2/ckpts'
    models={'original':Frontend(op/'yolo26m_detect.onnx',op/'rtmo-s_end2end.onnx'),
            'candidate':Frontend(OUT/'models/yolo26m_dark_candidate.onnx',OUT/'models/rtmo-s_dark_candidate.onnx'),
            'pose_only':Frontend(op/'yolo26m_detect.onnx',OUT/'models/rtmo-s_dark_candidate.onnx')}
    samples=[s for s in json.loads((OUT/'manifest.json').read_text())['samples'] if s['split']==args.split]
    records=[]
    for s in samples:
        path=OUT/f'dataset/images/{s["split"]}/{s["video_id"]}_{s["frame"]:06d}_enhanced.png'
        image=cv2.imread(str(path));h,w=image.shape[:2]
        scale=np.array([w/s['width'],h/s['height']])
        truth=np.array([p['bbox_xyxy'] for p in s['people']])*np.tile(scale,2)
        for name,frontend in models.items():
            tracks,result=frontend(image,s['timestamp'])
            boxes=np.array(result['boxes']).reshape(-1,6);pose_boxes=np.array(result['pose_boxes']).reshape(-1,4)
            detmatch=match(truth,boxes[:,:4]);posematch=match(truth,pose_boxes)
            correct=0;joints=0;associated=0
            for i,p in enumerate(s['people']):
                mask=np.array(p['keypoint_mask'],bool);joints+=int(mask.sum())
                if i in posematch:
                    j=posematch[i];pred=np.array(result['pose_xy'][j]);cf=np.array(result['pose_scores'][j])
                    error=np.linalg.norm(pred-np.array(p['keypoints_xy'])*scale,axis=1)/max(np.linalg.norm(truth[i,2:]-truth[i,:2]),1)
                    correct+=int((mask&(error<=.1)&(cf>=.3)).sum())
                if i in detmatch:
                    # Same box+pose association used by the independent person runtime.
                    for t in tracks:
                        if iou(truth[i],t.box)>=.5 and t.pose is not None and (t.pose[:,2]>=.3).sum()>=10:
                            associated+=1;break
            records.append(dict(id=s['id'],source=s['source'],video_id=s['video_id'],frame=s['frame'],model=name,
                gt_boxes=len(truth),det_matched=len(detmatch),det_predictions=len(boxes),
                pose_matched=len(posematch),pose_predictions=len(pose_boxes),
                labelled_joints=joints,correct_joints=correct,associated_pose_people=associated,**result))
    summary={}
    for name in models:
        rows=[r for r in records if r['model']==name]
        total=lambda key:sum(r[key] for r in rows)
        summary[name]=dict(images=len(rows),auxiliary_boxes=total('gt_boxes'),
            box_recall=total('det_matched')/max(total('gt_boxes'),1),
            box_precision=total('det_matched')/max(total('det_predictions'),1),
            box_unmatched=total('det_predictions')-total('det_matched'),
            pose_box_recall=total('pose_matched')/max(total('gt_boxes'),1),
            pose_unmatched=total('pose_predictions')-total('pose_matched'),
            labelled_joints=total('labelled_joints'),
            pck10_including_missed=total('correct_joints')/max(total('labelled_joints'),1),
            associated_pose_people=total('associated_pose_people'))
    save(OUT/f'evaluation/paired-{args.split}.json',dict(summary=summary,records=records,
        caveat='Agreement with filtered auxiliary labels; missing people are not exhaustively annotated.',
        thresholds=dict(person=.35,pose=.45,keypoint=.3,iou=.5,pck_diagonal=.1)))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
