"""Apply explicitly observed visual boxes and flagged interpolation, then re-propose pose."""
import json
import shutil

import cv2
import numpy as np
import torch
from rtmlib import RTMPose

from scan import OUT, save
from propose import MODELS, pose_tool, gamma, iou


def main() -> None:
    cv2.setNumThreads(1);torch.set_num_threads(2)
    spec=json.loads((OUT/'visual-track-anchors.json').read_text())
    tracks=spec['tracks']
    model=pose_tool(RTMPose,MODELS['rtmpose_s'],(192,256),(123.675,116.28,103.53),(58.395,57.12,57.375))
    counts=[]
    for path in sorted((OUT/'data').glob('*/annotations.json')):
        data=json.loads(path.read_text());v=data['video']
        if v['source'] not in tracks:continue
        backup=path.parent/'before-visual-refinement.json'
        if not backup.exists():shutil.copyfile(path,backup)
        data=json.loads(backup.read_text())
        anchors=np.array(tracks[v['source']],dtype=float)
        affected=0;explicit=0;filled=0
        for f in data['frames']:
            index=f['frame']
            if index<anchors[0,0] or index>anchors[-1,0]:continue
            box=np.array([np.interp(index,anchors[:,0],anchors[:,i]) for i in range(1,5)])
            box*=np.array([v['width']/480,v['height']/270]*2)
            is_anchor=bool(index in anchors[:,0]);explicit+=is_anchor;affected+=1
            raw=cv2.imread(str(OUT/f['image']));enhanced=cv2.imread(str(OUT/f['enhanced_image']))
            scale=np.array(raw.shape[1::-1])/np.array([v['width'],v['height']])
            candidates=[]
            for name,image in [('gamma',gamma(raw)),('retinexformer',enhanced)]:
                xy,cf=model(image,bboxes=[(box*np.tile(scale,2)).tolist()])
                candidates.append((name,np.column_stack([xy[0]/scale,cf[0]])))
            name,kp=max(candidates,key=lambda r:np.minimum(r[1][:,2],.8).sum())
            valid=(kp[:,2]>=.3)&((kp[:,:2]>=0)&(kp[:,:2]<[v['width'],v['height']])).all(1)
            previous=f['people'];kept=[];replaced=[]
            for p in previous:
                pbox=np.array(p['bbox_xyxy']);center=(pbox[:2]+pbox[2:])/2
                if iou(box,pbox)>.15 or ((center>=box[:2])&(center<=box[2:])).all():replaced.append(p)
                else:kept.append(p)
            if not replaced:filled+=1
            risks=['visual_anchor_needs_user_review'] if is_anchor else ['interpolated_box']
            if valid.sum()<10:risks.append('sparse_pose')
            p=dict(person_id='P900',bbox_xyxy=box.round(3).tolist(),bbox_source='assistant_visual_anchor' if is_anchor else 'interpolated_visual_anchors_unreviewed',
                bbox_score=None,box_review='pending',keypoints=kp.round(4).tolist(),keypoint_visibility=[None]*17,
                keypoint_usable=valid.tolist(),pose_source='rtmpose_s/'+name,pose_review='pending',
                proposal_support=['assistant_visual_anchor_trajectory'],risks=risks,max_pose_disagreement=None,
                alternative_boxes=[dict(source=x['bbox_source'],bbox_xyxy=x['bbox_xyxy']) for x in replaced],include_in_training=False)
            f['people']=kept+[p];f['risks']=sorted({risk for person in f['people'] for risk in person['risks']})
            f['assistant_visual_status']='box_keyframe_checked' if is_anchor else 'interpolated_not_visually_checked'
        data['visual_refinement']=dict(anchors=explicit,frames_with_interpolation=affected-explicit,
            frame_person_additions_without_matched_model_box=filled,anchor_source='visual-track-anchors.json',
            user_confirmed=False,pose_visibility_confirmed=False)
        save(path,data)
        # This is a cache invalidation for a changed draft, not a claim that the old video is current.
        (path.parent/'video-check.json').unlink(missing_ok=True)
        counts.append(dict(source=v['source'],**data['visual_refinement']))
        print(v['source'],data['visual_refinement'],flush=True)
    save(OUT/'visual-refinement-summary.json',dict(videos=counts,explicit_new_box_keyframes=sum(r['anchors'] for r in counts),
        interpolated_frames=sum(r['frames_with_interpolation'] for r in counts),include_in_training=False))


if __name__=='__main__':main()
