"""Validate user review exports without training or accepting model predictions as truth."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from scan import OUT, save


def inspect_feedback(payload: dict) -> dict:
    catalog=json.loads((OUT/'catalog.json').read_text())
    if payload.get('schema')!='dark_person_pose_feedback_v1' or payload.get('dataset')!=catalog['dataset']:
        raise ValueError('Wrong feedback schema/dataset')
    videos={v['id']:v for v in catalog['videos']}
    scans={};seen=set();statuses=Counter();eligible=Counter();pose=Counter();unassigned=[]
    for item in payload['records']:
        vid=item['video_id'];record=item['record'];index=record['frame'];unique=(vid,index)
        if unique in seen:raise ValueError(f'Duplicate reviewed frame: {unique}')
        seen.add(unique)
        if vid not in videos or item['sha256']!=videos[vid]['sha256'] or item['source']!=videos[vid]['source']:
            raise ValueError(f'Wrong source/hash: {vid}')
        if vid not in scans:scans[vid]=json.loads((OUT/'data'/vid/'scan.json').read_text())
        scan=scans[vid];v=scan['video']
        if not isinstance(index,int) or index<0 or index>=len(scan['frames']) or not scan['frames'][index]['eligible']:
            raise ValueError(f'Frame not in the raw L<75 set: {unique}')
        if abs(record['luma']-scan['frames'][index]['luma'])>1e-8:
            raise ValueError(f'Luma changed in feedback: {unique}')
        ids=[]
        for p in record['people']:
            ids.append(p['person_id']);b=p['bbox_xyxy'];kp=p['keypoints'];vis=p['keypoint_visibility']
            if len(b)!=4 or not all(math.isfinite(x) for x in b) or not (0<=b[0]<b[2]<=v['width'] and 0<=b[1]<b[3]<=v['height']):
                raise ValueError(f'Invalid native bbox: {unique}/{p["person_id"]}')
            if len(kp)!=17 or len(vis)!=17 or any(x not in (None,0,1,2) for x in vis):
                raise ValueError(f'Invalid COCO17 structure: {unique}')
            for k,visibility in zip(kp,vis):
                if len(k)!=3 or not all(math.isfinite(x) for x in k):raise ValueError(f'Non-finite keypoint: {unique}')
                if visibility in (1,2) and not (0<=k[0]<=v['width'] and 0<=k[1]<=v['height']):
                    raise ValueError(f'Reviewed joint out of image: {unique}')
        if len(set(ids))!=len(ids):raise ValueError(f'Duplicate person IDs in frame: {unique}')
        status=record['review_status'];statuses[status]+=1
        split=v['frozen_split']['split']
        if status=='confirmed_empty':
            if record['people'] or not record.get('exhaustive_person_annotation'):
                raise ValueError(f'Confirmed empty contradicts annotations: {unique}')
            eligible[split]+=1
        elif status=='approved':
            if not record['people'] or not record.get('exhaustive_person_annotation'):
                raise ValueError(f'Incomplete person audit: {unique}')
            if any(p['box_review']!='user_confirmed' for p in record['people']):
                raise ValueError(f'Unconfirmed box in approved frame: {unique}')
            eligible[split]+=1
            for p in record['people']:
                if p['pose_review']=='user_confirmed_visible_subset':
                    if any(x is None for x in p['keypoint_visibility']):
                        raise ValueError(f'Unreviewed visibility in approved pose: {unique}')
                    if any(x in (1,2) for x in p['keypoint_visibility']):pose[split]+=1
        if split=='unassigned':unassigned.append(dict(video_id=vid,frame=index))
    return dict(valid=True,reviewed_frames=len(seen),review_statuses=dict(statuses),
        detector_reviewed_frames_by_existing_split=dict(eligible),
        pose_reviewed_people_by_existing_split=dict(pose),unassigned_frames=unassigned,
        training_started=False,note='Validation only. Val/test stay held out. Unassigned AVI families require grouping before training.')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('feedback',type=Path)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    result=inspect_feedback(json.loads(args.feedback.read_text()))
    if args.report:save(args.report,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
