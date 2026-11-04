"""Store a visually placed body skeleton as an assistant draft, not confirmed visibility."""
import json
from scan import OUT, save


def main():
    spec=json.loads((OUT/'visual-pose-anchor.json').read_text())
    path=OUT/'data'/spec['video_id']/'annotations.json';data=json.loads(path.read_text())
    frame=next(f for f in data['frames'] if f['frame']==spec['frame'])
    p=next(p for p in frame['people'] if p['box_review']=='user_confirmed')
    if 'assistant_pose_original' not in p:
        p['assistant_pose_original']=dict(keypoints=p['keypoints'],source=p['pose_source'],usable=p['keypoint_usable'])
    p['keypoints']=[[0,0,0] for _ in range(17)];p['keypoint_usable']=[False]*17
    p['keypoint_visibility']=[None]*17
    sx=data['video']['width']/spec['reference_size'][0];sy=data['video']['height']/spec['reference_size'][1]
    for joint,xy in spec['points'].items():
        j=int(joint);p['keypoints'][j]=[xy[0]*sx,xy[1]*sy,0.0];p['keypoint_usable'][j]=True
    p['pose_source']='assistant_visual_body_joint_draft';p['pose_review']='pending'
    p['pose_score_note']='Zero model scores are placeholders for manually placed points, not model confidence; reviewed visibility remains null.'
    p['risks']=sorted(set(p['risks'])|{'visual_pose_draft_needs_user_review'})
    frame['risks']=sorted(set(frame['risks'])|{'visual_pose_draft_needs_user_review'})
    frame['assistant_note']=spec['note'];save(path,data)
    (path.parent/'video-check.json').unlink(missing_ok=True)
    print('One body-joint draft added; no user-confirmed pose or training labels created.')


if __name__=='__main__':main()
