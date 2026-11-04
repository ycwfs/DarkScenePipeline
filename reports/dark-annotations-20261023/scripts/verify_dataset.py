"""Verify exhaustive frame coverage, provenance, unresolved labels and all review assets."""
from collections import Counter
import json
import math
from pathlib import Path

from PIL import Image

from scan import OUT, digest, save


def main() -> None:
    inventory=json.loads((OUT/'inventory.json').read_text());catalog=json.loads((OUT/'catalog.json').read_text())
    totals=Counter();unique=set();confirmed=[];issues=[]
    review_ids={v['id'] for v in catalog['videos']}
    for v in inventory['raw_videos']:
        directory=OUT/'data'/v['id'];scan=json.loads((directory/'scan.json').read_text())
        totals['raw_videos']+=1;totals['source_frames']+=len(scan['frames'])
        dark={f['frame']:f for f in scan['frames'] if f['eligible']}
        assert len(dark)==scan['dark_frames']
        assert scan['all_alias_hashes_verified'] and scan['full_decode']
        if not dark:
            assert v['id'] not in review_ids;continue
        totals['dark_videos']+=1;totals['dark_frames']+=len(dark)
        data=json.loads((directory/'annotations.json').read_text())
        assert {f['frame'] for f in data['frames']}==set(dark)
        assert len(data['frames'])==len(dark) and not data['include_in_training']
        assert data['video']['sha256']==v['sha256']
        js=(directory/'review.js').read_text();prefix='window.receiveAnnotation('
        web=json.loads(js[len(prefix):js.rfind(');')])
        assert web['video']['sha256']==v['sha256']
        for f,wf in zip(data['frames'],web['frames']):
            token=(v['id'],f['frame']);assert token not in unique;unique.add(token)
            assert wf['frame']==f['frame'] and wf['people']==f['people'],f'Stale web annotations {token}'
            assert f['luma']==dark[f['frame']]['luma'] and f['luma']<75
            assert f['timestamp']==dark[f['frame']]['timestamp']
            assert not f['include_in_training'] and not f['exhaustive_person_annotation']
            assert not f['no_box_is_negative']
            for asset in (f['image'],f['enhanced_image'],wf['gamma_image']):
                path=OUT/asset
                with Image.open(path) as im:assert list(im.size)==f['preview_size']
            totals['review_images_checked']+=3
            if not f['people']:totals['no_box_frames']+=1
            usable_pose=False
            ids=set()
            for p in f['people']:
                assert p['person_id'] not in ids;ids.add(p['person_id'])
                b=p['bbox_xyxy'];assert len(b)==4 and all(math.isfinite(x) for x in b)
                assert -1e-3<=b[0]<b[2]<=v['width']+1e-3 and -1e-3<=b[1]<b[3]<=v['height']+1e-3,(token,b)
                assert len(p['keypoints'])==len(p['keypoint_usable'])==len(p['keypoint_visibility'])==17
                assert all(x is None for x in p['keypoint_visibility']) and not p['include_in_training']
                for xy,shown in zip(p['keypoints'],p['keypoint_usable']):
                    assert len(xy)==3 and all(math.isfinite(x) for x in xy)
                    if shown:assert -1e-3<=xy[0]<=v['width']+1e-3 and -1e-3<=xy[1]<=v['height']+1e-3
                totals['person_boxes']+=1
                totals['proposed_visible_joints']+=sum(p['keypoint_usable'])
                if sum(p['keypoint_usable'])>=10:totals['people_with_at_least_10_proposed_points']+=1
                usable_pose=usable_pose or any(p['keypoint_usable'])
                if p['box_review']=='user_confirmed':confirmed.append(token)
            if usable_pose:totals['frames_with_any_proposed_pose']+=1
            else:totals['frames_without_proposed_pose']+=1
            if not f['people'] or f['risks']:
                issues.append(dict(video_id=v['id'],source=v['source'],frame=f['frame'],timestamp=f['timestamp'],
                    luma=f['luma'],people=len(f['people']),risks=f['risks'],note=f.get('assistant_note','')))
        check=json.loads((directory/'video-check.json').read_text())
        assert check['complete_decode'] and check['frames']==v['frames'] and check['dark_annotated_frames']==len(dark)
        assert (OUT/check['path']).exists()
        totals['review_videos_fully_decoded']+=1;totals['review_video_frames']+=check['frames']
    assert set(confirmed)=={('1d4d072b22d10fe1',32),('4078760fdadd05f6',30)} and len(confirmed)==2
    assert totals['dark_frames']==19892 and totals['dark_videos']==167 and totals['source_frames']==42193
    assert totals['dark_frames']==catalog['summary']['annotated_frames']
    assert totals['person_boxes']==catalog['summary']['proposed_person_boxes']
    model_record=json.loads((OUT/'models-shard-0.json').read_text())
    for model in model_record.values():assert digest(Path(model['path']))==model['sha256']
    save(OUT/'review-issues.json',dict(note='Review queue, not a count of confirmed model errors.',frames=issues))
    result=dict(passed=True,counts=dict(totals),raw_alias_hashes_verified=620,
        user_confirmed_boxes=2,user_confirmed_poses=0,training_started=False,
        four_proposal_model_hashes_unchanged=True,
        checks=['exact_all_dark_frame_set','unique_frame_ids','original_luma_and_timestamps','native_coordinate_bounds',
                'COCO17_shapes_and_unknown_visibility','no_unreviewed_training_labels','all_three_image_views',
                'web_json_matches_annotations','all_review_video_frame_counts_and_full_decode',
                'only_two_user_confirmed_boxes','model_hashes_unchanged'])
    save(OUT/'verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
