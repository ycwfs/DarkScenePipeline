"""Separate visually audited people from unlabelled full-video output diagnostics."""
from collections import defaultdict
import csv
import json
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from replay import OUT, combinations, COMBINATIONS, save
from prepare import iou


def match(truth: list, predicted: np.ndarray, threshold: float = .5) -> dict:
    if not len(truth) or not len(predicted):
        return {}
    overlap = np.array([[iou(a,b[:4]) for b in predicted] for a in truth])
    # Maximise feasible cardinality before IoU; do not throw away valid assignments.
    r,c = linear_sum_assignment(-((overlap >= threshold)*10 + overlap))
    return {int(i):int(j) for i,j in zip(r,c) if overlap[i,j]>=threshold}


def ignored(box: np.ndarray, regions: list) -> bool:
    for region in regions:
        intersection = np.maximum(0,np.minimum(box[2:4],region[2:])-np.maximum(box[:2],region[:2])).prod()
        if intersection/max(np.prod(box[2:4]-box[:2]),1)>=.5:
            return True
    return False


def audit(row: dict, label: dict, threshold: float = .5) -> dict:
    result = {}
    for name,(boxes,poses) in combinations(row).items():
        matches = match(label['person_boxes'], boxes, threshold)
        extras = [i for i,b in enumerate(boxes) if i not in matches.values() and not ignored(b,label['ignore_regions'])]
        good = lambda pose: pose is not None and (pose[:,2]>=.3).sum()>=10
        result[name] = dict(gt=len(label['person_boxes']),matched=len(matches),predictions=len(boxes),
            unmatched=len(extras),true_associated_pose10=int(sum(good(poses[j]) for j in matches.values())),
            unmatched_associated_pose10=int(sum(good(poses[j]) for j in extras)),
            missing=[i for i in range(len(label['person_boxes'])) if i not in matches],
            unmatched_indices=extras,matched_indices=matches)
    for name in ['original','candidate']:
        entry=row[name]
        boxes=np.asarray(entry['pose_boxes']).reshape(-1,4)
        cf=np.asarray(entry['pose_scores']).reshape(-1,17)
        matches=match(label['person_boxes'],boxes,threshold)
        extras=[i for i,b in enumerate(boxes) if i not in matches.values() and not ignored(b,label['ignore_regions'])]
        result[name+'_raw_pose']=dict(gt=len(label['person_boxes']),matched=len(matches),predictions=len(boxes),unmatched=len(extras),
            matched_pose10=int(sum((cf[j]>=.3).sum()>=10 for j in matches.values())),
            unmatched_pose10=int(sum((cf[j]>=.3).sum()>=10 for j in extras)),
            missing=[i for i in range(len(label['person_boxes'])) if i not in matches],
            matched_visible_joints=[int((cf[j]>=.3).sum()) for j in matches.values()],
            unmatched_indices=extras)
    return result


def main() -> None:
    protocol=json.loads((OUT/'protocol.json').read_text())
    labels=json.loads((OUT/'visual-audit-labels.json').read_text())['labels']
    audits=[];video_rows=[];exhibit=[]
    for video,label in zip(protocol['videos'],labels):
        assert video['id']==label['video_id']
        rows=json.loads((OUT/'data'/video['id']/'outputs.json').read_text())['frames']
        selected=next(r for r in rows if r['frame']==label['frame'])
        audits.append(dict(video_id=video['id'],source=video['source'],index=label['index'],frame=label['frame'],
            by_iou={str(t):audit(selected,label,t) for t in [.3,.5,.7]}))
        counts=defaultdict(int);fakeframes=defaultdict(list)
        for row in rows:
            for name in ['original','candidate']:
                counts[name+'_boxes']+=len(row[name]['boxes'])
                counts[name+'_raw_pose']+=len(row[name]['pose_boxes'])
                visible=(np.asarray(row[name]['pose_scores']).reshape(-1,17)>=.3).sum(1)
                counts[name+'_raw_pose10']+=int((visible>=10).sum())
                if label['known_nonperson_boxes']:
                    if match(label['known_nonperson_boxes'],np.asarray(row[name]['boxes']).reshape(-1,6)):
                        fakeframes[name+'_detector'].append(row['frame'])
                    if match(label['known_nonperson_boxes'],np.asarray(row[name]['pose_boxes']).reshape(-1,4)):
                        fakeframes[name+'_raw_pose'].append(row['frame'])
            for name,(boxes,poses) in combinations(row).items():
                counts[name+'_associated_pose10']+=sum(p is not None and (p[:,2]>=.3).sum()>=10 for p in poses)
                if label['known_nonperson_boxes']:
                    matches=match(label['known_nonperson_boxes'],boxes)
                    if any(poses[j] is not None and (poses[j][:,2]>=.3).sum()>=10 for j in matches.values()):
                        fakeframes[name+'_associated_pose10'].append(row['frame'])
        video_rows.append(dict(video_id=video['id'],source=video['source'],frames=len(rows),**{k:int(v) for k,v in counts.items()}))
        if label['known_nonperson_boxes']:
            exhibit.append(dict(video_id=video['id'],source=video['source'],frames=len(rows),
                fixed_region=label['known_nonperson_boxes'][0], counts={k:len(v) for k,v in fakeframes.items()},frame_indices=dict(fakeframes)))
    summaries={}
    for t in ['0.3','0.5','0.7']:
        sums={}
        for name in audits[0]['by_iou'][t]:
            metrics={k for k,v in audits[0]['by_iou'][t][name].items() if isinstance(v,int)}
            sums[name]={key:sum(r['by_iou'][t][name][key] for r in audits) for key in metrics}
        summaries[t]=sums
    save(OUT/'visual-audit-results.json',dict(summary=summaries,records=audits,
        caveat='Assistant approximate boxes; not independent GT; pose10 is coverage, not keypoint localization accuracy.'))
    save(OUT/'full-replay-statistics.json',dict(videos=video_rows,
        totals={k:sum(r[k] for r in video_rows) for k in video_rows[0] if k not in ('video_id','source')},
        caveat='Unlabelled output counts, NOT precision/recall. Adjacent frames are correlated.'))
    save(OUT/'exhibit-audit.json',dict(videos=exhibit,
        caveat='Fixed known humanoid-exhibit region at IoU>=0.5; possible temporary occlusion by people. Not all false positives.'))
    with (OUT/'per-video.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(video_rows[0]));writer.writeheader();writer.writerows(video_rows)
    print(json.dumps(summaries['0.5'],indent=2))
    print('Mismatches:')
    for r in audits:
        s=r['by_iou']['0.5']
        if any(s[k]['missing'] or s[k]['unmatched'] for k in ['original','both','original_raw_pose','candidate_raw_pose']):
            print(r['index'],r['source'],{k:(s[k]['matched'],s[k]['unmatched'],s[k]['missing']) for k in s})


if __name__=='__main__':
    main()
