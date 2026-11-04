"""Freeze source-group splits before model evaluation, preserving old feedback holdouts."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

OUT=Path(__file__).resolve().parents[1]


def main() -> None:
    manifest=json.loads((OUT/'source-inventory.json').read_text())
    videos=manifest['videos']
    groups={row['id']:row['id'] for row in videos}

    def root(uid: str) -> str:
        while groups[uid]!=uid:
            uid=groups[uid]
        return uid

    tokens={}
    for row in videos:
        row_tokens=[]
        for alias in row['aliases']:
            if row['processed']:
                session=row.get('session',Path(alias).parts[1])
                token=('processed',session)
            else:
                # Same basename across original/cut directories is kept in one group.
                token=('raw',row['video_label'],Path(alias).stem)
            row_tokens.append(token)
        if not row['processed'] and any(Path(alias).parts[0] in
                ['real-actiondata','real-actiondata-cutdrink','real-actiondata-0909'] for alias in row['aliases']):
            # Visual review found renumbered trims (e.g. original drink3 becomes drink2).
            # Keep the entire older recording family together, even when basenames differ.
            row_tokens.append(('legacy_recording_family',row['video_label']))
        for token in row_tokens:
            if token in tokens:
                groups[root(row['id'])]=root(tokens[token])
            else:
                tokens[token]=row['id']
    by_group=defaultdict(list)
    for row in videos:
        by_group[root(row['id'])].append(row)
    assignments={}
    raw_by_label=defaultdict(list)
    processed_train=[]
    for gid,items in by_group.items():
        if any(row['sealed_holdout'] for row in items):
            assignments[gid]='test'
        elif items[0]['processed']:
            if any(row['dataset']=='online-negative-sample' for row in items):
                assignments[gid]='train'
            else:
                processed_train.append(gid)
        else:
            labels={row['video_label'] for row in items}
            assert len(labels)==1
            legacy=any(Path(alias).parts[0] in ['real-actiondata','real-actiondata-cutdrink','real-actiondata-0909']
                       for row in items for alias in row['aliases'])
            if legacy:
                assignments[gid]='train'
            else:
                raw_by_label[next(iter(labels))].append(gid)
    rng=random.Random(23)
    for label,gids in sorted(raw_by_label.items()):
        gids=sorted(gids)
        rng.shuffle(gids)
        count=max(1,round(.2*len(gids))) if len(gids)>=5 else 0
        for index,gid in enumerate(gids):
            assignments[gid]='val' if index<count else 'test' if index<2*count else 'train'
    for index,gid in enumerate(sorted(processed_train)):
        assignments[gid]='val' if index==0 else 'train'
    records=[dict(video_id=row['id'],source=row['source'],group=root(row['id']),
                  split=assignments[root(row['id'])],video_label=row['video_label'],
                  sealed_feedback_holdout=row['sealed_holdout']) for row in videos]
    out=dict(seed=23,scope='all local field videos and two platform negative batches',
             records=records,summary=dict(Counter(row['split'] for row in records)),
             per_split_video_label={split:dict(Counter(row['video_label'] for row in records if row['split']==split))
                                   for split in ['train','val','test']},
             constraints=['byte-identical copies and same-name original/cut videos share split',
                          'renumbered original/cut legacy recording families kept entirely in train',
                          'processed feedback groups by complete recording session',
                          'previously sealed two feedback sessions remain test-only',
                          'source labels are stratification only, not per-person annotations',
                          'historical field test videos may have been seen by initial pn2'])
    path=OUT/'splits.json'
    if path.exists():
        assert json.loads(path.read_text())==out,'Refusing to change frozen source splits'
    else:
        path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    protocol=dict(
        scope_confirmed='user: local field videos + platform negatives; visual person annotation first',
        init='videomamba_t_behavior_32f_realactall_full_pn2.pth',
        initial_sha256='d30b36c4c1ca9ab276442b4d0544b3d705f4ead68c7dcb5a721b696f9ee5f362',
        split_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        alpha=.7,threshold=.75,lum_gate=75.,
        candidate=dict(epochs=8,head_lr=1e-4,backbone_lr=1e-5,batch=8,seed=23,
                       optimizer='AdamW',weight_decay=.05,label_smoothing=.05,
                       augmentation='shared horizontal flip across all 32 frames',
                       sampling='class balanced, equal source-video mass within class',
                       distillation=False,amp_initial_scale=512.),
        selection='validation fused macro-F1 at 0.75, then RGB macro-F1, then lower other false-positive rate; baseline epoch0 eligible',
        test='sealed until validation checkpoint selection; no threshold tuning on test',
        annotation='visual track/time labels; no copied scene labels or prediction-generated ground truth',
        unsupported_classes=['fight','talk','shake_hands'],
        single_video_classes=['throw','fall'],
        production_modified=False,
        window_annotation_rule=dict(action_min_fraction=.5,max_unknown_fraction_for_action=.2,
            other_min_fraction=.9,other_requires_zero_named_action_frames=True,
            competing_action_tie='exclude',unreviewed_positive_track='exclude',
            uncertain_person_track='exclude',
            negative_video_label='inherit user/video-level other unless explicit person review supersedes'))
    protocol_path=OUT/'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text())==protocol
    else:
        protocol_path.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({key:out[key] for key in ['summary','per_split_video_label']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
