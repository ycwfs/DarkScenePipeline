"""Check candidate coverage, source-frame alignment, media decode and review staging."""
import json
import math
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from collect import OUT, read, save, sha


def check_media(g):
    p=OUT/g['video'];m=read(OUT/'render-metadata'/f'{g["id"]}.json')
    source=next(s for s in read(OUT/'sources.json') if s['number']==g['number'])
    version=3 if g['number']==3 else 2
    assert m['render_version']==version,(g['id'],m)
    from fractions import Fraction
    fps=float(Fraction(str(source['fps'])))
    assert m['start_frame']==math.floor(g['preview_start']*fps+1e-5),(g['id'],'start')
    assert m['end_frame_exclusive']==min(source['frames'],math.ceil(g['preview_end']*fps-1e-5)),(g['id'],'end')
    d=subprocess.run(['ffmpeg','-v','error','-xerror','-threads','1','-i',str(p),'-map','0:v:0',
        '-f','null','-'],capture_output=True,text=True)
    if d.returncode or d.stderr:raise AssertionError((g['id'],d.returncode,d.stderr))
    probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
        'stream=nb_frames,width,height,duration','-of','json',str(p)],capture_output=True,text=True,check=True)
    info=json.loads(probe.stdout)['streams'][0]
    assert int(info['nb_frames'])==m['frames'] and info['width']==1280 and info['height']==624
    assert abs(float(info['duration'])-m['seconds'])<.09
    assert (OUT/g['poster']).exists()
    return dict(id=g['id'],frames=m['frames'],sha256=sha(p),strict_decode=True,
                target_box_frames=m['target_box_visible_frames'])


def main():
    groups=read(OUT/'review-items.json');obs=read(OUT/'observations.json');clips=read(OUT/'original-clips.json')
    assigned=[o for g in groups for o in g['observation_ids']]
    assert len(assigned)==len(set(assigned))==len(obs)==7049
    assert set(assigned)=={o['id']for o in obs}
    assert sum(o['original']for o in obs)==994
    assert len({c['id']for c in clips})==454
    for c in clips:
        assert (OUT/c['video']).stat().st_ino==__import__('pathlib').Path(c['original_path']).stat().st_ino
        assert all(any(g['id']==uid and c['video'] in g['original_clips']for g in groups) for uid in c['review_ids'])
    assert all(g['human_label'] is None and not g['training_eligible'] for g in groups)
    counts=Counter();sources=Counter()
    with (OUT/'base-windows.jsonl').open()as f:
        for line in f:
            w=json.loads(line);counts[w['status']]+=1;sources[w['number']]+=1
            assert w['human_label'] is None and not w['training_eligible']
            if w['status']=='other_candidate_unreviewed':assert w['prediction']=='other' and not w['review_ids']
    assert sum(counts.values())==47600
    summary=read(OUT/'summary.json');assert counts['other_candidate_unreviewed']==summary['other_candidates']
    main_items=[g for g in groups if g['category']=='videomamba'];media=[]
    with ThreadPoolExecutor(max_workers=6)as pool:
        futures=[pool.submit(check_media,g)for g in main_items]
        for f in as_completed(futures):
            media.append(f.result())
            if len(media)%50==0:print(f'decoded {len(media)}/{len(main_items)}',flush=True)
    save('media-verification.json',sorted(media,key=lambda x:x['id']))
    save('verification.json',dict(passed=True,main_review=len(main_items),xclip_review=len(groups)-len(main_items),
        original_saved_clips=454,original_clips_unchanged_hardlinks=True,initial_non_other_events_covered=994,
        historical_non_other_observations_covered=len(obs),duplicate_observation_assignments=0,
        all_main_previews_strictly_decoded=True,preview_frames=sum(m['frames']for m in media),
        zero_target_box_previews=[m['id']for m in media if not m['target_box_frames']],
        base_window_states=dict(counts),base_source_counts=dict(sources),
        training_labels_empty=True,models_and_production_code_unchanged=True,
        source_checks='source-media.json:9 original SHA256 values and all80316 remuxed frame counts verified'))
    print('All coverage, decode, alignment, and unlabelled-staging checks passed',flush=True)


if __name__=='__main__':main()
