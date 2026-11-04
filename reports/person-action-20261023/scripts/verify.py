"""Audit source-track alignment, person-specific fusion and complete output media."""
from collections import Counter
import json
from pathlib import Path
import subprocess

import numpy as np

from person_core import DEFAULT_ACTION_THRESHOLD, fuse_person

OUT = Path(__file__).resolve().parents[1]
rows = json.loads((OUT/'sample-results.json').read_text())
checks = []
for row in rows:
    events = json.loads((OUT/'events'/f"{row['name']}.json").read_text())
    frames = json.loads((OUT/'events'/f"{row['name']}_frames.json").read_text())
    assert len(frames) == row['frames']
    by_frame = {}
    for frame in frames:
        tids = [t['track_id'] for t in frame['tracks']]
        assert len(tids) == len(set(tids)) == frame['people']
        by_frame[frame['frame']] = frame
    hashes = {}
    for event in events:
        indices, timestamps = event['source_frames'], event['source_timestamps']
        assert len(indices) == len(timestamps) == len(event['crop_boxes']) == 32
        assert all(a <= b for a, b in zip(indices, indices[1:]))
        assert indices[-1] == event['frame_index']
        for index, timestamp in zip(indices, timestamps):
            source = by_frame[index]
            assert timestamp == source['timestamp']
            assert any(t['track_id'] == event['track_id'] for t in source['tracks'])
        rgb = np.array(event['rgb_probs'], np.float32)
        skeleton = None if event['skeleton_probs'] is None else np.array(event['skeleton_probs'], np.float32)
        assert rgb.shape == (10,) and np.isfinite(rgb).all()
        np.testing.assert_allclose(rgb.sum(), 1, atol=1e-5)
        if skeleton is not None:
            assert skeleton.shape == (10,) and np.isfinite(skeleton).all()
            np.testing.assert_allclose(skeleton.sum(), 1, atol=1e-5)
        threshold = event.get('threshold', DEFAULT_ACTION_THRESHOLD)
        assert event['label'] == 'other' or event['confidence'] >= threshold
        expected = fuse_person(rgb, skeleton, [by_frame[i]['raw_luma'] for i in indices], threshold=threshold)
        assert expected['label'] == event['label'] and expected['mode'] == event['mode']
        assert abs(expected['confidence']-event['confidence']) < 1e-6
        np.testing.assert_allclose(expected['fused_probs'], event['fused_probs'], atol=1e-7)
        same_time = hashes.setdefault(event['frame_index'], [])
        same_time.append(event['crop_sha256'])
    assert all(len(v) == len(set(v)) for v in hashes.values()), 'Two people received identical RGB input'
    video = OUT/row['video']
    stream = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-count_frames','-show_entries','stream=width,height,nb_read_frames,codec_name,duration,avg_frame_rate',
        '-of','json',str(video)]))['streams'][0]
    assert stream['codec_name']=='h264' and int(stream['nb_read_frames'])==row['frames']
    assert [stream['width'],stream['height']] == row['output_size'] and not row['global_footer']
    decode = subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],
                            check=True,capture_output=True,text=True,timeout=120)
    assert not decode.stderr.strip(), decode.stderr
    checks.append(dict(name=row['name'], full_decode=True, track_alignment=True,
                       own_rgb_inputs=True, per_person_fusion=True, predictions=len(events), **stream))
summary = dict(videos=len(rows), frames=sum(r['frames'] for r in rows),
               predictions=sum(r['predictions'] for r in rows),
               max_detected_people=max(r['max_detected_people'] for r in rows),
               labels=dict(sum((Counter(r['labels']) for r in rows),Counter())),
               modes=dict(sum((Counter(r['modes']) for r in rows),Counter())), checks=checks)
(OUT/'verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
