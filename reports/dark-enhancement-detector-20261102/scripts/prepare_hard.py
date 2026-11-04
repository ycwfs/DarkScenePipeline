"""Freeze quartile frames from ALL eight dark test sources, before inference."""
import json
from pathlib import Path
import cv2
import numpy as np
from run import ROOT, OUT, save

manifest = json.loads((ROOT / 'reports/dark-finetune-20261024/manifest.json').read_text())
videos = [x['video'] for x in manifest['videos']
          if x['video']['frozen_split']['split'] == 'test']
assert len(videos) == 8, len(videos)
rows = []
(OUT / 'hard/native').mkdir(parents=True, exist_ok=True)
(OUT / 'hard/review').mkdir(exist_ok=True)
for video in sorted(videos, key=lambda x: x['id']):
    vid = video['id']
    old = json.loads((ROOT / f'reports/dark-annotations-20261023/data/{vid}/annotations.json').read_text())
    cap = cv2.VideoCapture(str(ROOT / video['source']))
    indices = [int(video['frames'] * q) for q in (.25, .5, .75)]
    panels = []
    for f in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, native = cap.read()
        assert ok
        row_id = f'H{len(rows) + 1:03d}'
        cv2.imwrite(str(OUT / f'hard/native/{row_id}.png'), native)
        annotation = next(x for x in old['frames'] if x['frame'] == f)
        enhanced = cv2.imread(str(ROOT / 'reports/dark-annotations-20261023' / annotation['enhanced_image']))
        assert enhanced is not None
        panels_for_frame = []
        for name, image in [('raw', native), ('cached NTIRE review only', enhanced)]:
            image = cv2.resize(image, (768, 432))
            banner = np.zeros((32, 768, 3), np.uint8)
            cv2.putText(banner, f'{row_id} F{f} L={annotation["luma"]:.1f} {name}', (6, 23),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1)
            panels_for_frame.append(np.vstack([banner, image]))
        panels.append(np.hstack(panels_for_frame))
        rows.append({'id': row_id, 'source': video['source'], 'video_id': vid, 'frame': f,
                     'luma': annotation['luma'], 'native_shape': list(native.shape[:2]),
                     'historical_proposals': [p['bbox_xyxy'] for p in annotation['people']]})
    cap.release()
    cv2.imwrite(str(OUT / f'hard/review/{vid}.jpg'), np.vstack(panels))
save(OUT / 'hard/protocol.json', {
    'selection': 'all eight frozen dark-test sources; floor(N*.25), floor(N*.5), floor(N*.75)',
    'reason': 'original 44 reliable auxiliary labels omit three hard sources entirely',
    'not_independent_test': True, 'model_selection': False,
    'annotation': 'review visible person boxes before new inference; unresolved regions ignored',
    'samples': rows})
print([(r['id'],r['source'],r['frame']) for r in rows])
