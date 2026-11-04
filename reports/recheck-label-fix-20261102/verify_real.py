"""Exercise the actual entry point, observing live/clip render calls without changing decisions."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import threading

import cv2

p = argparse.ArgumentParser()
p.add_argument('--operator', default='/app')
p.add_argument('--out', default='/evidence')
p.add_argument('--input', default='/evidence/input65.mp4')
a = p.parse_args()
sys.path.insert(0, a.operator)
import main
from darkpipe.constants import BEHAVIOR_ZH
from darkpipe.person import core, service, clips

out = Path(a.out)
original_render, original_text = core.render_people, core.text_patch
local = threading.local()
lock = threading.Lock()
counts = Counter()
recovered = set()
previews = []


def observe_text(text, *args):
    if getattr(local, 'texts', None) is not None:
        local.texts.append(text)
    return original_text(text, *args)


def renderer(kind):
    def checked(frame, tracks, shape, stamp, *args):
        local.texts = []
        shown = original_render(frame, tracks, shape, stamp, *args)
        texts, local.texts = local.texts, None
        for track in tracks:
            row = track.prediction
            if not row or stamp-track.last_inference > 1.:
                continue
            expected = (f'T{track.tid:02d} 其他' if row['label'] == 'other' else
                        f"T{track.tid:02d} {BEHAVIOR_ZH[row['label']]} {row['confidence']*100:.0f}%")
            assert expected in texts, (kind, stamp, expected, texts, row.get('recheck'))
            with lock:
                counts[kind+'_labels_checked'] += 1
                if row.get('recheck', {}).get('recovered'):
                    counts[kind+'_recovered_draws'] += 1
                    key = (kind, row['track_id'], row['history_generation'], row['timestamp'])
                    if key not in recovered and kind == 'output' and len(previews) < 4:
                        path = out/f'confirmed-{len(previews)+1}.jpg'
                        cv2.imwrite(str(path), shown)
                        previews.append(dict(file=path.name, timestamp=stamp, label=row['label'],
                                             confidence=row['confidence'], track_id=track.tid))
                    recovered.add(key)
        return shown
    return checked


core.text_patch = observe_text
service.render_people = renderer('output')
clips.render_people = renderer('clip')
args = main.build_parser().parse_args([
    '--video_path', a.input, '--gpu_ids', '0', '--enhance', 'off', '--sr', 'off',
    '--denoise', 'off', '--clip_denoise', 'off', '--extra_hosts', '',
    '--clip_dir', str(out/'outputs'), '--session_json', str(out/'session.json')])
main.run(args)
session = json.loads((out/'session.json').read_text())
event_paths = session['events_json'] if isinstance(session['events_json'], list) else [session['events_json']]
events = []
for event_path in event_paths:
    event_file = Path(event_path)
    payload = json.loads(event_file.read_text())
    events.extend(payload.get('events', []) if isinstance(payload, dict) else payload)
recovered_events = [row for row in events if row.get('recheck', {}).get('recovered')]
assert recovered_events, 'The source must exercise at least one actual recovered model result'
assert counts['output_recovered_draws'] > 0 and counts['clip_recovered_draws'] > 0
decoded = []
for path in sorted((out/'outputs').rglob('*.mp4')):
    cap = cv2.VideoCapture(str(path))
    expected = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        count += 1
    cap.release()
    assert count == expected and count > 0, (str(path), expected, count)
    decoded.append(dict(path=str(path.relative_to(out)), frames=count))
result = dict(scope='Real GPU inference in the original image, enhancement/SR off to reproduce the prior run; not a streaming or throughput acceptance',
              counts=dict(counts), events=len(events), recovered_events=len(recovered_events),
              recovered_summary=[{k:r[k] for k in ('timestamp', 'track_id', 'label', 'confidence', 'recheck')}
                                 for r in recovered_events],
              decoded=decoded, previews=previews)
(out/'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(result, ensure_ascii=False), flush=True)
