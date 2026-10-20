"""Check clip output contracts using archived code; no repository edits or GPUs."""
import json
from pathlib import Path
import sys
import threading
import time
import cv2
import numpy as np

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / 'package/dark-demo-v2'))
from darkpipe.clips import ClipRecorder, OPEN_AFTER
from darkpipe.stages.base import RecognitionEvent

cv2.setNumThreads(1)
out = root / 'clip-contract'
out.mkdir(exist_ok=True)
event = RecognitionEvent(0, 0., 'Drinking water', .99)
frame = np.full((64, 96, 3), 60, np.uint8)

rec = ClipRecorder(str(out), session='denoise', stage_dir=str(out / 'stage1'),
                   pre_sec=0, post_sec=100, max_sec=100, denoise='quality', realtime=False)
clean_calls = []
original_clean = rec._clean
def count_clean(image):
    clean_calls.append(1)
    return original_clean(image)
rec._clean = count_clean
for i in range(50):
    rec.push(frame.copy(), event if i == 0 else None, i / 25)
rec.close()
meta = json.loads(next((out / 'denoise').rglob('*.json')).read_text())
report = {'denoise': {'input_frames': 50, 'open_after': OPEN_AFTER,
                       'clean_calls': len(clean_calls),
                       'frames_written': meta['frames_written'],
                       'saved': rec.saved, 'writer_alive_after_close': rec._thread.is_alive()}}

entered, release = threading.Event(), threading.Event()
rec2 = ClipRecorder(str(out), session='close', stage_dir=str(out / 'stage2'),
                    pre_sec=0, post_sec=100, max_sec=100, denoise='quality', realtime=False)
def slow_clean(image):
    entered.set()
    release.wait(50)
    return image
rec2._clean = slow_clean
for i in range(50):
    rec2.push(frame.copy(), event if i == 0 else None, i / 25)
assert entered.wait(10)
t0 = time.monotonic()
rec2.close()
report['offline_close'] = {'realtime': rec2.realtime,
                            'close_seconds': time.monotonic() - t0,
                            'writer_alive_after_close': rec2._thread.is_alive(),
                            'queue_messages_after_close': rec2._q.qsize(),
                            'saved_at_return': rec2.saved,
                            'metadata_at_return': len(list((out / 'close').rglob('*.json')))}
release.set()
rec2._thread.join(10)
report['offline_close']['saved_after_unblock'] = rec2.saved
(root / 'clip-contract.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
