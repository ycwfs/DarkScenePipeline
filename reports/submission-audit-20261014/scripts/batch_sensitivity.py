"""Check a newly observed threshold-boundary window across real enhancement batch shapes."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import torch

sys.path.insert(0, '/opt/darkpipe/op')
from darkpipe.stages.enhance_retinexformer import RetinexformerStage
from darkpipe.stages.downscale import DownscaleStage
from darkpipe.stages.recognize import BehaviorRecognizer

torch.set_num_threads(4)
cv2.setNumThreads(2)
repo = Path('/ssd1/wfs/project/lowlight/DarkScenePipeline')
path = next((repo / 'online-negative-sample/20261012_083305_78').glob('*/20261012_101308_drinking_water_0104.mp4'))
cap = cv2.VideoCapture(str(path))
fps = cap.get(cv2.CAP_PROP_FPS)
raw = []
while True:
    ok, f = cap.read()
    if not ok: break
    raw.append(f)
cap.release()
raw = DownscaleStage(840)(raw)
ckpt = '/opt/darkpipe/op/ckpts'
r = BehaviorRecognizer(ckpt, span_sec=None, reject_tau=.65,
                        skel_fusion_alpha=.7, skel_no_person='soft',
                        fuse_mode='alpha', lum_gate=75)
r.load('cuda:0')
r.warmup()
report = {'video': str(path), 'frames': len(raw), 'fps': fps, 'runs': {}}
for chunk in (1, 8, 32):
    enhancer = RetinexformerStage(ckpt, chunk=chunk)
    enhancer.load('cuda:0')
    enh = enhancer(raw)
    data = []
    for tag, lo, hi in [('whole', 0, len(raw)), ('t0.0', 0, round(fps)),
                         ('t0.5', round(fps*.5), round(fps*.5)+round(fps))]:
        inds = np.linspace(lo, hi-1, 32).round().astype(int)
        r.reset()
        for i, ind in enumerate(inds):
            event = r.push(enh[int(ind)], i, i/32, raw_frame=raw[int(ind)])
        data.append({'window': tag, 'lo': lo, 'hi': hi, 'event': event.to_dict()})
    report['runs'][str(chunk)] = data
    enhancer.close()
    print('chunk', chunk, data, flush=True)
r.close()
Path('/audit/batch-sensitivity.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
