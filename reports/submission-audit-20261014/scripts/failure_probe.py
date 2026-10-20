"""Reproduce fatal processing failures while capture remains alive, without GPU work."""
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'package/dark-demo-v2'))
from darkpipe.config import PipelineConfig
from darkpipe import server


class FailingStage:
    name = 'reproduce-failed-enhancement'
    post_recognition = False
    def load(self, device):
        pass
    def __call__(self, frames):
        raise RuntimeError('injected enhancement failure')
    def close(self):
        pass


cfg = PipelineConfig(mode='serve', input='http://camera/live', device='cpu',
                     enhance='off', sr='off', recognize='off', clip_dir='')
app = server.build_app(cfg)
st = app.state.dark
st.capture_alive = True
st.frames_in = 1
st.raw.put((np.zeros((48, 64, 3), np.uint8), time.time()))
original = server.build_stages
server.build_stages = lambda cfg: ([FailingStage()], None)
errors = []
def process():
    try:
        server.process_loop(st)
    except Exception as e:
        errors.append(str(e))
t = threading.Thread(target=process)
t.start()
t.join(timeout=3)
server.build_stages = original
route = next(r for r in app.routes if getattr(r, 'path', None) == '/health')
response = route.endpoint()
out = {'processing_thread_alive': t.is_alive(), 'processing_errors': errors,
       'stop_flag': st.stop.is_set(), 'health_http_status': response.status_code,
       'health': json.loads(response.body)}
assert errors == ['injected enhancement failure'] and not t.is_alive()
assert response.status_code == 200 and json.loads(response.body)['status'] == 'ok'
(ROOT / 'failure-probe.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
