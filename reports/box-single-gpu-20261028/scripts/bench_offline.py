"""Single visible GPU, unchanged source frames, full offline output and clip writing."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import os
import subprocess
import threading

import cv2
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--runtime', required=True)
parser.add_argument('--name', required=True)
parser.add_argument('--ids', default='lighting-01,lighting-05,rule-喝水27,crowd-child,crowd-camera')
parser.add_argument('--max-frames', type=int, default=150)
parser.add_argument('--sr-scale', type=int, default=2)
args = parser.parse_args()
sys.path.insert(0, args.runtime)
from darkpipe.config import PipelineConfig
from darkpipe.person import service

assert torch.cuda.device_count() == 1, 'Benchmark must expose exactly one GPU'
manifest = json.loads((OUT / 'inputs.json').read_text())
destination = OUT / args.name
destination.mkdir(parents=True, exist_ok=True)
(destination / 'environment.json').write_text(json.dumps(dict(
    torch=torch.__version__, device=torch.cuda.get_device_name(),
    device_count=torch.cuda.device_count(), runtime=args.runtime,
    protocol='Offline full frames; no source skipping; startup separately timed; local files.'), indent=2))
BaseOutput, BaseEngine = service.Output, service.PersonEngine
for item in manifest:
    if args.ids != 'all' and item['id'] not in args.ids.split(','):
        continue
    dest = destination / item['id']
    dest.mkdir(exist_ok=True)
    records, actions, memory = [], [], []
    stop_monitor = threading.Event()
    def monitor():
        while not stop_monitor.is_set():
            rss = int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
            result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                     '--format=csv,noheader,nounits'], capture_output=True, text=True)
            gpu_mib = 0
            for line in result.stdout.splitlines():
                pid, value = line.split(',')
                if pid.strip() == str(os.getpid()) and value.strip().isdigit():
                    gpu_mib += int(value)
            memory.append(dict(monotonic=time.monotonic(), rss_bytes=rss, gpu_mib=gpu_mib))
            stop_monitor.wait(.5)
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

    class ObservedOutput(BaseOutput):
        def __init__(self, engine, callback, writer=None):
            def observed(shown, clean, record):
                records.append({k: v for k, v in record.items() if k not in ('tracks', 'keys')})
                callback(shown, clean, record)
            super().__init__(engine, observed, writer)

    class ObservedEngine(BaseEngine):
        def close(self):
            if self.pool is not None:
                for engine in self.pool.engines:
                    if engine is not None:
                        actions.extend(list(engine.calls))
            super().close()

    service.Output, service.PersonEngine = ObservedOutput, ObservedEngine
    cfg = PipelineConfig(person_mode=True, recognize='behavior', gpus='0', device='cuda:0',
        input_processed=item.get('processed', False), input=str(ROOT / item['input']),
        output=str(dest / 'output.mp4'), events_json=str(dest / 'events.json'),
        ckpt_dir=os.environ.get('BOX_WEIGHTS',str(ROOT / 'NVIDIA-Box/weights')), proc_max_side=720, sr='bicubic', sr_scale=args.sr_scale,
        reco_span_sec=1., clip_dir=str(dest / 'clips'), clip_session='review',
        max_frames=args.max_frames or None)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    rows = service.run_offline(cfg)
    stop_monitor.set()
    monitor_thread.join(timeout=5)
    summary = dict(input=item, stats=cfg.stats, elapsed_with_startup=time.monotonic()-started,
        torch_peak_allocated=torch.cuda.max_memory_allocated(),
        torch_peak_reserved=torch.cuda.max_memory_reserved(),
        rss_peak_bytes=max((r['rss_bytes'] for r in memory), default=0),
        nvidia_peak_mib=max((r['gpu_mib'] for r in memory), default=0),
        people_max=max((r['people'] for r in records), default=0),
        labels=dict(Counter(r['label'] for r in rows)))
    for name, value in [('summary', summary), ('frames', records), ('actions', actions), ('memory', memory)]:
        (dest / (name+'.json')).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(id=item['id'], fps=cfg.stats['fps'], people=summary['people_max'],
                         labels=summary['labels']), ensure_ascii=False), flush=True)
