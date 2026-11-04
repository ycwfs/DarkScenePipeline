"""Paced latest-frame replay at native fps; preserves source timestamps and reports drops.

Injects a capture adapter into the optimized runtime. No model or fusion changes.
This is a local replay, not a platform RTSP or camera test.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import queue
import threading
import time

import cv2
import torch

import runtime as rt
from action_engine import ROOT

OUT = Path(__file__).resolve().parents[1]


class LatestDecoder:
    """Bounded one-slot preprocessed input. Producer pace never follows consumer speed."""
    def __init__(self, source: Path, processed: bool = False, seconds: float = 0.):
        self.cap = cv2.VideoCapture(str(source))
        if not self.cap.isOpened():
            raise RuntimeError(f'Cannot open {source}')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.source_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total = max(self.source_count, math.ceil(seconds*self.fps))
        self.count = 0  # consumed frames, used by runtime's output count assertion
        self.dropped = 0
        self.processed = processed
        self.stop = threading.Event()
        self.done = threading.Event()
        self.error = None
        self.condition = threading.Condition()
        self.slot = None
        self.last_index = -1
        self.origin = time.monotonic()
        self.thread = threading.Thread(target=self._run, daemon=True, name='native-paced-source')
        self.thread.start()

    def _run(self):
        downscale = rt.DownscaleStage(720)
        try:
            for index in range(self.total):
                if self.stop.is_set():
                    break
                if index and index % self.source_count == 0:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, image = self.cap.read()
                if not ok:
                    raise RuntimeError(f'Unexpected decode EOF at {index}')
                shown = image[:rt.content_height(image.shape[0])] if self.processed else None
                raw = downscale([shown if self.processed else image])[0]
                luma = None if self.processed else rt.frame_luma(raw)
                remaining = self.origin+index/self.fps-time.monotonic()
                if remaining > 0 and self.stop.wait(remaining):
                    break
                with self.condition:
                    if self.slot is not None:
                        self.dropped += 1
                    self.slot = (index, raw, shown, luma)
                    self.condition.notify_all()
        except Exception as exc:
            self.error = exc
        finally:
            self.cap.release()
            self.done.set()
            with self.condition:
                self.condition.notify_all()

    def read(self):
        with self.condition:
            while self.slot is None and not self.done.is_set():
                self.condition.wait(.1)
            if self.error:
                raise RuntimeError('Live source failed') from self.error
            if self.slot is None:
                return None
            value, self.slot = self.slot, None
            self.count += 1
            self.last_index = value[0]
            return value

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError('Live source failed to stop')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cards', type=int, nargs='+', default=[1,2,3])
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--samples', nargs='+', default=['挥手3','留出_挥手28','多人0060'])
    parser.add_argument('--output', type=Path, default=OUT/'live')
    parser.add_argument('--process-workers', action='store_true')
    parser.add_argument('--auto-backend', action='store_true',help='Inline for one card, threads for two, processes for three or more')
    parser.add_argument('--long-field', action='store_true',help='Use a 30-second raw field clip instead of the short benchmark clips')
    parser.add_argument('--samples-file', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Refusing to overwrite live results')
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    samples = json.loads((ROOT/'reports/person-performance-20261023/evidence/protocol.json').read_text())['samples']
    selected = [x for x in samples if x['name'] in args.samples]
    assert len(selected)==len(args.samples)
    if args.long_field:
        selected = [dict(name='长现场_挥手',source='real-actiondata/挥手.mp4',processed=False)]
    if args.samples_file:
        selected = json.loads(args.samples_file.read_text())
    rt.save_json(args.output/'protocol.json',dict(cards=args.cards,seconds=args.seconds,samples=selected,
        auto_backend=args.auto_backend,process_workers=args.process_workers,
        code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in Path(__file__).parent.glob('*.py')},
        source_sha256={x['source']:hashlib.sha256((ROOT/x['source']).read_bytes()).hexdigest()for x in selected},
        method='Native-rate latest-frame replay, one slot, no source slowdown. Short videos loop; '
            'tracker and submission history reset at clip boundaries, including skipped boundary frames. '
            'Input FPS equals consumed unique source indices / emitted source duration. '
            'No RTSP, platform, HAMI, NFS or remote player included.'))
    results = []
    for cards in args.cards:
        runner = rt.PersonRuntime([f'cuda:{i}'for i in range(cards)], coalesce=True,
                                  process_workers=args.process_workers or (args.auto_backend and cards>=3))
        try:
            for warm in range(2):
                runner.run(selected[0],args.output/f'{cards}gpu'/f'warmup-{warm}')
            for item in selected:
                result = runner.run(item,args.output/f'{cards}gpu'/item['name'],paced=True,
                                    source_factory=lambda p,processed:LatestDecoder(p,processed,args.seconds))
                results.append(result)
                rt.save_json(args.output/'results.json',results)
        finally:
            runner.close()
    rt.save_json(args.output/'complete.json',dict(completed=True,runs=len(results)))


if __name__=='__main__':
    main()
