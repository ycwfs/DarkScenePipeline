"""Time the existing per-person demo, preserving its exact inference and output path.

Instrumentation wraps I/O and model calls; it does not implement another algorithm.
Paced replay models a FIFO source with nominal arrivals, without dropping frames.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / 'reports/person-action-20261023/scripts'))
import demo
import cv2
import numpy as np
import torch

CHECKPOINT = ROOT / 'reports/person-finetune-20261023/models/videomamba_t_behavior_32f_person_roi_20261023.pth'
CAPTURE = cv2.VideoCapture
WRITER = demo.VideoWriter
RENDER = demo.render_people


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def gpu_snapshot() -> str:
    return subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu,temperature.gpu',
        '--format=csv'], text=True)


class Probe:
    def __init__(self, paced: bool):
        self.paced = paced
        self.origin = None
        self.fps = 0.
        self.frames = []
        self.current = None
        self.close_seconds = 0.

    def timed(self, name: str, fn):
        def call(*args, **kwargs):
            start = time.monotonic()
            value = fn(*args, **kwargs)
            end = time.monotonic()
            self.current['stages'][name] = end - start
            return value
        return call

    def predict(self, fn, windows: list) -> list:
        start = time.monotonic()
        result = fn(windows)
        end = time.monotonic()
        self.current.update(action_seconds=end-start, predictions=len(result),
                            decision_time=end-self.origin if result else None)
        if result:
            self.current['newest_frame_age_seconds'] = [
                end-self.origin-w[-1].timestamp for w in windows]
            self.current['oldest_frame_age_seconds'] = [
                end-self.origin-w[0].timestamp for w in windows]
        return result

    def capture(self, *args, **kwargs):
        probe = self
        cap = CAPTURE(*args, **kwargs)
        self.fps = cap.get(cv2.CAP_PROP_FPS)

        class Capture:
            def __getattr__(self, name):
                return getattr(cap, name)

            def read(self):
                if probe.origin is None:
                    probe.origin = time.monotonic()
                index = len(probe.frames)
                arrival = probe.origin + index/probe.fps
                if probe.paced:
                    remaining = arrival-time.monotonic()
                    if remaining > 0:
                        time.sleep(remaining)
                start = time.monotonic()
                ok, frame = cap.read()
                if ok:
                    probe.current = dict(frame=index, media_time=index/probe.fps,
                        start_time=start-probe.origin, read_seconds=time.monotonic()-start,
                        queued_seconds=max(0., start-arrival), stages={})
                return ok, frame
        return Capture()

    def render(self, shown, current, shape, timestamp):
        self.current['people'] = len(current)
        start = time.monotonic()
        rendered = RENDER(shown, current, shape, timestamp)
        self.current['stages']['render'] = time.monotonic()-start
        return rendered

    def writer(self, *args, **kwargs):
        probe = self
        writer = WRITER(*args, **kwargs)

        class Writer:
            def __getattr__(self, name):
                return getattr(writer, name)

            def write(self, frame):
                start = time.monotonic()
                writer.write(frame)
                end = time.monotonic()
                probe.current.update(encode_write_seconds=end-start,
                    output_accept_time=end-probe.origin,
                    output_age_seconds=end-probe.origin-probe.current['media_time'])
                probe.frames.append(probe.current)

            def close(self):
                start = time.monotonic()
                writer.close()
                probe.close_seconds += time.monotonic()-start
        return Writer()


class TimedCall:
    def __init__(self, fn, probe: Probe, name: str):
        self.fn, self.probe, self.name = fn, probe, name

    def __call__(self, *args, **kwargs):
        return self.probe.timed(self.name, self.fn)(*args, **kwargs)


def run(item: dict, models, enhancer, folder: Path, paced: bool) -> dict:
    for sub in ('videos', 'events', 'stills', 'crops'):
        (folder/sub).mkdir(parents=True, exist_ok=True)
    probe = Probe(paced)
    predict, detector, pose = models.predict, models.detector, models.skeleton.pose
    cv2.VideoCapture, demo.VideoWriter, demo.render_people = probe.capture, probe.writer, probe.render
    models.predict = lambda windows: probe.predict(predict, windows)
    models.detector = TimedCall(detector, probe, 'detection')
    models.skeleton.pose = TimedCall(pose, probe, 'pose')
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = gpu_snapshot()
    try:
        result = demo.run_sample(item, models, TimedCall(enhancer, probe, 'enhancement'), folder)
        torch.cuda.synchronize()
    finally:
        cv2.VideoCapture, demo.VideoWriter, demo.render_people = CAPTURE, WRITER, RENDER
        models.predict, models.detector, models.skeleton.pose = predict, detector, pose
    assert len(probe.frames) == result['frames']
    result.update(paced=paced, processing_fps=result['frames']/result['elapsed'],
        encoder_close_seconds=probe.close_seconds,
        peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(),
        gpu_before=before, gpu_after=gpu_snapshot())
    write_json(folder/'frames.json', probe.frames)
    write_json(folder/'result.json', result)
    print('[benchmark]', item['name'], 'paced' if paced else 'throughput',
          f"{result['processing_fps']:.3f} fps", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT/'runs')
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f'Refusing to overwrite {args.output}')
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    names = ['喝水16', '挥手3', '留出_挥手28', '多人0060', '多人0054', '回传留出_181324']
    inventory = json.loads((ROOT/'reports/person-finetune-20261023/demo/sample-results.json').read_text())
    samples = [{k: v for k, v in next(x for x in inventory if x['name'] == name).items()
                if k in ('name', 'source', 'processed')} for name in names]
    sources = {x['source']: sha256(ROOT/x['source']) for x in samples}
    code_paths = [Path(__file__), ROOT/'reports/person-action-20261023/scripts/demo.py',
                  ROOT/'reports/person-action-20261023/scripts/person_core.py']
    model_paths = [CHECKPOINT] + [demo.OP/'ckpts'/name for name in
        ['NTIRE.pth', 'rtmo-s_end2end.onnx', 'stgcn_behavior_32f_realactall_full.pth', 'yolo26m_detect.onnx']]
    protocol = dict(schema_version=1, checkpoint=str(CHECKPOINT), samples=samples,
        source_sha256=sources, code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in code_paths},
        model_sha256={str(p.relative_to(ROOT)): sha256(p) for p in model_paths},
        torch=torch.__version__, opencv=cv2.__version__, python=platform.python_version(),
        cpu=platform.processor(), gpu=gpu_snapshot(), cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),
        torch_threads=4, opencv_threads=2, alpha=.7, threshold=.75, luma_gate=75,
        rgb_batch=2, rgb_frames=32, rgb_input_hw=[224, 400], processing_max_side=720,
        per_track_stride_seconds=.5, repetitions=3, warmup_passes=2,
        protocol='Unmodified demo.run_sample; wrappers add timestamps only. All frames decoded and encoded. '
            'Two complete warmup passes over drink16 and crowd0060. Three interleaved timed repetitions. '
            'Paced replay at source fps, FIFO nominal arrivals, no drops, no RTSP/network latency included. '
            'Newest-frame age = decision completion minus nominal arrival of newest window frame. '
            'Oldest-frame age additionally includes the approximately one-second observation window. '
            'No object prior is used; YOLO remains the person-box detector. Single GPU serial demo only.')
    write_json(args.output/'protocol.json', protocol)
    started = time.monotonic()
    models = demo.PersonModels('cuda:0')
    models.rgb.close()
    models.rgb = demo.BehaviorRecognizer(str(demo.OP/'ckpts'), skel_fusion_alpha=1.)
    models.rgb.ckpt = str(CHECKPOINT)
    models.rgb.load('cuda:0')
    models.rgb.warmup()
    enhancer = demo.RetinexformerStage(str(demo.OP/'ckpts'), chunk=1)
    enhancer.load('cuda:0')
    protocol['initialization_seconds'] = time.monotonic()-started
    protocol['detector_providers'] = models.detector.sess.get_providers()
    write_json(args.output/'protocol.json', protocol)
    results = []
    try:
        for repeat in range(2):
            for item in (samples[0], samples[3]):
                run(item, models, enhancer, args.output/f'warmup-{repeat+1}'/item['name'], False)
        for repeat in range(3):
            for item in samples:
                result = run(item, models, enhancer, args.output/f'repeat-{repeat+1}'/item['name'], False)
                results.append(dict(repeat=repeat+1, **result))
                write_json(args.output/'results.json', results)
        for item in samples:
            run(item, models, enhancer, args.output/'paced'/item['name'], True)
    finally:
        models.close()
        enhancer.close()
    write_json(args.output/'complete.json', dict(completed=True, timed_runs=len(results), paced_runs=len(samples)))


if __name__ == '__main__':
    main()
