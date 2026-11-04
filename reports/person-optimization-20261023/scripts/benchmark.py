"""Run the same six real inputs through the optimized single/multi-GPU runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import cv2
import torch

from runtime import PersonRuntime, save_json, ROOT
from action_engine import CHECKPOINT

OUT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cards', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--samples', nargs='+')
    parser.add_argument('--output', type=Path, default=OUT/'runs')
    parser.add_argument('--diagnostics', action='store_true')
    parser.add_argument('--no-paced', action='store_true')
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--process-workers', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f'Refusing to overwrite {args.output}')
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    baseline = json.loads((ROOT/'reports/person-performance-20261023/evidence/protocol.json').read_text())
    samples = baseline['samples']
    if args.samples:
        samples = [x for x in samples if x['name'] in args.samples]
        assert len(samples) == len(args.samples)
    protocol = dict(arguments={**vars(args), 'output': str(args.output)},
        checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        source_sha256=baseline['source_sha256'], samples=samples, cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),
        gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,uuid,memory.used,utilization.gpu', '--format=csv'], text=True),
        code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        coalesce=False, drop_frames=False, alpha=.7, threshold=.75, luma_gate=75, input=[32,224,400],
        output='Same original display dimensions, per-person drawing, H264 libx264 encoding',
        paced_method='FIFO nominal source arrivals at native fps, no source backpressure adjustment, local files only')
    save_json(args.output/'protocol.json', protocol)
    results = []
    for cards in args.cards:
        init_start = time.monotonic()
        runtime = PersonRuntime([f'cuda:{i}' for i in range(cards)], diagnostics=args.diagnostics,
                                process_workers=args.process_workers)
        init_seconds = time.monotonic()-init_start
        try:
            for warmup in range(args.warmups):
                warm_samples = [samples[0]] if len(samples)==1 else [samples[0], next((x for x in samples if x['processed']), samples[-1])]
                for item in warm_samples:
                    runtime.run(item, args.output/f'{cards}gpu'/f'warmup-{warmup+1}'/item['name'])
            for repeat in range(args.repeats):
                for item in samples:
                    result = runtime.run(item, args.output/f'{cards}gpu'/f'repeat-{repeat+1}'/item['name'])
                    result.update(cards=cards, repeat=repeat+1, initialization_seconds=init_seconds)
                    results.append(result)
                    save_json(args.output/'results.json', results)
            if not args.no_paced:
                for item in samples:
                    runtime.run(item, args.output/f'{cards}gpu'/'paced'/item['name'], paced=True)
        finally:
            runtime.close()
    save_json(args.output/'complete.json', dict(completed=True, runs=len(results)))


if __name__ == '__main__':
    main()
