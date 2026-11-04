"""Run the optimized independent per-person candidate on a local video."""
import argparse
from pathlib import Path

import cv2
import torch

from runtime import PersonRuntime, ROOT
from live_replay import LatestDecoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cards',type=int,default=1,help='Number of visible GPUs; physical IDs come from CUDA_VISIBLE_DEVICES')
    parser.add_argument('--processed',action='store_true',help='Platform clip with old footer; strip footer and do not enhance again')
    parser.add_argument('--mode',choices=['all-frames','latest-replay'],default='all-frames')
    parser.add_argument('--seconds',type=float,default=0.,help='Latest replay only: loop short source to this duration')
    parser.add_argument('--process-workers',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    video = args.video.resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    runner = PersonRuntime([f'cuda:{i}'for i in range(args.cards)],
                           coalesce=args.mode=='latest-replay',process_workers=args.process_workers)
    try:
        source_factory = (lambda p,processed:LatestDecoder(p,processed,args.seconds)) if args.mode=='latest-replay' else None
        runner.run(dict(name=video.stem,source=str(video),processed=args.processed),args.output,
                   paced=args.mode=='latest-replay',source_factory=source_factory)
    finally:
        runner.close()


if __name__=='__main__':main()
