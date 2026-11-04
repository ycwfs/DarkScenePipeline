"""Measure existing CPU preprocessing versus uint8 GPU preprocessing on real ROIs."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
DATA = ROOT/'reports/person-finetune-20261023'
sys.path.insert(0, str(ROOT/'reports/person-action-20261023/scripts'))
from person_core import OP
from darkpipe.stages.recognize import BehaviorRecognizer


@torch.inference_mode()
def main() -> None:
    torch.set_num_threads(4)
    reco = BehaviorRecognizer(str(OP/'ckpts'), skel_fusion_alpha=1.)
    reco.ckpt = str(DATA/'models/videomamba_t_behavior_32f_person_roi_20261023.pth')
    reco.load('cuda:0')
    reco.warmup()
    rows = [r for r in json.loads((DATA/'training-data.json').read_text())['records'] if r['split']=='val'][:16]
    crops = [np.load(DATA/r['path'])['rgb'] for r in rows]
    mean = torch.as_tensor(reco.cfg['mean'], device='cuda:0')
    std = torch.as_tensor(reco.cfg['std'], device='cuda:0')
    records = []
    for batch in (1, 2, 4, 8, 16):
        pinned = torch.empty((batch, 32, 224, 400, 3), dtype=torch.uint8, pin_memory=True)
        dest = torch.empty_like(pinned, device='cuda:0')
        expected = None
        for mode in ('baseline', 'uint8'):
            for repeat in range(5):
                torch.cuda.synchronize()
                start = time.perf_counter()
                if mode == 'baseline':
                    pixels = np.stack(crops[:batch])
                else:
                    for i, crop in enumerate(crops[:batch]):
                        np.copyto(pinned.numpy()[i], crop)
                packed = time.perf_counter()
                if mode == 'baseline':
                    array = (pixels.astype(np.float32)/255.-reco.cfg['mean'])/reco.cfg['std']
                    normalized = time.perf_counter()
                    x = torch.from_numpy(array).to('cuda:0').permute(0, 4, 1, 2, 3).contiguous()
                else:
                    normalized = packed
                    dest.copy_(pinned, non_blocking=True)
                    x = dest.float().div_(255.).sub_(mean).div_(std).permute(0, 4, 1, 2, 3).contiguous()
                torch.cuda.synchronize()
                uploaded = time.perf_counter()
                with torch.autocast('cuda', dtype=torch.float16):
                    prob = reco.net(x).softmax(1).float().cpu().numpy()
                ended = time.perf_counter()
                if mode == 'baseline':
                    expected = prob
                error = float(np.abs(prob-expected).max())
                if repeat >= 2:
                    row = dict(batch=batch, mode=mode, repeat=repeat-1,
                        pack_ms=1000*(packed-start), cpu_normalize_ms=1000*(normalized-packed),
                        upload_and_gpu_prepare_ms=1000*(uploaded-normalized),
                        forward_and_readback_ms=1000*(ended-uploaded), total_ms=1000*(ended-start),
                        windows_per_second=batch/(ended-start), max_prob_error=error,
                        top1_changes=int((prob.argmax(1)!=expected.argmax(1)).sum()))
                    records.append(row)
                    print(json.dumps(row), flush=True)
                    (OUT/'profile.json').write_text(json.dumps(records, indent=2)+'\n')
    reco.close()


if __name__ == '__main__':
    main()
