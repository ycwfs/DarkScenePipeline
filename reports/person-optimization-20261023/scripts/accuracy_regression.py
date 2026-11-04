"""Replay all 204 frozen person windows; never use accuracy to choose runtime batch size."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from action_engine import ActionEngine, ActionTask, ROOT
from person_core import Sample, BEHAVIORS
from runtime import save_json
from darkpipe.stages.recognize import reject_probs

DATA = ROOT/'reports/person-finetune-20261023'
OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA/'scripts'))
from train import metrics


def main() -> None:
    torch.set_num_threads(4)
    records = [r for r in json.loads((DATA/'training-data.json').read_text())['records'] if r['split']=='test']
    reference = json.loads((DATA/'evaluation/test-selected.json').read_text())
    expected = {r['id']: r for r in reference['records']}
    output, failures = [], []
    engine = ActionEngine('cuda:0', max_batch=4, diagnostics=True)
    try:
        for start in range(0, len(records), 4):
            rows, tasks = records[start:start+4], []
            for row in rows:
                with np.load(DATA/row['path']) as cache:
                    rgb, poses = cache['rgb'], cache['pose']
                window = tuple(Sample(ts, idx, pixels, np.asarray(box), pose, luma)
                    for ts,idx,pixels,box,pose,luma in zip(row['timestamps'],row['source_frames'],rgb,
                                                         row['crop_boxes'],poses,row['lumas']))
                tasks.append(ActionTask((row['video_id'], row['track_id'], 0), window,
                    tuple(row['crop_boxes'][-1]), 1., time.monotonic(), time.monotonic()))
            for row, result in zip(rows, engine.predict(tasks)):
                old = expected[row['id']]
                assert result['crop_sha256'] == row['crop_sha256']
                pr = np.asarray(result['rgb_probs'])
                rgb_label = BEHAVIORS[int(reject_probs(pr, BEHAVIORS, .75).argmax())]
                rgb_error = float(np.abs(pr-old['rgb_probs']).max())
                fused_error = float(np.abs(np.asarray(result['fused_probs'])-old['fused_probs']).max())
                same = rgb_label == old['rgb_label'] and result['label'] == old['fused_label'] and result['mode']==old['mode']
                if not same:
                    failures.append(row['id'])
                output.append(dict(id=row['id'],truth=row['label'],rgb_label=rgb_label,
                    fused_label=result['label'],mode=result['mode'],rgb_error=rgb_error,
                    fused_error=fused_error,labels_equal=same,rgb_probs=result['rgb_probs'],
                    fused_probs=result['fused_probs']))
    finally:
        engine.close()
    truth = [BEHAVIORS.index(r['truth']) for r in output]
    report = dict(windows=len(output), rgb=metrics(truth,[BEHAVIORS.index(r['rgb_label'])for r in output]),
        fused=metrics(truth,[BEHAVIORS.index(r['fused_label'])for r in output]),
        max_rgb_error=max(r['rgb_error']for r in output),max_fused_error=max(r['fused_error']for r in output),
        changed_labels_or_modes=failures, records=output)
    save_json(OUT/'accuracy-regression.json',report)
    print(json.dumps({k:v for k,v in report.items()if k!='records'},ensure_ascii=False),flush=True)
    assert not failures, failures


if __name__ == '__main__':
    main()
