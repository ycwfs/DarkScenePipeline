"""Evaluate the frozen test set once, after validation-only checkpoint selection."""
import hashlib
import json
from pathlib import Path
import torch
from train import OUT,OP,BehaviorRecognizer,evaluate,write_json


def main() -> None:
    selection=json.loads((OUT/'selection.json').read_text())
    protocol=json.loads((OUT/'protocol.json').read_text())
    dataset=json.loads((OUT/'training-data.json').read_text())
    assert dataset['ready_for_training']
    assert hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()==dataset['audit']['protocol_sha256']
    selected=Path(selection['path'])
    assert hashlib.sha256(selected.read_bytes()).hexdigest()==selection['sha256']
    records=[r for r in dataset['records']if r['split']=='test']
    assert records
    torch.set_num_threads(4)
    for name,checkpoint in [('baseline',OP/'ckpts'/protocol['init']),('selected',selected)]:
        output=OUT/'evaluation'/f'test-{name}.json'
        if output.exists():
            previous=json.loads(output.read_text())
            assert previous['checkpoint_sha256']==hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            assert previous['training_data_sha256']==hashlib.sha256((OUT/'training-data.json').read_bytes()).hexdigest()
            print(f'[skip] Existing verified {name} test results',flush=True)
            continue
        recognizer=BehaviorRecognizer(str(OP/'ckpts'),skel_fusion_alpha=1.)
        recognizer.ckpt=str(checkpoint);recognizer.load('cuda:0')
        metrics,outputs=evaluate(recognizer.net,records)
        write_json(output,dict(checkpoint=str(checkpoint),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            training_data_sha256=hashlib.sha256((OUT/'training-data.json').read_bytes()).hexdigest(),
            selection_sha256=hashlib.sha256((OUT/'selection.json').read_bytes()).hexdigest(),
            metrics=metrics,records=outputs))
        print(f'[test] {name}: '+json.dumps(metrics),flush=True)
        recognizer.close()
    write_json(OUT/'test-completion.json',dict(completed=True,selection_epoch=selection['epoch'],
        data_sha256=hashlib.sha256((OUT/'training-data.json').read_bytes()).hexdigest(),
        test_videos=len({r['video_id']for r in records}),test_windows=len(records),
        selection_unchanged=True,threshold_tuned=False))


if __name__=='__main__':main()
