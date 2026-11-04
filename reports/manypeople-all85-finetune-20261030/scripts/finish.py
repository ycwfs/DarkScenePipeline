"""Finish only after both trainings; lock old-val selection before diagnostics."""
import subprocess
import sys
import time
import numpy as np
from common import *
from start_training import wait


def launch(tag: str, gpu: str, script: str, *args: str) -> None:
    if tag in read(OUT/'jobs.json'):
        return
    subprocess.run([sys.executable,str(OUT/'scripts/jobs.py'),tag,gpu,script,'--',*args],check=True)


def prediction(branch: str, which: str, gpu: str) -> str | None:
    path=OUT/'predictions'/f'{branch}-{which}.npz'
    if path.exists():return None
    lock=read(OUT/'selection-lock.json')
    if which=='selected':model=lock['selected_models'][branch]['model']
    elif which=='trained':model=lock['best_trained_models'][branch]['model']
    elif which=='candidate':model=read(OUT/branch/'selection.json')['path']
    else:model=lock['all_checkpoint_models'][branch]['6' if branch=='rgb' else '20']['model']
    digest=sha(model);ids=scored_indices(read(OUT/'dataset.json')['records'])
    for meta in (OUT/'predictions').glob(f'{branch}-*.json'):
        old=read(meta);cached=meta.with_suffix('.npz')
        if old['sha256']!=digest or not cached.exists():continue
        with np.load(cached) as data:
            probs=data['probs']
            assert np.isfinite(probs[ids] if branch=='rgb' else probs[:,ids]).all()
        path.symlink_to(cached.resolve())
        save(path.with_suffix('.json'),dict(old,model=model,reused_from=str(cached),dataset_sha256=sha(OUT/'dataset.json')))
        print('REUSED',branch,which,flush=True)
        return None
    tag=f'predict-{branch}-{which}'
    launch(tag,gpu,'predict.py',branch,'--which',which)
    return tag


def main() -> None:
    while not (OUT/'training-complete.json').exists():
        if (OUT/'training-error.json').exists():raise RuntimeError('Training failed; see training-error.json')
        time.sleep(10)
    wait(['train-skeleton','train-rgb','objects'])
    launch('joint-select','4','joint_select.py');wait(['joint-select'])
    for phase in [('candidate','last'),('selected','trained')]:
        tags=[]
        for (branch,which),gpu in zip([(b,w) for w in phase for b in ['rgb','skeleton']],['0','1','4','5']):
            tag=prediction(branch,which,gpu)
            if tag:tags.append(tag)
        wait(tags)
    launch('evaluate-selected','4','evaluate_selected.py');wait(['evaluate-selected'])
    launch('verify-replay','4','verify_replay.py');wait(['verify-replay'])
    for which in ['trained','selected','last']:
        tag=f'export-{which}'
        launch(tag,'4','export_verify.py','--which',which);wait([tag])
    subprocess.run([sys.executable,str(OUT/'scripts/report.py')],check=True)
    save(OUT/'pipeline-complete.json',dict(training_and_analysis_complete=True,production_modified=False,
        selection_lock_sha256=sha(OUT/'selection-lock.json'),finished=time.time()))
    print('COMPLETE',flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        save(OUT/'pipeline-error.json',dict(error=repr(exc),time=time.time()))
        raise
