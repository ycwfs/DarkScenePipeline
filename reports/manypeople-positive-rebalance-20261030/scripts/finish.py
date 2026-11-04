"""Resume after training; freeze validation choice before repeated test regression."""
import subprocess
import sys
import time
from common import *


def wait(tags: list[str]) -> None:
    while True:
        jobs = read(OUT/'jobs.json')
        complete = True
        for tag in tags:
            state = json.loads(subprocess.check_output(['docker','inspect','--format','{{json .State}}',jobs[tag]['name']],text=True))
            if state['Running']:
                complete = False
            elif state['ExitCode'] != 0:
                raise RuntimeError(f"{tag} failed ({state['ExitCode']}); preserve logs and retry explicitly")
        if complete:return
        time.sleep(10)


def launch(tag: str, gpu: str, script: str, *args: str) -> None:
    if tag not in read(OUT/'jobs.json'):
        subprocess.run([sys.executable,str(OUT/'scripts/jobs.py'),tag,gpu,script,'--',*args],check=True)


def predict(branch: str, which: str, gpu: str) -> str | None:
    dest = OUT/'predictions'/f'{branch}-{which}.npz'
    if dest.exists():return None
    protocol = read(OUT/'protocol.json')
    if which == 'selected':model = read(OUT/'selection-lock.json')['selected_models'][branch]['model']
    elif which == 'candidate':model = read(OUT/branch/'selection.json')['path']
    else:model = str(OUT/branch/f"epoch-{protocol['training']['rgb_epochs' if branch=='rgb' else 'skeleton_epochs']:02d}.pth")
    digest = sha(model)
    for metadata in (OUT/'predictions').glob(f'{branch}-*.json'):
        previous = read(metadata)
        if previous['sha256'] == digest and metadata.with_suffix('.npz').exists():
            dest.symlink_to(metadata.with_suffix('.npz').resolve())
            save(dest.with_suffix('.json'),dict(previous,model=model,reused_from=str(metadata.with_suffix('.npz')),
                                              dataset_sha256=sha(OUT/'dataset.json')))
            print('REUSED',branch,which,flush=True)
            return None
    tag=f'predict-{branch}-{which}'
    launch(tag,gpu,'predict.py',branch,'--which',which)
    return tag


def main() -> None:
    wait(['train-rgb','train-skeleton'])
    tags=[]
    for branch,which,gpu in [('rgb','candidate','0'),('rgb','last','1'),('skeleton','candidate','4'),('skeleton','last','5')]:
        tag=predict(branch,which,gpu)
        if tag:tags.append(tag)
    wait(tags)
    launch('joint-select','4','joint_select.py');wait(['joint-select'])
    tags=[]
    for branch,gpu in [('rgb','0'),('skeleton','4')]:
        tag=predict(branch,'selected',gpu)
        if tag:tags.append(tag)
    wait(tags)
    launch('evaluate-selected','4','evaluate_selected.py');wait(['evaluate-selected'])
    launch('verify-replay','4','verify_replay.py');wait(['verify-replay'])
    subprocess.run([sys.executable,str(OUT/'scripts/report.py')],check=True)
    save(OUT/'pipeline-complete.json',dict(training_and_analysis_complete=True,production_modified=False,
                                        selection_lock_sha256=sha(OUT/'selection-lock.json'),finished=time.time()))
    print('COMPLETE',flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        save(OUT/'pipeline-error.json',dict(error=repr(exc),time=time.time()))
        raise
