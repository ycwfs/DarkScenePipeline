"""Run input verification, both branch training jobs and missing object evidence."""
import subprocess
import sys
import time
from common import *


def launch(tag: str, gpu: str, script: str, *args: str, ddp: int = 1) -> None:
    jobs = read(OUT/'jobs.json') if (OUT/'jobs.json').exists() else {}
    if tag in jobs:
        return
    cmd=[sys.executable,str(OUT/'scripts/jobs.py'),'--ddp',str(ddp),tag,gpu,script,*args]
    subprocess.run(cmd,check=True)


def wait(tags: list[str]) -> None:
    while True:
        jobs=read(OUT/'jobs.json');complete=True
        for tag in tags:
            state=json.loads(subprocess.check_output(['docker','inspect','--format','{{json .State}}',jobs[tag]['name']],text=True))
            if state['Running']:complete=False
            elif state['ExitCode']!=0:raise RuntimeError(f'{tag} failed; keep log and retry explicitly')
        if complete:return
        time.sleep(10)


def main() -> None:
    launch('verify-data','4','verify_data.py');wait(['verify-data'])
    launch('train-skeleton','5','train_models.py','skeleton')
    launch('objects','4','object_evidence.py')
    wait(['train-skeleton','objects'])
    launch('train-rgb','0,1,4,5','train_models.py','rgb',ddp=4)
    wait(['train-rgb'])
    save(OUT/'training-complete.json',dict(complete=True,finished=time.time()))
    print('TRAINING COMPLETE',flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        save(OUT/'training-error.json',dict(error=repr(exc),time=time.time()))
        raise
