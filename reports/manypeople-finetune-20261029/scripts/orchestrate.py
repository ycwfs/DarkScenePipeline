"""Host coordinator: bounded polling, explicit exit checks, resumable stage artifacts."""
import os
import subprocess
import sys
import time
from common import *

IMAGE='darkpipe-operator:0.3.0'
JOBS=read(OUT/'pipeline-jobs.json') if (OUT/'pipeline-jobs.json').exists() else {}


def launch(tag,gpu,script,*args):
    name='mp-ft29-'+tag
    if tag in JOBS:
        print('RESUME',tag,flush=True)
        return
    gpu_argument=f'"device={gpu}"' if ','in str(gpu) else f'device={gpu}'
    python=['/opt/conda/envs/darkpipe/bin/python','-u']
    if script=='train_models.py' and ','in str(gpu):python+=['-m','torch.distributed.run','--standalone','--nproc_per_node=2']
    cmd=['docker','run','-d','--name',name,'--gpus',gpu_argument,'--shm-size','4g',
        '--user',f'{os.getuid()}:{os.getgid()}','-v',f'{ROOT}:{ROOT}','-w',str(ROOT),
        '-e','OMP_NUM_THREADS=3','-e','OPENBLAS_NUM_THREADS=1','-e','XDG_CACHE_HOME=/tmp/ft29-cache',
        '-e','TRITON_CACHE_DIR=/tmp/ft29-triton',
        '--entrypoint','/bin/bash',IMAGE,'-c',
        ' '.join([*python,str(OUT/'scripts'/script),*args,'>',str(OUT/(tag+'.log')),'2>&1'])]
    cid=subprocess.check_output(cmd,text=True).strip()
    JOBS[tag]=dict(name=name,container=cid,gpu=gpu,script=script,args=args,started=time.time())
    save(OUT/'pipeline-jobs.json',JOBS);print('START',tag,gpu,flush=True)


def state(name):
    value=subprocess.check_output(['docker','inspect','--format','{{json .State}}',name],text=True)
    return json.loads(value)


def wait_jobs(names):
    while True:
        complete=True
        for name in names:
            s=state(name)
            if s['Running']:complete=False
            elif s['ExitCode']!=0:raise RuntimeError(f'{name} failed with exit {s["ExitCode"]}')
        if complete:return
        time.sleep(15)


def main():
    started=time.time();extract=read(OUT/'extraction-jobs.json')
    wait_jobs([j['name']for j in extract if j['scope']=='old'])
    for i,gpu in enumerate((4,5)):launch(f'objects-{i}',gpu,'object_evidence.py','--shard',f'{i}/2','--wait')
    wait_jobs([j['name']for j in extract])
    launch('assemble',3,'assemble.py');wait_jobs([JOBS['assemble']['name']])
    launch('train-rgb','0,1','train_models.py','rgb')
    launch('train-skeleton',3,'train_models.py','skeleton')
    launch('predict-rgb-baseline',2,'predict.py','rgb','--which','baseline')
    wait_jobs([JOBS['train-skeleton']['name']])
    launch('predict-skeleton-baseline',3,'predict.py','skeleton','--which','baseline')
    wait_jobs([JOBS['predict-skeleton-baseline']['name']]);launch('predict-skeleton-candidate',3,'predict.py','skeleton','--which','candidate')
    wait_jobs([JOBS['train-rgb']['name']]);launch('predict-rgb-candidate',0,'predict.py','rgb','--which','candidate')
    wait_jobs([j['name']for j in JOBS.values()])
    launch('joint-select',3,'joint_select.py');wait_jobs([JOBS['joint-select']['name']])
    launch('predict-rgb-selected',0,'predict.py','rgb','--which','selected')
    launch('predict-skeleton-selected',3,'predict.py','skeleton','--which','selected')
    wait_jobs([JOBS['predict-rgb-selected']['name'],JOBS['predict-skeleton-selected']['name']])
    launch('evaluate-selected',3,'evaluate_selected.py');wait_jobs([JOBS['evaluate-selected']['name']])
    save(OUT/'pipeline-complete.json',dict(complete=True,seconds=time.time()-started,jobs=JOBS))
    print('COMPLETE',flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as e:
        save(OUT/'pipeline-error.json',dict(error=repr(e),jobs=JOBS,time=time.time()))
        raise
