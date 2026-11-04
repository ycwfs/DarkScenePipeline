import os,subprocess,concurrent.futures,time
from common import *

def run(gpu):
    cmd=['docker','run','--rm','--name',f'darkpipe-lowconf-20261101-{gpu}','--gpus',f'device={gpu}',
        '--ipc=host','--network','none','--user',f'{os.getuid()}:{os.getgid()}',
        '--entrypoint','/opt/conda/envs/darkpipe/bin/python','-e','PYTHONUNBUFFERED=1',
        '-e','TRITON_CACHE_DIR=/tmp/lowconf-triton','-e','XDG_CACHE_HOME=/tmp/lowconf-cache',
        '-v',f'{ROOT}:{ROOT}:ro','-v',f'{OUT}:{OUT}:rw','-w',str(OP),'darkpipe-operator:0.3.0',
        '-u',str(OUT/'scripts/frontend.py'),'--shard',str(gpu)]
    save(OUT/'jobs'/f'frontend-{gpu}.json',dict(command=cmd,started=time.time()))
    with (OUT/'jobs'/f'frontend-{gpu}.log').open('w')as log:rc=subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT)
    return dict(gpu=gpu,exit_code=rc)

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=6)as pool:results=list(pool.map(run,range(6)))
    save(OUT/'frontend-jobs.json',results)
    assert all(r['exit_code']==0 for r in results),results
