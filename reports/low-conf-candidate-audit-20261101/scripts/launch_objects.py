import os,subprocess,time
from common import *
cmd=['docker','run','--rm','--name','darkpipe-lowconf-objects-20261101','--gpus','device=5','--ipc=host','--network','none','--user',f'{os.getuid()}:{os.getgid()}',
    '--entrypoint','/opt/conda/envs/darkpipe/bin/python','-e','PYTHONUNBUFFERED=1','-e','TRITON_CACHE_DIR=/tmp/lowconf-triton',
    '-v',f'{ROOT}:{ROOT}:ro','-v',f'{OUT}:{OUT}:rw','-w',str(OP),'darkpipe-operator:0.3.0','-u',str(OUT/'scripts/objects.py')]
save(OUT/'jobs/objects.json',dict(command=cmd,started=time.time()))
with(OUT/'jobs/objects.log').open('w')as log:rc=subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT)
save(OUT/'objects-job.json',dict(exit_code=rc,finished=time.time()))
raise SystemExit(rc)
