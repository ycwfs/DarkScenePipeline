import os,subprocess,time
from common import *

cmd=['docker','run','--rm','--name','darkpipe-other-action-objects-20261101','--gpus','device=5',
    '--ipc=host','--network','none','--user',f'{os.getuid()}:{os.getgid()}',
    '--entrypoint','/opt/conda/envs/darkpipe/bin/python','-e','PYTHONUNBUFFERED=1',
    '-e','TRITON_CACHE_DIR=/tmp/other-action-triton','-e','XDG_CACHE_HOME=/tmp/other-action-cache',
    '-v',f'{ROOT}:{ROOT}:ro','-v',f'{OUT}:{OUT}:rw','-w',str(OP),'darkpipe-operator:0.3.0',
    '-u',str(OUT/'scripts/object_infer.py')]
save(OUT/'jobs/object-command.json',dict(command=cmd,started=time.time()))
with(OUT/'jobs/object-infer.log').open('w')as log:code=subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT)
save(OUT/'jobs/object-completion.json',dict(exit_code=code,finished=time.time()))
raise SystemExit(code)
