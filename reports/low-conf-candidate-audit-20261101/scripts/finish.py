import subprocess,sys,time
from common import *
start=time.monotonic()
while not(OUT/'frontend-jobs.json').exists()or not(OUT/'objects-complete.json').exists():
    if time.monotonic()-start>7200:raise TimeoutError('GPU supplement incomplete')
    if(OUT/'frontend-jobs.json').exists():assert all(r['exit_code']==0 for r in read(OUT/'frontend-jobs.json'))
    if(OUT/'objects-job.json').exists():assert read(OUT/'objects-job.json')['exit_code']==0
    time.sleep(10)
for name in ('evaluate.py','verify.py','report.py'):
    print('START',name,flush=True)
    with(OUT/(name.replace('.py','')+'.log')).open('w')as f:
        subprocess.run([sys.executable,str(OUT/'scripts'/name)],stdout=f,stderr=subprocess.STDOUT,check=True)
    print('DONE',name,flush=True)
save(OUT/'analysis-complete.json',dict(complete=True,seconds=time.monotonic()-start,production_changed=False))
