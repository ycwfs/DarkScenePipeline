"""Sequential single-GPU verification of the independently extracted candidate."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]
BOX=OUT/'release-extracted/NVIDIA-Box'
env=os.environ.copy()
env.update(PYTHONUTF8='1',CUDA_VISIBLE_DEVICES='0',DARKPIPE_BOX_BACKEND='tensorrt',
    DARKPIPE_BOX_PROFILE='balanced',DARKPIPE_BOX_ACTION_BATCH='2',DARKPIPE_BOX_GRAPHS='0',
    BOX_RUNTIME=str(BOX/'runtime'),BOX_WEIGHTS=str(BOX/'weights'),BOX_TEST_SR='off',
    CUDA_MODULE_LOADING='LAZY',TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT='256',
    DARKPIPE_BOX_ENGINE_CACHE=str(OUT/'trt-engines'))
env['LD_LIBRARY_PATH']=':'.join(str(ROOT/'.venv/lib/python3.10/site-packages/nvidia'/n/'lib')for n in ('cudnn','cublas','cuda_runtime'))
commands=[
 ('archive-check-env',[str(BOX/'check_env.py')]),
 ('archive-entry',[str(BOX/'benchmark_box.py'),'--video',str(ROOT/'real-actiondata-all/喝水27.mp4'),'--output-dir',str(OUT/'archive-entry'),'--backend','tensorrt']),
 ('archive-demos',[str(OUT/'scripts/bench_offline.py'),'--runtime',str(BOX/'runtime'),'--name','archive-demos','--ids','all','--max-frames','0','--sr-scale','1']),
 ('archive-probe',[str(OUT/'scripts/bench_offline.py'),'--runtime',str(BOX/'runtime'),'--name','archive-probe','--max-frames','150','--sr-scale','1']),
 ('archive-faults',[str(OUT/'scripts/run_faults.py')])]
results=[]
for name,args in commands:
    command=[sys.executable,'-u',*args]
    print('START',name,flush=True)
    start=time.monotonic()
    with (OUT/(name+'.log')).open('w')as log:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    results.append(dict(name=name,command=command,elapsed_seconds=time.monotonic()-start,returncode=result.returncode))
    (OUT/'final-suite.json').write_text(json.dumps(dict(environment={k:env[k]for k in ('CUDA_VISIBLE_DEVICES','DARKPIPE_BOX_BACKEND','DARKPIPE_BOX_PROFILE','DARKPIPE_BOX_ACTION_BATCH','DARKPIPE_BOX_GRAPHS','TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT')},results=results),ensure_ascii=False,indent=2)+'\n')
    print('END',name,result.returncode,flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
