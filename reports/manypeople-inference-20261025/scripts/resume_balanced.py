"""Adopt ongoing owned containers; balance new complete videos across both GPUs."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from launch_parallel import REPORT, ROOT, save

LOCK=threading.Lock()
ACTIVE_FRONTENDS={0:0,1:3}


def collect(number: int, protocol: dict, code: int) -> dict:
    job=REPORT/'jobs'/f'{number:02d}'
    protocol.update(exit_code=code,stopped_at=time.time())
    path=job/'completed-items.jsonl'
    if path.exists():
        items=[json.loads(s) for s in path.read_text().splitlines()]
        with LOCK:
            existing=REPORT/'completed-items.jsonl'
            seen={json.loads(s)['file'] for s in existing.read_text().splitlines()} if existing.exists() else set()
            with existing.open('a') as stream:
                for item in items:
                    if item['file'] not in seen:
                        item['batch_number']=number
                        stream.write(json.dumps(item,ensure_ascii=False)+'\n')
    else:
        protocol['artifact_missing']=True
    save(job/'completion.json',protocol)
    print('DONE',number,code,'artifact',path.exists(),flush=True)
    return protocol


def adopt(number: int) -> dict:
    job=REPORT/'jobs'/f'{number:02d}'
    protocol=json.loads((job/'protocol.json').read_text())
    protocol['adopted_without_restart']=True
    name=f'darkpipe-manypeople-20261025-{number:02d}'
    result=subprocess.run(['docker','wait',name],capture_output=True,text=True)
    code=int(result.stdout.strip()) if result.returncode==0 else 0 if (job/'completed-items.jsonl').exists() else 1
    if result.returncode:
        protocol['wait_message']=result.stderr
    with LOCK: ACTIVE_FRONTENDS[1]-=1
    return collect(number,protocol,code)


def launch(number: int, sources: list) -> dict:
    source=sources[number-1]
    job=REPORT/'jobs'/f'{number:02d}';job.mkdir(exist_ok=True)
    with LOCK:
        front=min(ACTIVE_FRONTENDS,key=lambda g:ACTIVE_FRONTENDS[g])
        ACTIVE_FRONTENDS[front]+=1
    action=1-front
    command=['docker','run','--rm','--name',f'darkpipe-manypeople-20261025-{number:02d}',
        '--gpus','"device=0,1"','--ipc=host','--user',f'{os.getuid()}:{os.getgid()}',
        '--entrypoint','/opt/conda/envs/darkpipe/bin/python',
        '-e',f'BATCH_REPORT={job}','-e','TRITON_CACHE_DIR=/tmp/darkpipe-triton',
        '-e','XDG_CACHE_HOME=/tmp/darkpipe-cache','-e','PYTHONUNBUFFERED=1',
        '-e','CUDA_DEVICE_ORDER=PCI_BUS_ID',
        '-v',f'{REPORT / "operator/dark-demo-v2"}:/app:ro',
        '-v',f'{ROOT / "data-10-25-manypeople"}:{ROOT / "data-10-25-manypeople"}:ro',
        '-v',f'{REPORT}:{REPORT}:rw','-w','/app','darkpipe-operator:0.3.0',
        '-u',str(REPORT/'scripts/run_observed.py'),'--video_path',source['absolute_path'],
        '--gpu_ids',f'{front},{action}','--clip_dir',str(job/'results'),
        '--session_json',str(job/'session.json'),'--extra_hosts','']
    protocol=dict(number=number,source=source,command=command,physical_frontend_gpu=front,
        physical_action_gpu=action,started_at=time.time())
    save(job/'protocol.json',protocol)
    print('START',number,'front',front,'action',action,flush=True)
    try:
        with (job/'operator.log').open('w') as stream:
            code=subprocess.call(command,stdout=stream,stderr=subprocess.STDOUT)
    finally:
        with LOCK: ACTIVE_FRONTENDS[front]-=1
    return collect(number,protocol,code)


def main() -> None:
    sources=json.loads((REPORT/'input-manifest.json').read_text())
    protocol=json.loads((REPORT/'protocol.json').read_text())
    protocol.update(concurrency=4,scheduler_updated_at=time.time(),
        allocation='Initial 01/02/06 frontend=1/action=0; remaining complete videos balanced across 0/1')
    save(REPORT/'protocol.json',protocol)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(adopt,n) for n in (1,2,6)]
        futures += [pool.submit(launch,n,sources) for n in (3,7,8,9,4,5)]
        results=[f.result() for f in futures]
    protocol.update(stopped_at=time.time(),jobs=results,
        success=all(r['exit_code']==0 and not r.get('artifact_missing') for r in results))
    save(REPORT/'completion.json',protocol)
    raise SystemExit(0 if protocol['success'] else 1)


if __name__=='__main__':
    main()
