"""Run independent complete recordings concurrently using the original release."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import threading
import time

REPORT = Path(__file__).resolve().parents[1]
ROOT = REPORT.parents[1]
LOCK = threading.Lock()


def save(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def run(pair: tuple) -> dict:
    number, source = pair
    job = REPORT/'jobs'/f'{number:02d}'
    job.mkdir(exist_ok=True)
    name = f'darkpipe-manypeople-20261025-{number:02d}'
    command = ['docker','run','--rm','--name',name,'--gpus','"device=0,1"',
        '--ipc=host','--user',f'{os.getuid()}:{os.getgid()}',
        '--entrypoint','/opt/conda/envs/darkpipe/bin/python',
        '-e',f'BATCH_REPORT={job}','-e','TRITON_CACHE_DIR=/tmp/darkpipe-triton',
        '-e','XDG_CACHE_HOME=/tmp/darkpipe-cache','-e','PYTHONUNBUFFERED=1',
        '-e','CUDA_DEVICE_ORDER=PCI_BUS_ID',
        '-v',f'{REPORT / "operator/dark-demo-v2"}:/app:ro',
        '-v',f'{ROOT / "data-10-25-manypeople"}:{ROOT / "data-10-25-manypeople"}:ro',
        '-v',f'{REPORT}:{REPORT}:rw','-w','/app','darkpipe-operator:0.3.0',
        '-u',str(REPORT/'scripts/run_observed.py'),
        '--video_path',source['absolute_path'],'--gpu_ids','1,0',
        '--clip_dir',str(job/'results'),'--session_json',str(job/'session.json'),'--extra_hosts','']
    protocol=dict(number=number,source=source,command=command,started_at=time.time())
    save(job/'protocol.json',protocol)
    print('START',number,source['absolute_path'],flush=True)
    with (job/'operator.log').open('w') as log:
        code=subprocess.call(command,stdout=log,stderr=subprocess.STDOUT)
    protocol.update(exit_code=code,stopped_at=time.time())
    save(job/'completion.json',protocol)
    completed=job/'completed-items.jsonl'
    if completed.exists():
        items=[json.loads(s) for s in completed.read_text().splitlines()]
        with LOCK:
            with (REPORT/'completed-items.jsonl').open('a') as f:
                for item in items:
                    item['batch_number']=number
                    f.write(json.dumps(item,ensure_ascii=False)+'\n')
    else:
        protocol['artifact_missing']=True
    print('DONE',number,'exit',code,'artifact',completed.exists(),flush=True)
    return protocol


def main() -> None:
    sources=json.loads((REPORT/'input-manifest.json').read_text())
    protocol=dict(image_id=subprocess.check_output(['docker','image','inspect','darkpipe-operator:0.3.0',
        '--format','{{.Id}}'],text=True).strip(),started_at=time.time(),concurrency=3,
        physical_frontend_gpu=1,physical_action_gpu=0)
    save(REPORT/'protocol.json',protocol)
    # Include a short recording in the first group to provide a complete result early.
    order=[1,5,0,2,6,7,3,4,8]
    tasks=[(i+1,sources[i]) for i in order]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(run,tasks))
    protocol.update(stopped_at=time.time(),jobs=results,
        success=all(r['exit_code']==0 and not r.get('artifact_missing') for r in results))
    save(REPORT/'completion.json',protocol)
    raise SystemExit(0 if protocol['success'] else 1)


if __name__=='__main__':
    main()
