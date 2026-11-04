"""Resume completed/running jobs and start remaining independent recordings."""
import concurrent.futures
import json
import subprocess
import time

import resume_balanced as runner


def adopt(number: int, front: int) -> dict:
    job=runner.REPORT/'jobs'/f'{number:02d}'
    protocol=json.loads((job/'protocol.json').read_text())
    protocol['adopted_without_restart']=True
    result=subprocess.run(['docker','wait',f'darkpipe-manypeople-20261025-{number:02d}'],
        capture_output=True,text=True)
    code=int(result.stdout.strip()) if result.returncode==0 else 0 if (job/'completed-items.jsonl').exists() else 1
    if result.returncode: protocol['wait_message']=result.stderr
    with runner.LOCK: runner.ACTIVE_FRONTENDS[front]-=1
    return runner.collect(number,protocol,code)


def main() -> None:
    report=runner.REPORT
    sources=json.loads((report/'input-manifest.json').read_text())
    protocol=json.loads((report/'protocol.json').read_text())
    protocol.update(concurrency=6,scheduler_updated_at=time.time(),
        allocation='Complete files on physical GPUs 0/1; existing jobs retained, remaining files balanced by frontend count')
    runner.save(report/'protocol.json',protocol)
    active=[];pending=[];results=[]
    runner.ACTIVE_FRONTENDS.update({0:0,1:0})
    for number in range(1,len(sources)+1):
        job=report/'jobs'/f'{number:02d}'
        if (job/'completion.json').exists():
            item=json.loads((job/'completion.json').read_text())
            assert item['exit_code']==0 and not item.get('artifact_missing'),number
            results.append(item)
        elif (job/'protocol.json').exists():
            args=json.loads((job/'effective-arguments.json').read_text())
            front=int(args['gpu_ids'].split(',')[0])
            runner.ACTIVE_FRONTENDS[front]+=1
            active.append((number,front))
        else:
            pending.append(number)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(adopt,*item) for item in active]
        futures += [pool.submit(runner.launch,number,sources) for number in pending]
        results += [f.result() for f in futures]
    protocol.update(stopped_at=time.time(),jobs=results,
        success=all(r['exit_code']==0 and not r.get('artifact_missing') for r in results))
    runner.save(report/'completion.json',protocol)
    raise SystemExit(0 if protocol['success'] else 1)


if __name__=='__main__':
    main()
