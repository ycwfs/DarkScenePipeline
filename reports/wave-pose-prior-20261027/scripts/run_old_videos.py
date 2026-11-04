from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
from common import ROOT, OUT, save


def shard(items,gpu):
    for v in items:
        cmd=['docker','run','--rm','--network','none','--gpus',f'device={gpu}','--ipc=host',
            '-v',f'{ROOT}:{ROOT}','-v',f'{ROOT}/platform/dark-demo-v2:/app:ro',
            '--entrypoint','/opt/conda/envs/darkpipe/bin/python','darkpipe-operator:0.3.0',
            str(OUT/'scripts/paired_video.py'),'--number','0','--old-id',v['id']]
        with (OUT/f'old-{v["id"]}.log').open('w')as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('FINISHED',v['source'],flush=True)


def main():
    rows=json.loads((OUT/'old-demo-selection.json').read_text())['sources']
    with ThreadPoolExecutor(max_workers=3)as pool:
        jobs=[pool.submit(shard,rows[i::3],i+3)for i in range(3)]
        for job in jobs:job.result()
    save(OUT/'old-video-runs-complete.json',dict(passed=True,sources=[r['id']for r in rows]))


if __name__=='__main__':main()
