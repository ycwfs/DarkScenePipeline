"""Independent raw sources in the original image; paired rules share all inference."""
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
from common import OUT, ROOT, save


def run(number,gpu):
    for short in [False,True]:
        cmd=['docker','run','--rm','--network','none','--gpus',f'device={gpu}','--ipc=host',
            '-v',f'{ROOT}:{ROOT}','-v',f'{ROOT}/platform/dark-demo-v2:/app:ro',
            '--entrypoint','/opt/conda/envs/darkpipe/bin/python','darkpipe-operator:0.3.0',
            str(OUT/'scripts/paired_video.py'),'--number',str(number)]
        if short:cmd.append('--prior-demo')
        scope='short'if short else 'long'
        with (OUT/f'video-{number:02}-{scope}.log').open('w')as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('FINISHED',number,scope,flush=True)


def main():
    assert (OUT/'protocol.json').exists()
    with ThreadPoolExecutor(max_workers=3)as pool:
        jobs=[pool.submit(run,n,g)for n,g in [(1,0),(7,1),(9,2)]]
        for job in jobs:job.result()
    save(OUT/'video-runs-complete.json',dict(passed=True,full=[1,7,9],previous_short=[1,7,9]))


if __name__=='__main__':main()
