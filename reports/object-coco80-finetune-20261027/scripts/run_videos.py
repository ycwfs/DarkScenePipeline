"""Run independent sources on separate GPUs after the candidate export is verified."""
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import time

from common import OUT, ROOT, save


def run(number, gpu):
    for short in [False,True]:
        command=['docker','run','--rm','--network','none','--gpus',f'device={gpu}','--ipc=host',
            '-v',f'{ROOT}:{ROOT}','-v',f'{ROOT}/platform/dark-demo-v2:/app:ro',
            '--entrypoint','/opt/conda/envs/darkpipe/bin/python','darkpipe-operator:0.3.0',
            str(OUT/'scripts/paired_long.py'),'--number',str(number)]
        if short:command.append('--prior-demo')
        suffix='short' if short else 'long'
        with (OUT/f'video-{number:02}-{suffix}.log').open('w')as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('VIDEO FINISHED',number,suffix,flush=True)


def main():
    while not (OUT/'export-verification.json').exists():time.sleep(10)
    assert json.loads((OUT/'export-verification.json').read_text())['passed']
    with ThreadPoolExecutor(max_workers=3)as pool:
        jobs=[pool.submit(run,n,g)for n,g in [(1,2),(7,4),(9,5)]]
        for job in jobs:job.result()
    cmd=['docker','run','--rm','--network','none','--gpus','device=4','--ipc=host',
        '-v',f'{ROOT}:{ROOT}','-v',f'{ROOT}/platform/dark-demo-v2:/app:ro',
        '--entrypoint','/opt/conda/envs/darkpipe/bin/python','darkpipe-operator:0.3.0',str(OUT/'scripts/robot_probe.py')]
    if not (OUT/'robot-probe.json').exists():
        with (OUT/'robot-probe.log').open('w')as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    save(OUT/'video-runs-complete.json',dict(passed=True,numbers=[1,7,9],full=True,previous_short=True,robot_probe=True))


if __name__=='__main__':main()
