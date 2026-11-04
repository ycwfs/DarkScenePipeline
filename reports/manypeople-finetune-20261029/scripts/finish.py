"""Continue after training: parity, scientific report, real service videos and decode."""
import argparse
import sys
import time
import subprocess
from common import *
from orchestrate import JOBS,launch,wait_jobs


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--control-gpu',type=int,default=3)
    parser.add_argument('--demo-gpus',default='0,1,2')
    parser.add_argument('--attempt',default='')
    args=parser.parse_args()
    demo_gpus=[int(g) for g in args.demo_gpus.split(',')]
    def tag(name):return name+('-'+args.attempt if args.attempt else '')
    if not demo_gpus or len(set(demo_gpus))!=len(demo_gpus):
        parser.error('demo-gpus must contain distinct GPU indices')
    start=time.monotonic()
    while not (OUT/'pipeline-complete.json').exists():
        if (OUT/'pipeline-error.json').exists():raise RuntimeError(read(OUT/'pipeline-error.json'))
        if not Path('/proc/'+str(read(OUT/'orchestrator-state.json')['pid'])).exists():
            jobs=read(OUT/'pipeline-jobs.json')
            bad=[]
            for name in ('train-rgb','train-skeleton','joint-select','predict-rgb-selected','predict-skeleton-selected','evaluate-selected'):
                if name in jobs:
                    state=subprocess.run(['docker','inspect','--format','{{.State.ExitCode}} {{.State.Running}}',jobs[name]['name']],capture_output=True,text=True)
                    bad.append((name,state.stdout.strip(),state.stderr.strip()))
            if any('1' in row[1].split()[:1] for row in bad):
                save(OUT/'finish-error.json',dict(stage_status=bad,reason='Orchestrator stopped before pipeline-complete'))
                raise RuntimeError('Orchestrator stopped before pipeline-complete')
        if time.monotonic()-start>21600:raise TimeoutError('Training coordinator has not finished in six hours')
        time.sleep(20)
    JOBS.update(read(OUT/'pipeline-jobs.json'))
    launch('verify-replay',args.control_gpu,'verify_replay.py');wait_jobs([JOBS['verify-replay']['name']])
    # Figures use host Matplotlib; the production image intentionally has no plotting dependency.
    with (OUT/'report-host.log').open('a') as log:
        subprocess.run([sys.executable,str(OUT/'scripts/report.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
    launch(tag('build-candidate'),args.control_gpu,'build_candidate.py');wait_jobs([JOBS[tag('build-candidate')]['name']])
    for shard,gpu in enumerate(demo_gpus):launch(tag(f'demos-{shard}'),gpu,'demo.py','--shard',f'{shard}/{len(demo_gpus)}')
    wait_jobs([JOBS[tag(f'demos-{i}')]['name']for i in range(len(demo_gpus))])
    launch(tag('verify-demos'),args.control_gpu,'verify_demos.py');wait_jobs([JOBS[tag('verify-demos')]['name']])
    save(OUT/'completion.json',dict(complete=True,seconds=time.monotonic()-start,
        selection_lock_sha256=sha(OUT/'selection-lock.json'),models=read(OUT/'candidate-config.json')['models'],production_modified=False))
    print('FINISHED',flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as e:
        save(OUT/'finish-error.json',dict(error=repr(e),time=time.time()));raise
