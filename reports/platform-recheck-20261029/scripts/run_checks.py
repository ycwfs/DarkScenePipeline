"""Run a single explicitly selected check with owned processes and frozen archive mounts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
OP = OUT / 'release-extracted/dark-demo-v2'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['cpu', 'live', 'faults', 'demos'])
    args = parser.parse_args()
    name = f'darkpipe-platform-recheck-{args.stage}-{os.getpid()}'
    base = ['docker', 'run', '--rm', '--name', name, '--network', 'host', '--ipc=host',
            '--entrypoint', '/opt/conda/envs/darkpipe/bin/python',
            '-v', str(OP)+':/app:ro', '-v', str(ROOT)+':/repo:ro',
            '-v', str(OP)+':/repo/platform/dark-demo-v2:ro',
            '-v', str(OP/'oputil.py')+':/repo/platform/oputil.py:ro',
            '-v', str(OUT)+':/evidence:rw', '-w', '/app']
    if args.stage == 'cpu':
        command = base + ['-v', str(ROOT/'reports/person-object-release-20261026/test-tools')+':/test-tools:ro',
                          '-e', 'DARKPIPE_V2_OPERATOR_DIR=/app', 'darkpipe-operator:0.3.0',
                          '-u', '/evidence/scripts/archive_cpu.py']
    elif args.stage == 'demos':
        command = base + ['--gpus', 'device=4', 'darkpipe-operator:0.3.0', '-u', '/evidence/scripts/demo.py']
    else:
        command = [sys.executable, str(OUT/'scripts'/('run_live.py' if args.stage == 'live' else 'run_faults.py')),
                   '--cards', '0,1,2']
    start = time.time()
    record = dict(stage=args.stage, command=command, started_unix=start)
    path = OUT / f'{args.stage}-command.json'
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    monitor = None
    with (OUT/f'{args.stage}-run.log').open('w') as log:
        try:
            if args.stage in ('live', 'faults'):
                monitor = subprocess.Popen(['nvidia-smi', '--id=0,1,2',
                    '--query-gpu=timestamp,index,memory.used,utilization.gpu,temperature.gpu,power.draw',
                    '--format=csv', '--loop=10', '--filename='+str(OUT/f'{args.stage}-gpu.csv')])
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        finally:
            if monitor:
                monitor.terminate()
                monitor.wait(timeout=10)
    record.update(returncode=result.returncode, elapsed_seconds=time.time()-start)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
