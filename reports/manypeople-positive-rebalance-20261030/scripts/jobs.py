"""Launch explicit experiment jobs; immutable repository mount, writable run only."""
import argparse
import os
import shlex
import subprocess
import time
from common import *


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('tag')
    parser.add_argument('gpus')
    parser.add_argument('script')
    parser.add_argument('arguments', nargs='*')
    parser.add_argument('--ddp', type=int, default=1)
    args = parser.parse_args()
    state_path = OUT / 'jobs.json'
    jobs = read(state_path) if state_path.exists() else {}
    assert args.tag not in jobs, 'Keep previous attempts; use a distinct tag'
    name = 'mp-rb30-' + args.tag
    gpu_arg = f'"device={args.gpus}"' if ',' in args.gpus else f'device={args.gpus}'
    python = ['/opt/conda/envs/darkpipe/bin/python', '-u']
    if args.ddp > 1:
        assert len(args.gpus.split(',')) == args.ddp
        python += ['-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={args.ddp}']
    command = python + [str(OUT / 'scripts' / args.script), *args.arguments]
    shell = shlex.join(command) + ' > ' + shlex.quote(str(OUT / f'{args.tag}.log')) + ' 2>&1'
    cmd = ['docker', 'run', '-d', '--name', name, '--gpus', gpu_arg, '--shm-size', '4g',
           '--user', f'{os.getuid()}:{os.getgid()}', '-v', f'{ROOT}:{ROOT}:ro',
           '-v', f'{OUT}:{OUT}:rw', '-w', str(ROOT), '-e', 'OMP_NUM_THREADS=3',
           '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'XDG_CACHE_HOME=/tmp/rb30-cache',
           '-e', 'TRITON_CACHE_DIR=/tmp/rb30-triton', '--entrypoint', '/bin/bash',
           'darkpipe-operator:0.3.0', '-c', shell]
    cid = subprocess.check_output(cmd, text=True).strip()
    jobs[args.tag] = dict(name=name, container=cid, gpus=args.gpus, command=command,
                         started=time.time(), script_sha256=sha(OUT / 'scripts' / args.script))
    save(state_path, jobs)
    print(json.dumps(jobs[args.tag]))


if __name__ == '__main__':
    main()
