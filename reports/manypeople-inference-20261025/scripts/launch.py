"""Launch the frozen release in the original image on physical GPUs 1 and 0."""
import json
import os
from pathlib import Path
import subprocess
import time


def main() -> None:
    report = Path(__file__).resolve().parents[1]
    root = report.parents[1]
    data = root / 'data-10-25-manypeople'
    operator = report / 'operator/dark-demo-v2'
    name = 'darkpipe-manypeople-20261025'
    command = ['docker', 'run', '--rm', '--name', name, '--gpus', '"device=0,1"',
        '--ipc=host', '--user', f'{os.getuid()}:{os.getgid()}',
        '--entrypoint', '/opt/conda/envs/darkpipe/bin/python',
        '-e', f'BATCH_REPORT={report}', '-e', 'TRITON_CACHE_DIR=/tmp/darkpipe-triton',
        '-e', 'XDG_CACHE_HOME=/tmp/darkpipe-cache', '-e', 'PYTHONUNBUFFERED=1',
        '-e', 'CUDA_DEVICE_ORDER=PCI_BUS_ID',
        '-v', f'{operator}:/app:ro', '-v', f'{data}:{data}:ro',
        '-v', f'{report}:{report}:rw', '-w', '/app', 'darkpipe-operator:0.3.0',
        '-u', str(report / 'scripts/run_observed.py'),
        '--video_path', str(data / 'extracted'), '--gpu_ids', '1,0',
        '--clip_dir', str(report / 'results'), '--session_json', str(report / 'session.json'),
        '--extra_hosts', '']
    # Visible devices are physical cards 0 and 1; select 1 for the offline frontend.
    protocol = dict(command=command, physical_frontend_gpu=1, physical_action_gpu=0,
        image_id=subprocess.check_output(['docker', 'image', 'inspect',
            'darkpipe-operator:0.3.0', '--format', '{{.Id}}'], text=True).strip(),
        started_at=time.time())
    (report / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+'\n')
    with (report / 'operator.log').open('w') as log:
        code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
    protocol.update(exit_code=code, stopped_at=time.time())
    (report / 'completion.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+'\n')
    raise SystemExit(code)


if __name__ == '__main__':
    main()
