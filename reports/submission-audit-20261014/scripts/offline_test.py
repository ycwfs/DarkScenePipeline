"""Run the archived main.py in the loaded image; record command, exit and outputs."""
import json
from pathlib import Path
import shutil
import subprocess

root = Path(__file__).resolve().parent
pkg = root / 'package/dark-demo-v2'
repo = Path('/ssd1/wfs/project/lowlight/DarkScenePipeline')
doc = json.loads((pkg / 'suanzi.json').read_text())
inputs = root / 'offline-input'
inputs.mkdir(exist_ok=True)
for stem in ('挥手22', '捡东西2', '追逐16', '喝水16'):
    target = inputs / (stem + '.mp4')
    if not target.exists():
        target.symlink_to(repo / 'real-actiondata-0909' / target.name)
cases = [
    ('offline-batch', '/audit/offline-input', {}),
    ('offline-dark', str(repo / 'real-actiondata-0909/追逐19.mp4'), {}),
    ('offline-shard', str(repo / 'real-actiondata-0909/other/other78.mp4'),
     {'clip_dir': '', 'gpu_ids': '0,1'})
]
report = {}
for tag, src, overrides in cases:
    out = root / tag
    out.mkdir(exist_ok=True)
    values = {i['name']: i.get('default', '') for i in doc['inputs']}
    values.update(video_path=src, clip_dir='/audit/' + tag + '/output',
                  hdfs_output_dir='/audit/' + tag + '/copy',
                  session_json='/audit/' + tag + '/session.json', extra_hosts='')
    values.update(overrides)
    argv = []
    for item in doc['implementation']['container']['args']:
        if isinstance(item, str):
            argv.append(item)
        else:
            value = values[list(item.values())[0]]
            argv.append(str(value).lower() if isinstance(value, bool) else str(value))
    gpus = '"device=0,1"' if tag == 'offline-shard' else 'device=1'
    cmd = ['docker', 'run', '--rm', '--name', 'darkpipe-audit-' + tag,
           '--gpus', gpus, '--cpus', '8', '--memory', '12g',
           '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'OMP_NUM_THREADS=4',
           '-e', 'MKL_NUM_THREADS=4', '-e', 'OPENBLAS_NUM_THREADS=1',
           '-v', str(pkg) + ':/opt/darkpipe/op:ro', '-v', str(root) + ':/audit',
           '-v', str(repo) + ':' + str(repo) + ':ro', '-w', '/opt/darkpipe/op',
           doc['implementation']['container']['image'],
           *doc['implementation']['container']['command'], *argv]
    (out / 'command.json').write_text(json.dumps(cmd, ensure_ascii=False, indent=2))
    with (out / 'container.log').open('w') as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=900)
    record = {'exit_code': result.returncode, 'source': src}
    if (out / 'session.json').exists():
        record['session'] = json.loads((out / 'session.json').read_text())
    record['files'] = [str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()]
    report[tag] = record
    (root / 'offline-summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(tag, 'exit', result.returncode,
          'status', record.get('session', {}).get('status'), flush=True)
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
