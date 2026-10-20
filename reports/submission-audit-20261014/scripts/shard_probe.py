import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
source = '/ssd1/wfs/project/lowlight/DarkScenePipeline/real-actiondata-0909/喝水16.mp4'
video = root / 'shard-input.mp4'
subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                '-stream_loop', '3', '-i', source, '-an', '-c:v', 'copy', str(video)], check=True)
out = root / 'offline-shard-long'
out.mkdir(exist_ok=True)
cmd = json.loads((root / 'offline-shard/command.json').read_text())
cmd[cmd.index('--name') + 1] = 'darkpipe-audit-shard-long'
for flag, value in {'--video_path': '/audit/shard-input.mp4',
                     '--session_json': '/audit/offline-shard-long/session.json',
                     '--hdfs_output_dir': '/audit/offline-shard-long/copy'}.items():
    cmd[cmd.index(flag) + 1] = value
(out / 'command.json').write_text(json.dumps(cmd, ensure_ascii=False, indent=2))
with (out / 'container.log').open('w') as log:
    res = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=120)
report = {'source_frames': 300, 'minimum_frames_for_two_shards': 240,
          'exit_code': res.returncode,
          'log': (out / 'container.log').read_text(),
          'session': json.loads((out / 'session.json').read_text()) if (out / 'session.json').exists() else None}
(root / 'shard-probe.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(res.returncode, report['log'][-3500:], flush=True)
