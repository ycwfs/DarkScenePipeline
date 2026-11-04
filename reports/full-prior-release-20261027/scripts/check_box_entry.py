"""Exercise the independently extracted native Box entry, then check real artifacts."""
import json
import os
from pathlib import Path
import subprocess
import sys

OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parents[1]
box=OUT/'release-extracted/NVIDIA-Box'
out=OUT/'box-native';out.mkdir(exist_ok=True)
command=[sys.executable,'-u',str(box/'run.py'),'--video_path',str(ROOT/'real-actiondata-all/喝水27.mp4'),
    '--gpu_ids','0','--output-dir',str(out/'output'),'--session_json',str(out/'session.json')]
(out/'command.json').write_text(json.dumps(command,ensure_ascii=False,indent=2)+'\n')
with (out/'operator.log').open('w')as log:
    run=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
assert run.returncode==0,run.returncode
session=json.loads((out/'session.json').read_text())
videos=list((out/'output').rglob('*.mp4'))
assert videos
for video in videos:
    subprocess.run(['ffmpeg','-nostdin','-v','error','-xerror','-i',str(video),'-f','null','-'],check=True)
event_files=[p for p in (out/'output').rglob('*.json') if p.name.endswith('events.json') or '事件' in p.name]
result=dict(passed=True,exit_code=run.returncode,session=session,decoded_videos=[str(p.relative_to(OUT))for p in videos],event_files=[str(p.relative_to(OUT))for p in event_files])
(out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('NATIVE BOX ENTRY PASSED',len(videos),'decoded videos',flush=True)
