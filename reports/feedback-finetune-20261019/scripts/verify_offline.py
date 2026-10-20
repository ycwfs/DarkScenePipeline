"""Validate full offline decoding, clip finalization and saved pose events."""
import json
from pathlib import Path
import subprocess
import cv2
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reports/feedback-finetune-20261019'
session=json.loads((OUT/'real-offline/session.json').read_text())
assert session['status']=='finished'
assert session['stats']['frames']==session['stats']['frames_written']==75
assert session['events_total']==4
for key in ['clips_abandoned','clips_forced','frames_dropped','clips_unfinished','clips_failed','write_errors']:
    assert session['clips'][key]==0,(key,session['clips'][key])
events=json.loads(Path(session['events_json'][0]).read_text())
assert all(e['label']=='Drinking water' and e['skeleton_active'] and e['actor_poses'] for e in events)
results=[]
for p in sorted((OUT/'real-offline/delivered').rglob('*.mp4')):
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,nb_frames,r_frame_rate:format=duration','-of','json',str(p)]))
    st=probe['streams'][0]
    assert st['codec_name']=='h264' and (st['width'],st['height'])==(1440,874)
    cap=cv2.VideoCapture(str(p));count=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        if p.name.endswith('处理后.mp4') and count==40:
            cv2.imwrite(str(OUT/'real-offline/frame-0040.jpg'),frame)
        count+=1
    cap.release()
    assert count==int(st['nb_frames'])
    if p.name.endswith('处理后.mp4'):assert count==75
    else:
        meta=json.loads(p.with_suffix('.json').read_text())
        assert count==meta['frames_written'] and meta['label_key']=='drinking_water'
    results.append(dict(file=str(p.relative_to(OUT)),frames=count,**probe))
result=dict(status='passed',input_frames=75,full_output_frames=75,events=4,clips=session['clips'],videos=results)
(OUT/'real-offline/verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print([(r['frames'],r['streams'][0]['codec_name'],r['format']['duration']) for r in results])
