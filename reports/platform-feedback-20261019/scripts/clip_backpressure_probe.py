"""Real quality-denoise/H.264 recorder under queue saturation; synthetic event, no ML."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import cv2

ap=argparse.ArgumentParser()
ap.add_argument('--op-dir',required=True); ap.add_argument('--frame',required=True)
ap.add_argument('--out',required=True)
a=ap.parse_args();sys.path.insert(0,a.op_dir)
from darkpipe.clips import ClipRecorder
from darkpipe.stages.base import RecognitionEvent

cv2.setNumThreads(1)
frame=cv2.imread(a.frame);assert frame is not None
out=Path(a.out);out.mkdir(exist_ok=True)
rec=ClipRecorder(str(out),pre_sec=2,post_sec=2,denoise='quality',queue_frames=8,
                 session='clips',stage_dir=str(out/'stage'))
times=[];clean_times=[];peak=0;original=rec._clean

def clean(image):
    start=time.perf_counter(); result=original(image)
    clean_times.append(time.perf_counter()-start); return result

rec._clean=clean
start=time.perf_counter()
try:
    for i in range(400):
        time.sleep(max(0,start+i/25-time.perf_counter()))
        event=RecognitionEvent(i,i/25,'Waving',.9) if i==50 else None
        image=frame.copy();began=time.perf_counter()
        rec.push(image,event,time.time())
        times.append(time.perf_counter()-began);peak=max(peak,rec._q.qsize())
finally:
    rec.close()
media=[]
for video in Path(rec.root).rglob('*.mp4'):
    result=subprocess.run(['ffprobe','-v','error','-show_entries','stream=codec_name,width,height,nb_frames,duration,avg_frame_rate','-of','json',str(video)],capture_output=True,text=True,check=True)
    decode=subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],capture_output=True,text=True,timeout=60)
    assert decode.returncode==0,decode.stderr
    meta=json.loads(video.with_suffix('.json').read_text())
    media.append(dict(file=video.name,probe=json.loads(result.stdout),metadata=meta))
report=dict(scope='Recorder only: real 1440x874 frame, real quality denoising/H.264, one synthetic Waving trigger; queue reduced to 8 to force pressure.',
            input_frames=400,input_fps=25,queue_limit=8,queue_peak=peak,
            frame_shape=list(frame.shape),clean_calls=len(clean_times),
            clean_median_ms=statistics.median(clean_times)*1000,
            push_max_ms=max(times)*1000,push_p99_ms=sorted(times)[int(.99*(len(times)-1))]*1000,
            elapsed_seconds=time.perf_counter()-start,stats=rec.stats(),media=media)
(out/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='media'},ensure_ascii=False),flush=True)
assert rec.saved==1 and rec.dropped_frames>0
assert rec.forced_close==rec.abandoned==rec.stats()['write_errors']==rec.stats()['clips_unfinished']==0
assert len(media)==1 and 3.8 <= float(media[0]['probe']['streams'][0]['duration']) <= 4.3
assert max(times)<.2,'producer blocked under writer backpressure'
