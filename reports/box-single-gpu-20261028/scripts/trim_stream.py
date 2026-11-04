"""Remove only transport black pre/tail from actual received output, never re-infer."""
import json,re,subprocess
from pathlib import Path
OUT=Path(__file__).resolve().parents[1];live=OUT/'live-final';source=live/'full-output-stream.mp4'
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','json',str(source)]));duration=float(probe['format']['duration'])
cmd=['ffmpeg','-nostdin','-v','info','-threads','2','-i',str(source),'-vf','blackdetect=d=1:pic_th=0.995:pix_th=0.03','-an','-f','null','-']
r=subprocess.run(cmd,capture_output=True,text=True,check=True);(live/'black-boundaries.log').write_text(r.stderr)
black=[tuple(map(float,m))for m in re.findall(r'black_start:([0-9.]+) black_end:([0-9.]+) black_duration:([0-9.]+)',r.stderr)]
start=next((end for beg,end,length in black if beg<=.1),0.)
end=next((beg for beg,stop,length in black if stop>=duration-.2 and beg>start+290),duration)
assert 295<=end-start<=305,(start,end,duration,black)
out=OUT/'live-final/full-output-5min.mp4';subprocess.run(['ffmpeg','-nostdin','-v','error','-ss',str(start),'-i',str(source),'-t',str(end-start),'-an','-c:v','libx264','-preset','veryfast','-crf','18','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart','-y',str(out)],check=True)
subprocess.run(['ffmpeg','-nostdin','-v','error','-xerror','-threads','2','-i',str(out),'-f','null','-'],check=True)
(OUT/'stream-trim.json').write_text(json.dumps(dict(source=str(source.relative_to(OUT)),output=str(out.relative_to(OUT)),original_seconds=duration,trim_start=start,trim_end=end,seconds=end-start,black_intervals=black,method='Only trim contiguous black transport preroll/tail from externally recorded RTMP output; preserve all middle frames/gaps; H264 re-encode, no model re-inference or test markers'),indent=2)+'\n')
print('TRIM',start,end,end-start)
