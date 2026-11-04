import argparse,json,subprocess
from pathlib import Path
from collections import Counter
import numpy as np
OUT=Path(__file__).resolve().parents[1]
def summary(name):
 p=OUT/name
 complete=json.loads((p/'complete.json').read_text())
 session=json.loads((p/'session.json').read_text())
 rows=[json.loads(l)for l in (p/'frames.jsonl').read_text().splitlines()]
 protocol=json.loads((p/'protocol.json').read_text());ready=protocol['ready']
 faults=json.loads((p/'faults.json').read_text())
 intervals=[]
 for start,end in [('source_pause','source_resume'),('source_down','source_up')]:
  a=next((r['time']for r in faults if r['action']==start),None)
  b=next((r['time']for r in faults if r['action']==end),None)
  if a is not None:intervals.append((a,b+5))
 def stable(t):return t>10 and not any(a-1<=t<=b for a,b in intervals)
 health=json.loads((p/'health-history.json').read_text())
 steady=[r['health']for r in health if 'health'in r and stable(r['time']) and r['health']['status']=='ok']
 def stats(v):
  return dict(n=len(v),min=float(np.min(v)),median=float(np.median(v)),p95=float(np.percentile(v,95)),max=float(np.max(v))) if v else None
 frame_map={r['frame']:r for r in rows}
 receivers=[]
 for receiver in json.loads((p/'receivers.json').read_text()):
  latency=[];seen=set();times=[]
  for r in receiver['frames']:
   frame=frame_map.get(r['frame'])
   if frame is None or not stable(r['time']-ready):continue
   latency.append(r['time']-frame['origin']-frame['timestamp'])
   if r['frame']not in seen:times.append(r['time']);seen.add(r['frame'])
  receivers.append(dict(url=receiver['url'],frames=len(receiver['frames']),unique_frames=len(seen),
    capture_to_receiver_seconds=stats(latency),reconnect_errors=len(receiver['errors'])))
 clips=[]
 for path in p.rglob('*.mp4'):
  subprocess.run(['ffmpeg','-v','error','-xerror','-threads','2','-i',str(path),'-f','null','-'],check=True)
  meta=json.loads(path.with_suffix('.json').read_text())
  clips.append(meta)
 counts=Counter(r['closed_because']for r in clips)
 output=dict(name=name,complete=complete,source=protocol['source'],cards=protocol['cards'],duration=protocol['seconds'],
  steady_definition='health ok, >10s after ready; input fault interval and5s recovery excluded',
  processing_fps=stats([r['fps_proc']for r in steady]),processing_latency_ms=stats([r['latency_ms_last']for r in steady]),
  people=stats([r['people']for r in rows]),decision=session['person'],receivers=receivers,
  clips_total=len(clips),clips_full_decode=True,clips_context_complete=sum(c['context_complete']for c in clips),
  clip_close_reasons=dict(counts),recording_counters={k:v for k,v in session['clips'].items()if k not in ('clips','totals')},
  events=session['event_log_stats'],faults=faults,rss_mb=stats([r['rss_bytes']/1024**2 for r in rows]),
  rss_first_last=[rows[0]['rss_bytes']/1024**2,rows[-1]['rss_bytes']/1024**2])
 (p/'verification.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:output[k]for k in ['name','processing_fps','processing_latency_ms','people','clips_total','clips_context_complete','clip_close_reasons','rss_first_last']},ensure_ascii=False))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('names',nargs='+');args=parser.parse_args()
 for name in args.names:summary(name)
