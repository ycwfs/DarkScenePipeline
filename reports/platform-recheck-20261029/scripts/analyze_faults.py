"""Require observed loss, real decoded recovery, epoch resets, and drained writers."""
from pathlib import Path
from collections import Counter
import json
from analyze_live import stats,video
OUT=Path(__file__).resolve().parents[1];D=OUT/'faults'
def read(p):return json.loads(p.read_text())
def lines(p):return[json.loads(x)for x in p.read_text().splitlines()if x.strip()]
def main():
 h=read(D/'health-history.json');f=read(D/'faults.json');protocol=read(D/'protocol.json');receivers=read(D/'receivers.json');frames=[v for r in receivers for v in r['frames']];actual=lines(D/'frames.jsonl');session=read(D/'session.json')
 assert read(D/'complete.json')['normal_stop'] and read(D/'cleanup.json')['owned_processes_alive']==0
 assert len(f)==7 and session['status']=='stopped';assert all(r['frames']for r in receivers)
 byaction={r['action']:r['time']for r in f};recoveries={}
 for action in ('source_resume','output_up','source_up'):
  t=byaction[action];healthy=[r for r in h if r['time']>=t and r.get('health',{}).get('status')=='ok'];decoded=[r for r in frames if r['time']>=t]
  assert healthy and decoded;delay=healthy[0]['time']-t;dd=decoded[0]['time']-t
  assert delay<20 and dd<20,(action,delay,dd)
  recoveries[action]=dict(health_ok_seconds=delay,receiver_activity_seconds=dd)
 losses={}
 for down,up in [('source_pause','source_resume'),('output_down','output_up'),('source_down','source_up')]:
  rows=[r for r in h if byaction[down]+2<=r['time']<=byaction[up]]
  losses[down]=dict(statuses=dict(Counter(r.get('health',{}).get('status','request_error')for r in rows)),http_statuses=dict(Counter(r.get('http_status')for r in rows)))
  assert any(r.get('health',{}).get('status')!='ok'for r in rows),(down,'no degraded indication')
 last=h[-1]['health'];assert last['status']=='ok' and last['reconnects']>=2 and last['push_restarts']>=1
 for key in ('clips_failed','clips_forced','clips_abandoned','write_errors'):
  assert last['clips'][key]==0,(key,last['clips'][key])
 clips=[json.loads(p.read_text())for p in(D/'clips').rglob('*.json')if p.with_suffix('.mp4').exists()]
 # The archive stores epoch in both recording path and person-event lineage.
 events=lines(next((D/'clips').rglob('events.jsonl')))
 epochs=sorted(set(r['capture_epoch']for r in actual));assert len(epochs)>=3
 for c in clips:
  assert c['capture_epoch'] in epochs and c['action_key']!='other'
 decoded=[video(p)for p in(D/'clips').rglob('*.mp4')];assert all(v['passed']for v in decoded)
 staleness=[]
 for down,up in [('source_pause','source_resume'),('source_down','source_up')]:
  start,end=byaction[down]+3,byaction[up];inside=[r for r in actual if start<=r['output_monotonic']<=end]
  staleness.append(dict(outage=down,processed_frames_after_three_seconds=len(inside),last_output_after_fault_seconds=max((r['output_monotonic']-byaction[down] for r in actual if byaction[down]<=r['output_monotonic']<=end),default=None)))
 result=dict(passed=True,seconds=protocol['seconds'],recoveries=recoveries,outage_indications=losses,epochs=epochs,reconnects=last['reconnects'],push_restarts=last['push_restarts'],
  received_frames=[len(r['frames'])for r in receivers],expected_receiver_disconnects=[len(r['errors'])for r in receivers],stale_processing=staleness,events=len(events),
  clips_saved=len(clips),clips_context_complete=sum(c['context_complete']for c in clips),clip_frame_drops=last['clips']['frames_dropped'],
  final_clip_counters={k:last['clips'][k]for k in ('clips_failed','clips_forced','clips_abandoned','write_errors')},session_clips=session.get('clips',session.get('clip_stats')),
  clip_decode=decoded,normal_stop=True,owned_processes_alive=0,limitation='Controlled localhost outages, local storage; not real NFS failure or unlimited reliability guarantee. RTMP may repeat last image briefly until input-lost warning appears. Receiver activity after input recovery includes warning/last-image frames; it is not fresh-source end-to-end recovery latency. A few buffered in-flight frames can finish after the source pause.')
 (OUT/'fault-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in('clip_decode','session_clips')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
