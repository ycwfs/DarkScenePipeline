"""Audit final-package stream results, final labels, clip metadata, and all video decodes."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

import numpy as np

OUT=Path(__file__).resolve().parents[1]

def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def stats(x):
    a=np.asarray(x,float)
    return dict(count=len(a),minimum=float(a.min()),median=float(np.median(a)),p95=float(np.percentile(a,95)),maximum=float(a.max())) if len(a) else None

def video(path):
    probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
        'stream=width,height,codec_name,avg_frame_rate,nb_frames:format=duration','-of','json',str(path)],capture_output=True,text=True,check=True)
    run=subprocess.run(['ffmpeg','-nostdin','-v','error','-xerror','-threads','2','-i',str(path),'-f','null','-'],capture_output=True,text=True)
    return dict(path=str(path.relative_to(OUT)),passed=run.returncode==0,metadata=json.loads(probe.stdout),stderr=run.stderr)

def main():
    live=OUT/'live';complete=read(live/'complete.json');session=read(live/'session.json')
    health=read(live/'health-history.json');timeline=read(live/'source-timeline.json')
    start,end=timeline['source_start_monotonic'],timeline['source_end_monotonic']
    main_health=[r for r in health if start<=r['monotonic']<=end]
    frames=lines(live/'frames.jsonl')
    main_frames=[r for r in frames if start<=r['origin']+r['timestamp']<=end]
    events=lines(next((live/'clips').rglob('events.jsonl')))
    clips=[read(p) for p in (live/'clips').rglob('*.json') if p.with_suffix('.mp4').exists()]
    reasons=Counter(e['object_prior']['reason'] for e in events)
    vetoes=[e for e in events if e['object_prior']['reason']=='no_compatible_object']
    assert all(e['label']=='other' and e['action_label'] in ('drink','pick_up') for e in vetoes)
    assert all(e['label']==e['object_prior']['label'] for e in events)
    assert all(e['threshold']==.925 for e in events)
    assert all(e['object_prior']['object_min_conf']==.581 and e['object_prior']['lum_gate']==35 for e in events)
    assert all(c['action_key']!='other' for c in clips)
    for e in events:
        p=e['object_prior']
        if p['applied']:
            assert e['raw_luma']>=35
            assert any(s['available'] and s['binding_valid'] and s['raw_luma']>=35 for s in p['samples'])
        if p['reason']=='single_frame_support':assert p['evidence']
    assert complete['normal_stop'] and session['status']=='stopped'
    config=read(live/'effective-config.json')
    for key,value in dict(skel_fusion_alpha=.972,lum_gate=35.,reco_min_conf=.925,object_min_conf=.581,person_object_prior=True).items():
        assert config[key]==value,(key,config[key])
    end_health=health[-1]['health']
    for key in ('clips_failed','clips_forced','clips_abandoned','frames_dropped','write_errors'):
        assert end_health['clips'][key]==0,(key,end_health['clips'][key])
    snapshots=[r['engine'] for r in main_frames if r.get('engine')]
    byte_counts=[e['object_history_bytes'] for e in snapshots]
    assert all(b<=128*1024*1024 for b in byte_counts)
    checks=list(OUT.glob('functional/*/output.mp4'))+list((OUT/'functional').rglob('clips/**/*.mp4'))
    checks+=list((live/'clips').rglob('*.mp4'))+list(live.glob('viewer-*.mkv'))+[OUT/'full-output-stream.mp4']
    with ThreadPoolExecutor(max_workers=2) as pool:decoded=list(pool.map(video,checks))
    save(OUT/'video-verification.json',decoded)
    assert all(v['passed'] for v in decoded)
    gaps=np.diff([r['output_monotonic'] for r in main_frames])
    result=dict(passed=True,complete=complete,session_status=session['status'],
        main_health_samples=len(main_health),health_statuses=dict(Counter(r.get('health',{}).get('status','request_error')for r in main_health)),
        fps_proc=stats([r['health']['fps_proc'] for r in main_health if 'health' in r]),
        capture_to_output_ms=stats([1000*(r['output_monotonic']-r['origin']-r['timestamp'])for r in main_frames]),
        output_intervals_seconds=stats(gaps),
        main_frames_processed=len(main_frames),all_event_count=len(events),
        final_labels=dict(Counter(e['label']for e in events)),prior_reasons=dict(reasons),
        vetoed_actions=dict(Counter(e['action_label']for e in vetoes)),
        action_result_age_seconds=stats([e['decision_age']for e in events]),
        object_history_bytes=stats(byte_counts),
        clips_saved=len(clips),clips_context_complete=sum(c.get('context_complete',False)for c in clips),
        clip_labels=dict(Counter(c['action_key']for c in clips)),
        clip_totals=session.get('clips',session.get('clip_stats')),
        final_health_clip_counters={k:end_health['clips'][k]for k in ('clips_saved','clips_failed','clips_forced','clips_abandoned','frames_dropped','write_errors')},
        reconnects=end_health['reconnects'],push_restarts=end_health['push_restarts'],hls_restarts=end_health['hls_restarts'],
        parent_rss_bytes=stats([r['rss_bytes']for r in main_frames]),
        sse_events=read(live/'sse-status.json')['events'],
        decoded_videos=len(decoded),stream_pixels_unmodified=True,
        source_seconds=read(OUT/'input-manifest.json')['seconds'],
        limitation='Local loopback, RTX3090 three cards, finite five-minute stream; not onsite vGPU/NFS/HDFS or infinite-duration guarantee; no independent end-to-end receiver latency probe this run.')
    save(OUT/'verification.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
