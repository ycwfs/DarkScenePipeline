"""Summarize actual receiver timestamps and clip completeness, keeping test failures visible."""
from collections import Counter
import json
from pathlib import Path
import subprocess

import numpy as np

OUT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text())


def quantiles(values):
    if not values:return None
    return dict(n=len(values),p50=float(np.percentile(values,50)),p95=float(np.percentile(values,95)),
                maximum=max(values))


def analyze(folder):
    p=OUT/folder
    session=read(p/'operator/session.json')
    result=read(p/'operator/network/result.json')
    frames=read(p/'operator/network/frames.json')
    events=read(p/'operator/network/events.json')
    protocol=read(p/'protocol.json'); origin=protocol['origin']
    faults=read(p/'faults.json')
    frame_by_id={r['frame']:r for r in frames}
    receivers=[]
    for receiver in read(p/'receivers.json'):
        paired=[(r,frame_by_id[r['frame']])for r in receiver['frames']if r['frame']in frame_by_id]
        latency=[r['time']-origin-f['timestamp']for r,f in paired]
        unique={r['frame']for r,_ in paired}
        recovery=[]
        for fault in faults:
            if fault['action']not in ['output_up','source_up','source_resume','viewer_reconnect']:continue
            candidates=[r['time']-origin-fault['time']for r,f in paired
                if r['time']-origin>=fault['time'] and f['timestamp']>=fault['time']]
            recovery.append(dict(action=fault['action'],seconds=min(candidates)if candidates else None))
        # All-frame latency includes deliberately frozen source/output periods. Report a
        # separate normal-operation interval list; never hide these periods in one average.
        intervals=([(5,12),(24,30),(43,50),(62,65),(70,76)]if folder=='final-stop' else
                   [(10,58),(85,108),(132,153),(180,198),(210,235)])
        normal=[r['time']-origin-f['timestamp']for r,f in paired
                if any(a<=r['time']-origin<=b and a<=f['timestamp']<=b for a,b in intervals)]
        receivers.append(dict(url=receiver['url'],decoded=len(receiver['frames']),
            matched=len(paired),unique_source_frames=len(unique),
            outage_slates=sum(r['frame']==(1<<24)-1 for r in receiver['frames']),
            unpaired=len(receiver['frames'])-len(paired),
            all_received_age=quantiles(latency),normal_received_age=quantiles(normal),
            normal_intervals=intervals,recovery=recovery,client_disconnects=len(receiver['errors'])))
    clips=session['clips']
    summary=dict(folder=folder,service_error=session['error'],harness_complete=(p/'complete.json').exists(),
        captured=result['source_frames_offered'],processed=result['frames'],
        dropped_input=result['dropped_source_frames'],seconds=result['source_duration'],
        processing_fps=result['processing_fps'],submitted=result['submitted'],completed=result['completed'],
        coalesced=result['coalesced'],decision_age=quantiles([e['decision_age']for e in events]),
        max_candidate_people=max(f['people']for f in frames),
        clip_counters={k:v for k,v in clips.items()if k!='clips'},
        clip_closing_reasons=dict(Counter(c['closed_because']for c in clips['clips'])),
        capture=session['capture'],receivers=receivers,faults=faults)
    assert summary['captured']==summary['processed']+summary['dropped_input']
    assert summary['submitted']==summary['completed']+summary['coalesced']
    assert clips['events_received']==len(events), 'Recorder missed a delivered event'
    assert clips['frames_recording_input']+clips.get('dispatcher_frames_dropped',0)==len(frames)
    return summary


def verify_videos():
    checked=[]
    folders=['network-fast/operator/clips','network-off/operator/clips','final-stop/operator/clips','capacity']
    for folder in folders:
        for path in sorted((OUT/folder).rglob('*.mp4')):
            if '.stage'in path.parts:continue
            sidecar=read(path.with_suffix('.json'))
            info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                '-show_entries','stream=codec_name,width,height,nb_read_frames,duration','-of','json',str(path)]))['streams'][0]
            subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(path),'-threads','1','-f','null','-'],
                           check=True,capture_output=True,timeout=45)
            assert info['codec_name']=='h264'
            assert int(info['nb_read_frames'])==sidecar['frames_written']
            difference=abs(float(info['duration'])-sidecar['duration_seconds'])
            assert difference<=.25,(path,difference)
            checked.append(dict(file=str(path.relative_to(OUT)),frames=int(info['nb_read_frames']),
                duration=float(info['duration']),duration_error=difference,
                person=sidecar['person_id'],action=sidecar['action_key'],
                closed_because=sidecar['closed_because'],width=info['width'],height=info['height']))
    (OUT/'evidence/video-verification.json').write_text(json.dumps(checked,ensure_ascii=False,indent=2)+'\n')
    return checked


def main():
    rows=[analyze(folder)for folder in ['network-fast','network-off','final-stop']]
    (OUT/'summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    videos=verify_videos()
    print('Verified',len(videos),'videos;',sum(r['frames']for r in videos),'decoded frames')
    for r in rows:
        print(r['folder'],r['processed'],r['processing_fps'],r['clip_counters']['totals'],
              [(x['decoded'],x['normal_received_age'])for x in r['receivers']])


if __name__=='__main__':main()
