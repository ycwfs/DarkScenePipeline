"""Audit recorded events and delivered media without re-enhancing already-rendered clips."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / 'platform/dark-demo-v2'))
from darkpipe.constants import BEHAVIOR_DISPLAY, BEHAVIORS
from darkpipe.render import append_label_bar
from darkpipe.stages.base import RecognitionEvent

cv2.setNumThreads(2)
THRESHOLDS = [.65, .70, .75, .80, .85, .89, .90]
sessions, all_events, clips = [], [], []
for path in sorted((ROOT / 'input').glob('*/session.json')):
    session = json.loads(path.read_text())
    events = [json.loads(s) for s in (path.parent / 'events.jsonl').read_text().splitlines() if s.strip()]
    actions = [r for r in events if r['label'] != 'Other']
    all_events.extend(dict(r, session_id=path.parent.name) for r in events)
    sessions.append(dict(session_id=path.parent.name, status=session['status'],
                         started_at=session['started_at'], stopped_at=session['stopped_at'],
                         events=len(events), action_events=len(actions),
                         action_labels=dict(Counter(r['label'] for r in actions)),
                         max_action_confidence=max((r['confidence'] for r in actions), default=0),
                         clips=session['clips']))
    for metadata in sorted(path.parent.glob('*/*.json')):
        rec=json.loads(metadata.read_text()); video=metadata.with_suffix('.mp4')
        cap=cv2.VideoCapture(str(video)); assert cap.isOpened(), video
        fps=cap.get(cv2.CAP_PROP_FPS); width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Current 720 / bicubic x2 output is 1440x808 + a 66-pixel label strip.
        # Classify the label name only (exclude confidence digits), using the shipped font.
        body_h=next(h for h in range(height-200,height) if h+max(48,round(.08*h))+(max(48,round(.08*h))&1)==height)
        templates={}
        for label in BEHAVIORS:
            display=BEHAVIOR_DISPLAY.get(label,label)
            image=append_label_bar(np.zeros((body_h,width,3),np.uint8), RecognitionEvent(0,0,display,.9))
            # First two characters uniquely identify these ten Chinese display labels.
            left=max(8,int(.02*width)); bar_h=height-body_h; crop_w=round(bar_h*.55)*2
            templates[display]=image[body_h:,left:left+crop_w].astype(np.float32)
        labels=[]; errors=[]; count=0; first=None; last=None
        while True:
            ok, frame=cap.read()
            if not ok:break
            if first is None:first=frame.copy()
            last=frame
            crop=frame[body_h:,left:left+crop_w].astype(np.float32)
            scores=sorted((float(np.mean((crop-t)**2)), lab) for lab,t in templates.items())
            labels.append(scores[0][1]);errors.append(scores[0][0]); count+=1
        cap.release()
        assert count>0, video
        item=dict(path=str(video.relative_to(ROOT)), filename=video.name,
                  label=rec['label'], confidence=rec['confidence'],
                  width=width,height=height,frames=count,fps=fps,video_seconds=count/fps,
                  recorded_duration=rec['duration_seconds'],closed_because=rec['closed_because'],
                  first_event=rec['first_event'], label_frames=dict(Counter(labels)),
                  label_first=labels[0],label_last=labels[-1],
                  template_mse_max=max(errors),
                  template_method='Nearest shipped-font RGB template of first two label characters; diagnostic, not human ground truth.')
        clips.append(item)
        if video.name=='20261015_181324_drinking_water_0005.mp4':
            (ROOT/'example-label-frames.json').write_text(json.dumps(dict(item,labels=labels),ensure_ascii=False,indent=2))
            cv2.imwrite(str(ROOT/'example-first.png'),first);cv2.imwrite(str(ROOT/'example-last.png'),last)
            # Preserve all 50 labels in a diagnostic contact sheet for manual inspection.
            capture=cv2.VideoCapture(str(video)); strips=[]
            while True:
                ok,frame=capture.read()
                if not ok:break
                strip=frame[body_h:,:260].copy()
                strips.append(strip)
            capture.release()
            rows=[np.hstack(strips[i:i+5]) for i in range(0,len(strips),5)]
            cv2.imwrite(str(ROOT/'example-all-labels.png'),np.vstack(rows))

actions=[r for r in all_events if r['label']!='Other']
sweep=[]
for tau in THRESHOLDS:
    keep=[r for r in actions if r['confidence']>=tau]
    sweep.append(dict(threshold=tau, false_events_assuming_all_other=len(keep),
                      removed=len(actions)-len(keep),
                      reduction_fraction=(len(actions)-len(keep))/len(actions),
                      by_label=dict(Counter(r['label'] for r in keep)),
                      by_session=dict(Counter(r['session_id'] for r in keep)),
                      rounding_boundary_events=sum(abs(r['confidence']-tau)<=.00005 for r in actions)))
report=dict(annotation='All supplied sessions are provisionally treated as other, per user; no new training performed.',
            source_sha256=hashlib.sha256((REPO/'realdata-10-09/darkpipe.tar.gz').read_bytes()).hexdigest(),
            sessions=sessions, events=len(all_events), named_events=len(actions), videos=len(clips),
            threshold_parameter='reco_min_conf (act_tau is only for max_prior)',
            score_precision='Production JSON rounds confidence to 4 decimals; these are logged-event counts.',
            sweep=sweep, clips=clips,
            top_false_events=[{k:r[k] for k in ('session_id','wall_time','label','confidence')} for r in sorted(actions,key=lambda r:r['confidence'],reverse=True)[:10]])
(ROOT/'feedback-analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('events',len(all_events),'action events',len(actions),'videos',len(clips))
print('sweep',[(r['threshold'],r['false_events_assuming_all_other']) for r in sweep])
print('example',[(r['filename'],r['label_frames']) for r in clips if r['filename']=='20261015_181324_drinking_water_0005.mp4'])
print('all-other clips',sum(set(r['label_frames'])=={'Other'} for r in clips))
