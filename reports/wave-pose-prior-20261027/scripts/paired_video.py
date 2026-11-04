"""Full-frame offline production inference, paired detectors on exactly the same tasks.

Only a post-fusion wave geometry prior differs. All models and the production
object prior execute once; the alternative reuses the exact same person samples.
This diagnostic harness is not a throughput or infinite-stream benchmark.
"""
import argparse
from collections import Counter, OrderedDict, deque
import copy
import json
from pathlib import Path
import queue
import sys
import time
from common import OUT, ROOT
from wave_prior import apply_wave_prior

import cv2
import numpy as np
import onnxruntime as ort
import torch


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--number',type=int,required=True);p.add_argument('--max-frames',type=int)
    p.add_argument('--prior-demo',action='store_true',help='Exact previous short demo inputs; secondary context regression')
    p.add_argument('--old-id',default='',help='Frozen old onsite positive demo source ID')
    p.add_argument('--wave-short',default='',help='Wave-focused raw excerpt ID')
    p.add_argument('--suffix',default='');p.add_argument('--operator',default='/app')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parents[1]
    dest=out/('short-regression'if a.prior_demo else 'long-videos')/f'new-{a.number:02}{a.suffix}'
    if a.old_id:
        assert not a.prior_demo
        dest=out/'old-videos'/a.old_id
    if a.wave_short:
        assert not a.prior_demo and not a.old_id
        dest=out/'wave-shorts'/a.wave_short
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'summary.json').exists():raise FileExistsError(dest)
    sys.path.insert(0,a.operator)
    from darkpipe.config import PipelineConfig
    from darkpipe.person import object_runtime, object_overlay
    from darkpipe.person.runtime import PersonEngine
    from darkpipe.person.service import Output, make_clips
    from darkpipe.person.core import render_people
    from darkpipe.media import VideoWriter
    torch.set_num_threads(2);cv2.setNumThreads(2)
    records=json.loads((root/'reports/manypeople-inference-20261025/input-manifest.json').read_text())
    if a.wave_short:
        source=next(v for v in json.loads((out/'wave-short-inputs.json').read_text())['sources']if v['id']==a.wave_short)
    elif a.old_id:
        old=next(v for v in json.loads((out/'old-demo-selection.json').read_text())['sources']if v['id']==a.old_id)
        assert not old['processed']
        source=dict(old,absolute_path=str(root/old['source']),duration=old['seconds'],number=0)
    else:
        source=next(r for r in records if r['number']==a.number)
    if a.prior_demo:
        demo=next(r for r in json.loads((root/'reports/object-overlay-20261027/inputs.json').read_text())if r['id']==f'new-{a.number:02}')
        input_path=root/demo['input'];probe=cv2.VideoCapture(str(input_path));assert probe.isOpened()
        source=dict(source,absolute_path=str(input_path),frames=int(probe.get(cv2.CAP_PROP_FRAME_COUNT)),
            duration=demo['seconds'],fps=probe.get(cv2.CAP_PROP_FPS),source_offset=demo['start'],scope='previous short demo, restarted track/window context')
        source['original_video_sha256']=source['sha256']
        source['sha256']=demo['input_sha256']
        probe.release()
    alt_rows={};old_prior=object_runtime.PersonObjectPrior
    key=lambda r:(r['session'],r['track_id'],r['history_generation'],r['frame_index'])

    class PairPrior(old_prior):
        def apply(self,task,row):
            super().apply(task,row)
            alternative=copy.deepcopy(row)
            apply_wave_prior(task,alternative,min_frames=3,lum_gate=35.)
            if row['label']=='wave':
                sweep={}
                for count in [1,3,5,8]:
                    for consecutive in [False,True]:
                        check=copy.deepcopy(row)
                        apply_wave_prior(task,check,count,35.,consecutive=consecutive)
                        sweep[f'{count}_'+('consecutive'if consecutive else 'total')]=check['label']
                alternative['wave_sweep']=sweep
            for field in ('rgb_probs','skeleton_probs','fused_probs','source_frames','action_label','action_confidence','object_prior'):
                assert alternative[field]==row[field],field
            if row['label']!='wave':assert alternative['label']==row['label']
            alt_rows[key(row)]=alternative

    object_runtime.PersonObjectPrior=PairPrior
    cfg=PipelineConfig(person_mode=True,recognize='behavior',gpus='0',device='cuda:0',
        input=source['absolute_path'],ckpt_dir=str(Path(a.operator)/'ckpts'),
        proc_max_side=720,sr='bicubic',sr_scale=2,reco_span_sec=1.,
        clip_dir=str(dest/'baseline-clips'),clip_session='review',
        skel_fusion_alpha=.972,lum_gate=35.,reco_min_conf=.925,object_min_conf=.581,person_object_prior=True)
    cap=cv2.VideoCapture(cfg.input);assert cap.isOpened()
    fps=cap.get(cv2.CAP_PROP_FPS);origin=time.monotonic()
    baseline_clips=make_clips(cfg,origin,offline=True)
    second_cfg=copy.copy(cfg);second_cfg.clip_dir=str(dest/'candidate-clips')
    candidate_clips=make_clips(second_cfg,origin,offline=True)
    event_files={s:(dest/f'{s}-events.jsonl').open('w')for s in ('baseline','candidate')}
    frame_log=(dest/'frames.jsonl').open('w');rows=[];alternatives=[]
    def event(row):
        alternative=alt_rows[key(row)]
        for field in ('completed','decision_age','queue_seconds','batch_size'):
            alternative[field]=row[field]
        rows.append(row);alternatives.append(alternative)
        for variant,value,clips in [('baseline',row,baseline_clips),('candidate',alternative,candidate_clips)]:
            event_files[variant].write(json.dumps(value)+'\n');clips.event(value)

    class PairOutput(Output):
        def __init__(self,engine):
            self.second=object_overlay.ObjectOverlay()
            self.second_writer=VideoWriter(str(dest/'candidate.mp4'),fps=fps)
            super().__init__(engine,None,VideoWriter(str(dest/'baseline.mp4'),fps=fps))

        def _run(self):
            waiting=deque()
            def show(item):
                frame,tracks,record=item
                clean=self.engine.picture(frame)
                evidence=self.engine.overlay_evidence()
                objects=self.objects.update(frame,tracks,record,evidence)
                alternate_tracks=[]
                for track in tracks:
                    t=copy.copy(track)
                    if t.prediction is not None:t.prediction=alt_rows[key(t.prediction)]
                    alternate_tracks.append(t)
                alternate_evidence=[alt_rows[key(r)] for r in evidence]
                other_record=dict(record,tracks=alternate_tracks)
                other_objects=self.second.update(frame,alternate_tracks,other_record,alternate_evidence)
                for variant,ts,obs,writer,clips,rec in [
                    ('baseline',tracks,objects,self.writer,baseline_clips,record),
                    ('candidate',alternate_tracks,other_objects,self.second_writer,candidate_clips,other_record)]:
                    shown=render_people(clean,ts,frame.shape,record['timestamp'],cfg.pose_overlay,obs)
                    writer.write(shown)
                    # Clip dispatch mutates prediction pointers: give each recorder its own copies.
                    clip_record=dict(rec,tracks=[copy.copy(t) for t in ts],object_overlays=obs)
                    clips.frame(clean,clip_record)
                    if obs and not (dest/f'{variant}-preview.jpg').exists():cv2.imwrite(str(dest/f'{variant}-preview.jpg'),shown)
                frame_log.write(json.dumps(dict(frame=record['frame'],timestamp=record['timestamp'],
                    people=record['people'],raw_luma=record['raw_luma'],
                    baseline_objects=objects,candidate_objects=other_objects))+'\n')
                self.count+=1
                if self.count%250==0:
                    (dest/'progress.json').write_text(json.dumps(dict(frames=self.count,source_seconds=self.count/fps,
                        elapsed=time.monotonic()-origin,events=len(rows)))+'\n')
            try:
                while True:
                    item=self.queue.get()
                    if item is None:
                        while waiting:show(waiting.popleft())
                        break
                    waiting.append(item)
                    while waiting and (item[2]['timestamp']-waiting[0][2]['timestamp']>=1.2 or len(waiting)>64):
                        show(waiting.popleft())
            except Exception as exc:self.error=exc
            finally:
                for writer in [self.writer,self.second_writer]:
                    try:writer.close()
                    except Exception as exc:self.error=self.error or exc

    engine=output=None;index=0;started=time.monotonic()
    try:
        engine=PersonEngine(cfg,origin,event,realtime=False)
        output=PairOutput(engine)
        while a.max_frames is None or index<a.max_frames:
            ok,frame=cap.read()
            if not ok:break
            output.put(engine.process(frame,index,index/fps,0,fps));index+=1
        engine.collect(drain=True)
    finally:
        cap.release();error=None
        for resource in [output,baseline_clips,candidate_clips,engine]:
            if resource is not None:
                try:resource.close()
                except Exception as exc:error=error or exc
        for f in [*event_files.values(),frame_log]:f.close()
        if error:raise error
    if a.max_frames is None:assert abs(index-source['frames'])<=1,(index,source['frames'])
    summary=dict(source=source,frames=index,output_frames=output.count,seconds=index/fps,
        elapsed=time.monotonic()-started,paired_checks=len(rows),
        baseline=dict(actions=dict(Counter(r['label']for r in rows)),objects=dict(output.objects.stats),clips=baseline_clips.stats()),
        candidate=dict(actions=dict(Counter(r['label']for r in alternatives)),objects=dict(output.second.stats),clips=candidate_clips.stats()),
        transitions=dict(Counter(f'{r["label"]}->{s["label"]}'for r,s in zip(rows,alternatives))),
        config=dict(alpha=.972,lum_gate=35,action_threshold=.925,object_threshold=.581,wave_min_frames=3,keypoint_conf=.5,wave_count_mode='total_same_wrist'),
        methodology='Complete raw-video production frontend/action execution once; paired production object-prior calls on exact same tasks. '
            'Both use current seven-class detector; candidate adds same-wrist above-both-shoulders in >=3 distinct source frames. '
            'Separate complete visualizations and clip recorders. Offline diagnostic, not real-time throughput test.')
    (dest/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
