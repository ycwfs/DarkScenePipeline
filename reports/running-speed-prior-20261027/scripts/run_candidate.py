"""Actual unwrapped candidate service: offline output, per-person clips and events."""
import argparse
import json
import sys
from pathlib import Path
from common import OUT,ROOT,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--number',type=int);p.add_argument('--source-id');p.add_argument('--excerpt');a=p.parse_args()
    sys.path.insert(0,str(OUT/'operator/dark-demo-v2'))
    from darkpipe.config import PipelineConfig
    from darkpipe.person.service import run_offline
    if a.excerpt:
        source=next(v for v in json.loads((OUT/'short-inputs.json').read_text())if v['id']==a.excerpt)
        dest=OUT/'short-videos'/a.excerpt;path=source['absolute_path']
    elif a.number:
        source=next(v for v in json.loads((ROOT/'reports/manypeople-inference-20261025/input-manifest.json').read_text())if v['number']==a.number)
        dest=OUT/'long-videos'/f'new-{a.number:02}';path=source['absolute_path']
    else:
        source=next(v for v in json.loads((ROOT/'reports/wave-pose-prior-20261027/protocol.json').read_text())['old_sources']if v['id']==a.source_id)
        path=str(ROOT/source['source']);dest=OUT/'short-videos'/a.source_id
    dest.mkdir(parents=True,exist_ok=True);assert not (dest/'summary.json').exists()
    cfg=PipelineConfig(person_mode=True,recognize='behavior',gpus='0',device='cuda:0',input=path,
        input_processed=source.get('processed',False),output=str(dest/'output.mp4'),events_json=str(dest/'events.json'),
        ckpt_dir=str(OUT/'operator/dark-demo-v2/ckpts'),proc_max_side=720,sr='bicubic',sr_scale=2,
        reco_span_sec=1.,clip_dir=str(dest/'clips'),clip_session='review')
    rows=run_offline(cfg)
    save(dest/'summary.json',dict(source=source,events=len(rows),stats=cfg.stats,
        threshold=cfg.running_pose_min_speed,missing_policy='bypass',
        running_before=sum(r.get('action_label')=='chase'for r in rows),running_after=sum(r['label']=='chase'for r in rows),
        speed_vetoes=sum(r.get('running_prior',{}).get('applied',False)and not r['running_prior'].get('supported',True)for r in rows)))

if __name__=='__main__':main()
