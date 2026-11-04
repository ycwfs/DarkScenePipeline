"""Original-image functional run; real models and complete files, no substituted labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

p=argparse.ArgumentParser();p.add_argument('--operator',default='/app');p.add_argument('--repo',default='/repo');p.add_argument('--output',default='/evidence/functional');a=p.parse_args()
sys.path.insert(0,a.operator)
from darkpipe.config import PipelineConfig
from darkpipe.person.service import run_offline

out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
rows=[]
for name in ('喝水27','捡东西16','挥手30','喝水4'):
    target=out/name;target.mkdir(exist_ok=True)
    cfg=PipelineConfig(person_mode=True,recognize='behavior',gpus='0',device='cuda:0',
        input=str(Path(a.repo)/'real-actiondata-all'/f'{name}.mp4'),
        output=str(target/'output.mp4'),events_json=str(target/'events.json'),
        ckpt_dir=str(Path(a.operator)/'ckpts'),proc_max_side=720,
        sr='bicubic',sr_scale=2,reco_span_sec=1.,
        clip_dir=str(target/'clips'),clip_session='review')
    events=run_offline(cfg)
    result=dict(source=name,stats=cfg.stats,labels=dict(Counter(e['label'] for e in events)),
        actions_before_prior=dict(Counter(e.get('action_label',e['label'])for e in events)),
        prior_reasons=dict(Counter(e['object_prior']['reason'] for e in events)))
    rows.append(result)
    (out/'summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    assert all(e['threshold']==.925 for e in events)
    print(json.dumps(result,ensure_ascii=False),flush=True)
assert sum(r['prior_reasons'].get('single_frame_support',0)for r in rows)>0
assert sum(r['prior_reasons'].get('below_luma_gate',0)for r in rows)>0
print('FUNCTIONAL PASSED',flush=True)
