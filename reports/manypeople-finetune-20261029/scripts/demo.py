"""Actual shared production offline service with explicit baseline/candidate parameters."""
import argparse
import sys
from common import *
sys.path.insert(0,str(OP))
from darkpipe.config import PipelineConfig
from darkpipe.person.service import run_offline


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',default='0/1');a=p.parse_args()
    i,n=map(int,a.shard.split('/'));candidate=read(OUT/'candidate-config.json')
    for job in read(OUT/'demo-jobs.json')[i::n]:
        for which in ('baseline','selected'):
            dest=OUT/'demos'/job['id']/which;dest.mkdir(parents=True,exist_ok=True)
            if (dest/'summary.json').exists():continue
            config=dict(skel_fusion_alpha=.972,lum_gate=35.,reco_min_conf=.925,object_min_conf=.581,
                wave_pose_min_frames=3,running_pose_min_speed=.35,person_object_prior=True,proc_max_side=720)
            if which=='selected':
                config.update({k:candidate['parameters'][k]for k in config})
            cfg=PipelineConfig(person_mode=True,recognize='behavior',gpus='0',device='cuda:0',input=job['source'],
                input_processed=job['processed'],output=str(dest/'output.mp4'),events_json=str(dest/'events.json'),
                ckpt_dir=str(OP/'ckpts')if which=='baseline'else candidate['checkpoint_dir'],
                sr='off',reco_span_sec=1.,clip_dir=str(dest/'clips'),clip_session='review',**config)
            events=run_offline(cfg)
            save(dest/'summary.json',dict(job=job,configuration=config,events=len(events),stats=cfg.stats,
                labels={c:sum(e['label']==c for e in events)for c in LABELS},finite_file_demo=True,
                whole_frame_ground_truth=False))
            print('DEMO',job['id'],which,len(events),flush=True)


if __name__=='__main__':main()
