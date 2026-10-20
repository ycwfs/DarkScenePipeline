"""Load isolated operator/box releases and compare real CUDA predictions on matched inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reports/feedback-finetune-20261019'


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--operator',type=Path,required=True)
    ap.add_argument('--box',type=Path,required=True)
    args=ap.parse_args()
    sys.path[:0]=[str(args.operator.resolve()),str(args.box.resolve())]
    from darkpipe.constants import CKPT_FILES,BEHAVIOR_DISPLAY
    from darkpipe.stages.recognize import BehaviorRecognizer
    from darkpipe.stages.enhance_retinexformer import RetinexformerStage
    from darkpipe.stages.downscale import DownscaleStage
    from darkpipe.pose_overlay import ActionPoseOverlay
    from darkpipe.render import append_label_bar
    from darkpipe.media import VideoWriter
    import behavior_infer as box
    torch.set_num_threads(4);cv2.setNumThreads(2)
    assert box.RECO_MIN_CONF==.75
    model=args.operator/'ckpts'/CKPT_FILES['behavior']
    assert hashlib.sha256(model.read_bytes()).hexdigest()==hashlib.sha256(Path(box.DEFAULT_CKPT).read_bytes()).hexdigest()
    rec=BehaviorRecognizer(str(args.operator/'ckpts'),stride=32,reject_tau=.75)
    rec.load('cuda:0');assert rec.skel is not None
    rec.warmup()
    boxed=box.BehaviorRecognizer(device='cuda:0');assert boxed.aux is not None,boxed.aux_reason
    enhancer=RetinexformerStage(str(args.operator/'ckpts'),chunk=1);enhancer.load('cuda:0')
    down=DownscaleStage(720)
    dst=OUT/'delivery-verification';dst.mkdir(exist_ok=True)
    src=ROOT/'real-actiondata-0909'
    choices=[(src/(s+'.mp4'),False) for s in ['喝水16','挥手22','捡东西2','追逐16','追逐19']]
    choices += [(args.box/'sample_data'/s,False) for s in ['sample_drink.mp4','sample_wave.mp4']]
    choices += [(src/'other/other1.mp4',False)]
    manifest=json.loads((OUT/'data-manifest.json').read_text())
    hold=[r for r in manifest['clips'] if r['split']=='holdout']
    choices += [(OUT/'data/videos'/r['source'],True) for r in [hold[0],hold[-1]]]
    rows=[]
    for path,processed in choices:
        cap=cv2.VideoCapture(str(path));assert cap.isOpened(),path
        raw=[]
        while True:
            ok,frame=cap.read()
            if not ok:break
            if processed:frame=frame[:808]
            raw.append(down([frame])[0])
        cap.release();assert raw,path
        idx=np.linspace(0,len(raw)-1,32).round().astype(int)
        original=[raw[int(i)] for i in idx]
        lit=original if processed else enhancer(original)
        rec.reset();rec.lum_gate=0 if processed else 75
        for i,(frame,before) in enumerate(zip(lit,original)):
            ev=rec.push(frame,i,i/32,raw_frame=None if processed else before)
        result=boxed.predict(lit,None if processed else original,lum_gate=0 if processed else 75)
        assert ev.label==BEHAVIOR_DISPLAY[result['label_reco']],(path,ev.label,result['label_reco'])
        assert abs(ev.confidence-result['confidence_reco'])<.003,(path,ev.confidence,result['confidence_reco'])
        overlay=ActionPoseOverlay()
        frame=overlay.draw(lit[-1],lit[-1],ev,ev.timestamp)
        frame=append_label_bar(frame,ev)
        cv2.imwrite(str(dst/(path.stem+'.jpg')),frame)
        rows.append(dict(sample=path.name,processed_replay=processed,label=ev.label,
                         operator_confidence=ev.confidence,box_confidence=result['confidence_reco'],
                         skeleton_active=ev.skeleton_active,drawn_actors=overlay.last_drawn,
                         matched=True))
        print('[verified]',json.dumps(rows[-1],ensure_ascii=False),flush=True)
    (dst/'result.json').write_text(json.dumps(dict(model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        count=len(rows),all_matched=True,rows=rows),ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__': main()
