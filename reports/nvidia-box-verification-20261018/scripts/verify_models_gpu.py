"""Run only on an available/authorized GPU. Uses the extracted delivery package.

Defaults: package samples plus all 18 C-suite windows and selected P regression clips.
Records full precision probabilities; these are regression examples, not held-out metrics.
"""
from pathlib import Path
import argparse
import json
import sys
import time

import cv2
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--box', type=Path, required=True)
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    sys.path.insert(0, str(args.box.resolve()))
    sys.path.insert(0, str(args.repo.resolve() / 'platform/dark-demo-v2'))
    from behavior_infer import BehaviorRecognizer, LABELS, frame_luma
    from lowlight_infer import LowLightEnhancer
    from video_io import downscale_long_side, sample_uniform
    from darkpipe.stages.recognize import BehaviorRecognizer as PlatformRecognizer
    from darkpipe.constants import BEHAVIOR_DISPLAY
    from darkpipe.stages.enhance_retinexformer import RetinexformerStage

    enhancer = LowLightEnhancer(str(args.box / 'weights/NTIRE.pth'), args.device, chunk=1)
    rec = BehaviorRecognizer(str(args.box / 'weights/videomamba_t_behavior_32f_realactall_full_pn.pth'), args.device)
    assert rec.aux is not None, rec.aux_reason
    reference = PlatformRecognizer(str(args.box / 'weights'), stride=32, reject_tau=.65,
                                   skel_fusion_alpha=.7, skel_no_person='soft', lum_gate=75)
    reference.load(args.device)
    assert reference.skel is not None, 'Platform skeleton did not load'
    reference_enhancer = RetinexformerStage(str(args.box / 'weights'), chunk=1)
    reference_enhancer.load(args.device)
    probe = cv2.imread(str(args.box / 'sample_data/dark_frame.png'))
    a, b = enhancer.enhance([probe])[0], reference_enhancer([probe])[0]
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    assert diff.max() == 0, 'Single-frame enhancement differs from platform'
    cv2.imwrite(str(args.output / 'dark_frame_enhanced_gpu.png'), a)
    rows = []
    summary = dict(device=args.device, gpu=torch.cuda.get_device_name(torch.device(args.device)),
                   config={'proc_max_side':720, 'alpha':.7, 'lum_gate':75, 'reject_tau':.65},
                   enhancement={'input_mean':float(probe.mean()), 'output_mean':float(a.mean()),
                                'platform_max_abs_difference':int(diff.max())},
                   skeleton_pose_ms={'box':rec.aux.pose_ms, 'platform':reference.skel.pose_ms},
                   scope='Regression replay including training data; not held-out accuracy or real-time throughput.')

    def run(path, suite, expected, slide=False):
        cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS) or 25.
        raw = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            raw.append(downscale_long_side(frame, 720))
        cap.release()
        assert raw, path
        lit = enhancer.enhance(raw)
        windows = [('whole',0,len(raw))]
        if slide:
            width = max(2, int(round(fps)))
            windows.extend((f't{start/fps:.1f}', start, start+width)
                           for start in range(0, len(raw)-width+1, max(1,int(round(.5*fps)))))
        for name, lo, hi in windows:
            frames = sample_uniform(lit[lo:hi],32)
            original = sample_uniform(raw[lo:hi],32)
            start = time.perf_counter()
            result = rec.predict(frames, original)
            elapsed = time.perf_counter()-start
            reference.reset(); event = None
            for i,(frame,raw_frame) in enumerate(zip(frames,original)):
                value=reference.push(frame, i, i/31, raw_frame=raw_frame)
                if value is not None:
                    event=value
            assert event is not None
            expected_display = BEHAVIOR_DISPLAY.get(result['label_reco'], result['label_reco'])
            agreement = event.label == expected_display and abs(event.confidence-result['confidence_reco']) < .002
            record=dict(suite=suite, source=str(path), window=name, expected=expected,
                        correct=result['label_reco']==expected, result=result,
                        box_seconds_including_first_use=elapsed,
                        platform={'label':event.label,'confidence':event.confidence,'note':reference.prior_note},
                        matches_platform=agreement)
            rows.append(record)
            with (args.output/'windows.jsonl').open('a') as stream:
                stream.write(json.dumps(record,ensure_ascii=False)+'\n')
            print(f'{suite} {path.stem}@{name}: {result["label_reco"]} {result["confidence_reco"]:.4f} expected={expected} match={agreement}',flush=True)
            assert agreement, record

    (args.output/'windows.jsonl').write_text('')
    run(args.box/'sample_data/sample_drink.mp4','sample','drink')
    run(args.box/'sample_data/sample_wave.mp4','sample','wave')
    for stem, label in [('挥手22','wave'),('捡东西2','pick_up'),('追逐16','chase'),('喝水16','drink')]:
        candidates=list((args.repo/'real-actiondata-0909').rglob(stem+'.mp4'))
        assert len(candidates)==1, (stem, candidates)
        run(candidates[0],'C',label,True)
    # Selected historical high-scoring false positives; deliberately not a P-suite rate.
    pdir=args.repo/'online-negative-sample/20261012_083305_78'
    for suffix in ['0029','0001','0010']:
        candidates=sorted(pdir.rglob('*_'+suffix+'.mp4'))
        if candidates:
            run(candidates[0],'P-subset','other')
    for suite in sorted(set(r['suite'] for r in rows)):
        subset=[r for r in rows if r['suite']==suite]
        summary[suite]={'windows':len(subset),'correct':sum(r['correct'] for r in subset),
                        'platform_agreement':sum(r['matches_platform'] for r in subset),
                        'gate_blocked':sum(r['result']['gate_blocked'] for r in subset)}
    summary['all_expected_labels_correct']=all(r['correct'] for r in rows)
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
