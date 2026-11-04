"""Run the isolated candidate's unwrapped offline entry and compare paired evidence."""
import argparse
import json
import sys
from common import OUT, ROOT, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wave-short')
    parser.add_argument('--verify-existing', action='store_true', help='Audit saved inference without running models again')
    args = parser.parse_args()
    sys.path.insert(0,str(OUT/'operator/dark-demo-v2'))
    from darkpipe.config import PipelineConfig
    from darkpipe.person.service import run_offline
    if args.wave_short:
        source=next(v for v in json.loads((OUT/'wave-short-inputs.json').read_text())['sources']if v['id']==args.wave_short)
        input_path=source['absolute_path']
        reference_dir=OUT/'wave-shorts'/source['id']
        dest=OUT/'integrated-bright-candidate'
    else:
        source=next(v for v in json.loads((OUT/'old-demo-selection.json').read_text())['sources']if v['source'].endswith('/挥手3.mp4'))
        input_path=str(ROOT/source['source'])
        reference_dir=OUT/'old-videos'/source['id']
        dest=OUT/'integrated-candidate'
    dest.mkdir(exist_ok=True)
    cfg=PipelineConfig(person_mode=True,recognize='behavior',gpus='0',device='cuda:0',
        input=input_path,output=str(dest/'output.mp4'),events_json=str(dest/'events.json'),
        ckpt_dir=str(OUT/'operator/dark-demo-v2/ckpts'),proc_max_side=720,sr='bicubic',sr_scale=2,
        reco_span_sec=1.,clip_dir=str(dest/'clips'),clip_session='review',wave_pose_min_frames=3)
    rows=json.loads((dest/'events.json').read_text()) if args.verify_existing else run_offline(cfg)
    reference=[json.loads(line)for line in (reference_dir/'candidate-events.jsonl').read_text().splitlines()]
    assert len(rows)==len(reference)
    import numpy as np
    max_difference = {'rgb_probs': 0., 'skeleton_probs': 0., 'fused_probs': 0.}
    for actual,expected in zip(rows,reference):
        for key in ['track_id','history_generation','frame_index','source_frames','bbox','label','action_label','wave_prior','object_prior']:
            assert actual[key]==expected[key],key
        for key in max_difference:
            if actual[key] is not None:
                delta = float(np.max(np.abs(np.asarray(actual[key])-np.asarray(expected[key]))))
                assert np.isfinite(delta)
                max_difference[key] = max(max_difference[key], delta)
    vetoes=sum(row['wave_prior'].get('applied',False) and row['label']=='other' for row in rows)
    if args.wave_short:
        assert vetoes > 0, 'Bright integrated probe must exercise an actual wave veto'
    save(OUT/('integrated-bright-candidate-check.json' if args.wave_short else 'integrated-candidate-check.json'),dict(passed=True,source=input_path,events=len(rows),wave_vetoes=vetoes,stats=cfg.stats,
        same_person_frames_labels_and_prior_evidence=True,
        max_probability_difference=max_difference,
        probability_reproducibility_within_2e_4=all(v<=2e-4 for v in max_difference.values()),
        caveat='Pass means exact person/frame/label/prior integration consistency across independent model loads. Probability reproducibility is reported separately, not silently relaxed. Paired experiments share identical probabilities. One short source, not a real-time test.'))
    print('Integrated candidate matches paired experiment on',len(rows),'events')


if __name__=='__main__':main()
