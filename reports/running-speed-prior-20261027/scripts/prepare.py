import ast
import copy
import json
from common import OUT, ROOT, WAVE, save, digest


def main():
    assert not (OUT/'protocol.json').exists(), 'Frozen protocol exists; reproduce in a fresh experiment directory'
    save(OUT/'protocol.json', dict(task='Running speed prior; no retraining',
        baseline='Accepted wave-prior candidate + official seven-class object model',
        defaults=dict(alpha=.972,lum_gate=35,action_threshold=.925,object_threshold=.581,wave_min_frames=3,side=720),
        formula='Mean over frame pairs of mean Euclidean displacement of >=5 common reliable joints / mean person bbox height / timestamp delta; nearest lag .2s within [.16,.28]; no gap >.12s; >=3 pairs covering >=.3s. Same person/generation, source frames deduplicated. Conf>=.5, joints within frame and 20%-expanded person bbox. No pose smoothing.',
        policy='Gate35; low/unknown raw luma or disabled skeleton bypass; bright insufficient geometry rejects running; zero threshold disables.',
        grid=[round(i*.05,2)for i in range(51)],
        selection='Select on historical val + user-confirmed 01/07/09 running candidates only. Preserve >=95% of baseline true val running windows and all reviewed child-running candidates; maximize removed known walking false triggers, tie choose smallest positive threshold. If impossible retain disabled and explain. Train diagnostic; historical test opened after selection, report only.',
        uncertainty='Identify child in raw context, do not derive truth from the speed rule. Other actors running triggers treated walking per user; uncertain identity excluded. Repeated/overlapping windows not independent trials.',
        speed_code_sha256=digest(OUT/'implementation/running_prior.py')))
    # Freeze text semantic correction before any XCLIP scores.
    target=ROOT/'reports/xclip-person-20261027';target.mkdir(exist_ok=True)
    constants=ast.parse((ROOT/'darkpipe/constants.py').read_text())
    prompts=next(ast.literal_eval(n.value)for n in constants.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='BEHAVIOR_PROMPTS'for t in n.targets))
    prompts['chase']=['a person running','a person jogging','a person sprinting','a child or adult running quickly']
    prompts['other'].remove('a person running alone')
    save(target/'prompts.json',prompts)
    save(target/'protocol.json',dict(model='microsoft/xclip-base-patch16-zero-shot',
        weights_sha256=digest(ROOT/'ckpts/xclip-base-patch16-zero-shot/pytorch_model.bin'),
        input='Retinexformer enhanced720 -> person detector -> box/appearance tracker (no pose) -> 15% person ROI -> aspect-preserving224square ->32frames/1s -> XCLIP',
        no_skeleton=True,no_object_prior=True,no_training=True,
        prompts_sha256=digest(target/'prompts.json'),
        thresholds=[0,.4,.75,.85,.925],primary_threshold=.4,
        calibration='Historical val only: choose threshold on .00:.01:.99 grid maximizing macro F1 over represented classes, tie higher other recall then higher threshold; held-out historical test scored after lock. No prompt tuning after score inspection.',
        evaluation='Historical all val/test sources; matched frozen person/time windows for classifier comparison plus full no-pose end-to-end 01/07/09 and selected short demos. Paired fixed boxes may originate from prior pose-assisted tracking; explicitly distinguished from no-pose end-to-end.',
        limits='Public zero-shot model is not onsite fine-tuned; per-person crops lose some interaction context. Different probability scales, do not compare .925 as equal calibration. Same historical data is not a new independent external test.'))
    print('Protocols and XCLIP prompts frozen')

if __name__=='__main__':main()
