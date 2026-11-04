"""Promote the explicitly accepted wave/running candidate without replacing models."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
OP = ROOT / 'platform/dark-demo-v2'
CANDIDATE = ROOT / 'reports/running-speed-prior-20261027/operator/dark-demo-v2'
FILES = ['main.py', 'suanzi.json', 'darkpipe/config.py', 'darkpipe/person/core.py',
         'darkpipe/person/runtime.py', 'darkpipe/person/action_engine.py',
         'darkpipe/person/wave_prior.py', 'darkpipe/person/running_prior.py']


def main():
    assert not (OUT / 'promotion.json').exists(), 'Already promoted; do not overwrite snapshots'
    old = json.loads((ROOT / 'platform/dark-demo-v2.tar.gz.manifest.json').read_text())
    (OUT / 'previous-package-manifest.json').write_text(json.dumps(old, ensure_ascii=False, indent=2)+'\n')
    candidate = json.loads((CANDIDATE.parents[1] / 'candidate-code-manifest.json').read_text())
    records = {}
    for name in FILES:
        source, target = CANDIDATE / name, OP / name
        sha = hashlib.sha256(source.read_bytes()).hexdigest()
        assert sha == candidate['files'][name]
        before = None
        if target.exists():
            before = hashlib.sha256(target.read_bytes()).hexdigest()
            backup = OUT / 'before' / name
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(target, backup)
        shutil.copyfile(source, target)
        records[name] = dict(before=before, after=sha)
    models = OP / 'models.json'
    shutil.copyfile(models, OUT / 'before/models.json')
    data = json.loads(models.read_text())
    data['version'] = '20261027-full-person-priors'
    data['defaults'].update(wave_pose_min_frames=3, running_pose_min_speed=.35)
    data['wave_prior'] = dict(enabled=True, min_frames=3, keypoint_conf=.5,
        policy='Same wrist above both shoulders on at least three unique source frames within existing one-second window; no continuity requirement; low/unknown raw luma and disabled skeleton bypass; bright insufficient pose rejects.')
    data['running_prior'] = dict(enabled=True, min_speed=.35, units='person_box_heights_per_second',
        policy='Mean displacement speed of common reliable joints over about .2s pairs in the existing one-second window; min5 joints/min3 pairs/span.3s; no coordinate smoothing; low/unknown raw luma, disabled skeleton or insufficient reliable speed bypass.')
    data['deployment_threshold_note'] += ' User explicitly adopted wave3 and running.35 priors on2026-10-27; integrated release verification: reports/full-prior-release-20261027. Original six models unchanged.'
    models.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    (OUT / 'promotion.json').write_text(json.dumps(dict(authorized='User explicitly requested formal integration of both priors', files=records,
        defaults=data['defaults'], model_policy='Original six model hashes unchanged; no training'),ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    main()
