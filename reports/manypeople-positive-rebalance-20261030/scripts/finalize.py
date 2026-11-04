"""Audit completed experiment artifacts without altering checkpoints or production."""
from collections import Counter
from datetime import datetime, timezone
import csv
import numpy as np
from common import OUT, ROOT, read, save, sha


def main() -> None:
    protocol = read(OUT / 'protocol.json')
    dataset = read(OUT / 'dataset.json')
    previous = ROOT / 'reports/manypeople-finetune-20261029'
    rows = dataset['records']
    original = read(previous / 'dataset.json')['records']
    assert len(rows) == len(original) == 7020
    assert dataset['protocol_sha256'] == sha(OUT / 'protocol.json')
    assert sha(previous / 'dataset.json') == protocol['previous_dataset_sha256']
    assert all({k: v for k, v in row.items() if k != 'train_selected'} == old
               for row, old in zip(rows, original))
    selected = [r for r in rows if r['train_selected']]
    assert len(selected) == 5087
    assert all(r['split'] == 'train' and r['label_kind'] != 'pseudo' for r in selected)
    assert sum(r['scope'] == 'old' for r in selected) == 5009
    assert Counter(r['label'] for r in selected if r['scope'] == 'new') == {
        'talk': 33, 'drink': 4, 'wave': 2, 'other': 39}
    split_groups = {split: {r['group'] for r in rows if r['split'] == split}
                    for split in ['train', 'val', 'test']}
    assert not split_groups['train'] & (split_groups['val'] | split_groups['test'])
    assert not split_groups['val'] & split_groups['test']
    for branch, epochs in [('rgb', 6), ('skeleton', 20)]:
        history = read(OUT / branch / 'history.json')
        assert [h['epoch'] for h in history] == list(range(epochs + 1))
    lock = read(OUT / 'selection-lock.json')
    assert lock['grid_count'] == 147
    assert lock['epochs'] == {'rgb': 0, 'skeleton': 2}
    assert lock['config'] == dict(rgb='selected', skeleton='selected',
                                  alpha=.972, gate=35., action=.925, object=.581)
    for info in lock['selected_models'].values():
        assert sha(info['model']) == info['sha256']
    assert read(OUT / 'pipeline-complete.json')['selection_lock_sha256'] == sha(OUT / 'selection-lock.json')
    with (OUT / 'window-results.csv').open() as stream:
        scored = list(csv.DictReader(stream))
    assert len(scored) == 828 and len({r['id'] for r in scored}) == 828
    assert Counter((r['scope'], r['split']) for r in scored) == {
        ('old', 'val'): 198, ('new', 'val'): 117,
        ('old', 'test'): 214, ('new', 'test'): 299}
    assert all(r['selected'] == r['previous_same_defaults'] for r in scored if r['split'] == 'test')
    test_ids = {r['id'] for r in scored}
    ids = [i for i, r in enumerate(rows) if r['id'] in test_ids]
    for branch in ['rgb', 'skeleton']:
        with np.load(OUT / 'predictions' / f'{branch}-selected.npz') as cache:
            probs = cache['probs']
            relevant = probs[ids] if branch == 'rgb' else probs[:, ids]
            assert np.isfinite(relevant).all()
            assert np.allclose(relevant.sum(-1), 1., atol=.002)
    parity = read(OUT / 'replay-parity.json')
    assert parity['checked'] == 2484 and parity['mismatched_labels'] == 0
    production = {}
    for key, info in protocol['models'].items():
        digest = sha(info['path'])
        assert digest == info['sha256'], key
        production[key] = dict(path=info['path'], sha256=digest)
    archives = {}
    for name in ['platform/dark-demo-v2.tar.gz', 'platform/dark-demo-v2.zip', 'NVIDIA-Box.zip']:
        path = ROOT / name
        expected = path.with_name(path.name + '.sha256').read_text().split()[0]
        actual = sha(path)
        assert actual == expected, name
        archives[name] = actual
    for script in (OUT / 'scripts').glob('*.py'):
        compile(script.read_text(), str(script), 'exec')
    required = ['README.md', 'analysis-report.md', 'stats-appendix.md', 'figure-catalog.md',
                'window-results.csv', 'selection-lock.json', 'adoption-decision.json',
                'replay-parity.json', 'fixed-threshold-comparison.json', 'previous-comparison.json']
    required += [str(p.relative_to(OUT)) for p in sorted((OUT / 'figures').glob('*'))]
    assert len(required) == 14
    inventory = {name: dict(sha256=sha(OUT / name), bytes=(OUT / name).stat().st_size)
                 for name in required}
    save(OUT / 'final-artifacts.json', inventory)
    result = dict(training_and_evaluation_complete=True, data_audit_passed=True,
                  completed_utc=datetime.now(timezone.utc).isoformat(),
                  rgb_epochs=6, skeleton_epochs=20, selected_epochs=lock['epochs'],
                  selection_lock_sha256=sha(OUT / 'selection-lock.json'),
                  scored_windows=828, replay_checked=2484, replay_label_mismatches=0,
                  production_model_hashes=production, production_archive_hashes=archives,
                  production_modified=False, release_decision='retain_production',
                  export_verification_complete=(OUT / 'export-verification.json').exists(),
                  scope='Fixed-window regression on previously inspected sources; no new full-stream validation.')
    save(OUT / 'completion.json', result)
    print('PASS: data splits, 26 trained epochs, 828 scored windows, 2484 parity checks, '
          '6 production model hashes and 3 delivery archive hashes')


if __name__ == '__main__':
    main()
