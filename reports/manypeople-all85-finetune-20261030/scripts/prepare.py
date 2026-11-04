"""Enroll all reviewed positives while preserving old held-out source groups."""
from collections import Counter
import copy
import importlib.util
import numpy as np
from common import *


def main() -> None:
    assert not (OUT / 'protocol.json').exists(), 'Preserve frozen protocols'
    previous = ROOT / 'reports/manypeople-positive-rebalance-20261030'
    original = ROOT / 'reports/manypeople-finetune-20261029'
    spec = importlib.util.spec_from_file_location('negative_selection', previous / 'scripts/prepare.py')
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    old = read(previous / 'dataset.json')
    rows = copy.deepcopy(old['records'])
    positives = [r for r in rows if r['scope'] == 'new' and r['label_kind'] == 'human'
                 and r['label'] != 'other']
    negatives = [r for r in rows if r['scope'] == 'new' and r['split'] == 'train'
                 and r['label_kind'] == 'human' and r['label'] == 'other']
    assert len(positives) == 85
    chosen = helpers.negative_subset(negatives, 85)
    chosen_ids = {r['id'] for r in chosen}
    for row in rows:
        row['train_selected'] = (row['scope'] == 'old' and row['split'] == 'train'
            or row['scope'] == 'new' and row['label_kind'] == 'human'
            and (row['label'] != 'other' or row['id'] in chosen_ids))
        row['evaluation_role'] = score_role(row)
    selected = [r for r in rows if r['train_selected']]
    assert len(selected) == 5179
    assert all(r['label_kind'] != 'pseudo' for r in selected)
    old_train = {r['group'] for r in selected if r['scope'] == 'old'}
    old_held = {r['group'] for r in rows if r['scope'] == 'old' and r['split'] != 'train'}
    assert not old_train & old_held
    assert all(r['train_selected'] for r in rows if r['scope'] == 'new'
               and r['label_kind'] == 'human' and r['label'] != 'other')
    protocol = read(previous / 'protocol.json')
    protocol.update(user_request='All 85 reviewed crowded non-other windows + all 5009 old train windows + 85 reviewed other; preserve old val/test by explicit user reply.',
        previous_experiment=str(previous), previous_dataset_sha256=sha(previous / 'dataset.json'),
        historical_test_limit='All crowded positives include the former val/test and are now training data. No crowded positive holdout remains. Unselected crowded other shares capture groups/people with training. Old source splits are preserved but historically inspected, so old test is regression only.',
        new_splits='Original split field is historical provenance only; train_selected overrides it for all new positives.',
        weak_policy='No pseudo-label training or scoring; no XCLIP.', production_modified=False)
    protocol['training'].update(sampling='Class/source weighted, old/new mass .7/.3 when both exist; new other mass .1. All 85 positives enrolled; 85 human other from original new train only.',
        checkpoint_selection='Old canonical val only. Branch diagnostics at .75; joint fixed full-scheme parameters. No crowded label is used as held-out selection data.',
        initialized_from='Original production weights, not prior finetune checkpoints')
    protocol['tuning'].update(alpha=[.972], gate=[35.], action=[.925], object=[.581],
        objective='Old canonical val only: every supported positive-class recall >= baseline-.05; minimize old-other FPR, then supported macro F1 and positive TP; deterministic earlier-epoch tie-break. Also retain best trained-only pair and fixed last pair as predeclared diagnostics.')
    protocol['summary'] = dict(selected_train=helpers.sample_audit(selected, .1),
        all_positive_counts=dict(Counter(r['label'] for r in positives)),
        new_other_counts=dict(Counter(r['source_id'] for r in chosen)),
        positive_review_events=len({x for r in positives for x in r['review_ids']}))
    protocol['evaluation'] = dict(primary='Old source-held-out regression',
        crowded_positives='Training fit for all 85; not independent accuracy/generalization',
        crowded_other='Separate 85 trained and remaining 566 untrained windows; same-source diagnostics, not independent negatives',
        thresholds_fixed=True, last_epoch_predeclared=True, best_trained_pair_old_validation_only=True)
    save(OUT / 'protocol.json', protocol)
    save(OUT / 'dataset.json', dict(old, records=rows, protocol_sha256=sha(OUT / 'protocol.json'),
                                  train_selection_required=True))
    save(OUT / 'selected-other.json', chosen)
    save(OUT / 'sampling-audit.json', protocol['summary'])
    save(OUT / 'protocol-amendment-before-training.json', dict(
        reason='User explicitly moves all 85 human positives into training; old training-only scope confirmed.',
        training_started=False, previous_test_seen=True, crowded_positive_holdout_exists=False))
    save(OUT / 'joint-selection-protocol.json', dict(rgb_epochs=list(range(7)), skeleton_epochs=list(range(21)),
        grid_count=147, fixed_parameters=dict(alpha=.972, gate=35., action=.925, object=.581),
        selection_sources='old canonical validation only', tie_break='earliest RGB then earliest skeleton'))
    (OUT / 'skeleton-inputs.npz').symlink_to((previous / 'skeleton-inputs.npz').resolve())
    (OUT / 'predictions').mkdir()
    for branch in ['rgb', 'skeleton']:
        path = original / 'predictions' / f'{branch}-baseline.npz'
        with np.load(path) as pred:
            assert pred['ids'].tolist() == [r['id'] for r in rows]
            probs = pred['probs']; ids = scored_indices(rows)
            assert np.isfinite(probs[ids] if branch == 'rgb' else probs[:, ids]).all()
        dest = OUT / 'predictions' / path.name
        dest.symlink_to(path.resolve())
        save(dest.with_suffix('.json'), dict(read(path.with_suffix('.json')),
            dataset_sha256=sha(OUT / 'dataset.json'), reused_from=str(path), exact_record_input_order_preserved=True))
    # Only original old val/test and new 01/02/03/04/06/08 evidence exists.
    # Complete new 05/07/09 evidence separately without writing through a symlink.
    (OUT / 'object-evidence').mkdir()
    for path in (original / 'object-evidence').glob('*.json'):
        (OUT / 'object-evidence' / path.name).symlink_to(path.resolve())
    with np.load(OUT / 'skeleton-inputs.npz') as cache:
        valid = cache['valid'][GATES.index(35.)]
    audit = dict(records=len(rows), old_train=5009, old_val=233, old_test=262,
        new_positive=85, new_other=85, rgb_train=len(selected),
        skeleton_train=sum(r['train_selected'] and bool(valid[i]) for i,r in enumerate(rows)),
        skeleton_new_positive=sum(r['scope']=='new' and r['label']!='other'
            and r['train_selected'] and bool(valid[i]) for i,r in enumerate(rows)),
        old_group_isolation=True, new_all_capture_groups_seen_by_training=True,
        pseudo_used=False, scored_windows=len(scored_indices(rows)))
    save(OUT / 'data-verification.json', audit)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
