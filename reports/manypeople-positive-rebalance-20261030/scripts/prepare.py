"""Keep source splits; select reviewed negatives without consulting held-out scores."""
from collections import Counter, defaultdict
import copy
from common import *

PREVIOUS = ROOT / 'reports/manypeople-finetune-20261029'


def negative_subset(rows: list[dict], count: int) -> list[dict]:
    # Round-robin source and historical confusion type. Spread identities and
    # review episodes before taking another window from the same person.
    groups = defaultdict(list)
    for row in rows:
        original = LABELS[max(range(10), key=lambda i: row['legacy_rgb'][i])]
        groups[row['source_id'], original].append(row)
    selected = []
    used_tracks, used_reviews, used_sources = Counter(), Counter(), Counter()
    while len(selected) < count:
        available = [key for key, values in groups.items() if values]
        if not available:
            break
        key = min(available, key=lambda k: (used_sources[k[0]],
            sum(r['source_id'] == k[0] and LABELS[max(range(10), key=lambda i: r['legacy_rgb'][i])] == k[1]
                for r in selected), k))
        def rank(row: dict) -> tuple:
            track = (row['source_id'], row['track_id'])
            reviews = row['review_ids']
            return (used_tracks[track], max((used_reviews[x] for x in reviews), default=0),
                    sha_text(row['id']))
        item = min(groups[key], key=rank)
        groups[key].remove(item)
        selected.append(item)
        used_sources[item['source_id']] += 1
        used_tracks[item['source_id'], item['track_id']] += 1
        used_reviews.update(item['review_ids'])
    assert len(selected) == count
    return selected


def sha_text(value: str) -> str:
    return hashlib.sha256(('29:' + value).encode()).hexdigest()


def sample_audit(rows: list[dict], new_other_mass: float) -> dict:
    counts = Counter((r['label'], r['scope'], r['source_id']) for r in rows)
    sources = Counter((c, d) for c, d, _ in counts)
    domains = {c: {d for cc, d in sources if cc == c} for c in LABELS}
    masses, effective = Counter(), Counter()
    for row in rows:
        c, d, source = row['label'], row['scope'], row['source_id']
        mass = (.7 if d == 'old' else .3) if len(domains[c]) == 2 else 1.
        if c == 'other' and d == 'new':
            mass = new_other_mass
        weight = mass / (sources[c, d] * counts[c, d, source])
        key = f'{d}/{c}'
        masses[key] += weight
        effective[key] += weight * row['weight']
    total = sum(masses.values())
    return dict(n=len(rows), counts=dict(Counter(f"{r['scope']}/{r['label']}" for r in rows)),
                sampling_probability={k: v / total for k, v in masses.items()},
                expected_weighted_loss_mass={k: v / total for k, v in effective.items()})


def main() -> None:
    assert not (OUT / 'protocol.json').exists(), 'Never overwrite a frozen experiment'
    old = read(PREVIOUS / 'dataset.json')
    rows = copy.deepcopy(old['records'])
    positives = [r for r in rows if r['scope'] == 'new' and r['split'] == 'train'
                 and r['label_kind'] == 'human' and r['label'] != 'other']
    negatives = [r for r in rows if r['scope'] == 'new' and r['split'] == 'train'
                 and r['label_kind'] == 'human' and r['label'] == 'other']
    chosen = negative_subset(negatives, len(positives))
    chosen_ids = {r['id'] for r in chosen}
    for row in rows:
        row['train_selected'] = row['split'] == 'train' and (row['scope'] == 'old'
            or row['label_kind'] == 'human' and (row['label'] != 'other' or row['id'] in chosen_ids))
    selected = [r for r in rows if r['train_selected']]
    assert len(positives) == len(chosen) == 39
    assert len(selected) == 5087
    assert all(r['label_kind'] != 'pseudo' for r in selected)
    assert all(r['split'] == 'train' for r in selected)
    assert [r['id'] for r in rows] == [r['id'] for r in old['records']]
    assert all({k: v for k, v in r.items() if k != 'train_selected'} == old['records'][i]
               for i, r in enumerate(rows))
    held = [r for r in rows if r['split'] != 'train']
    assert not {r['group'] for r in selected} & {r['group'] for r in held}
    protocol = read(PREVIOUS / 'protocol.json')
    protocol['user_request'] = 'Keep all eligible human non-other training windows; include only a small reviewed other subset. Preserve old ROI capability.'
    protocol['previous_experiment'] = str(PREVIOUS)
    protocol['previous_dataset_sha256'] = sha(PREVIOUS / 'dataset.json')
    protocol['historical_test_limit'] += ' Previous-round test outcomes were already seen before this revision; repeat test is a regression check, not an untouched test.'
    protocol['weak_policy'] = 'No predicted other is used for training or scoring in this revision.'
    protocol['training']['sampling'] = 'Original class/source sampler; all old train records; all new human train positives; 1:1 unique reviewed other/positive pool. New other class-domain mass .1 instead of .3. No pseudo labels.'
    protocol['training']['epoch_draws'] = dict(rgb=5600, skeleton=3788)
    protocol['training']['epoch_draws_reason'] = 'Match previous effective full batches: RGB700x8, skeleton59x64; isolate data selection rather than reduce optimizer updates.'
    protocol['tuning'].update(alpha=[.972], gate=[35.], action=[.925], object=[.581])
    protocol['tuning']['objective'] = 'Fixed production thresholds. Joint validation checkpoint choice only; old supported-class recall >=baseline-.05, new positive macro recall >=baseline-.10; minimize equal-domain other FPR, then macroF1 and positive TP.'
    protocol['summary'] = dict(previous_train=sample_audit([r for r in rows if r['split'] == 'train'], .3),
                               selected_train=sample_audit(selected, .1))
    save(OUT / 'protocol.json', protocol)
    save(OUT / 'dataset.json', dict(old, records=rows, protocol_sha256=sha(OUT / 'protocol.json'),
                                    train_selection_required=True))
    save(OUT / 'selected-other.json', chosen)
    save(OUT / 'sampling-audit.json', protocol['summary'])
    save(OUT / 'protocol-amendment-before-training.json', dict(reason='User-requested reduced other experiment; all thresholds frozen',
        previous_test_seen=True, current_training_started=False, split_policy='preserve original source/acquisition groups'))
    save(OUT / 'joint-selection-protocol.json', dict(rgb_epochs=list(range(7)), skeleton_epochs=list(range(21)),
        grid_count=147, fixed_parameters=dict(alpha=.972, gate=35., action=.925, object=.581),
        prior_test_seen=True, current_test_used_for_checkpoint_selection=False))
    # Immutable extraction and baseline outputs are reused by exact record order.
    for name in ['skeleton-inputs.npz', 'object-evidence']:
        (OUT / name).symlink_to(PREVIOUS / name, target_is_directory=(PREVIOUS / name).is_dir())
    (OUT / 'predictions').mkdir()
    for branch in ['rgb', 'skeleton']:
        (OUT / 'predictions' / f'{branch}-baseline.npz').symlink_to(PREVIOUS / 'predictions' / f'{branch}-baseline.npz')
        meta = read(PREVIOUS / 'predictions' / f'{branch}-baseline.json')
        meta.update(reused_from=str(PREVIOUS / 'predictions' / f'{branch}-baseline.npz'),
                    dataset_sha256=sha(OUT / 'dataset.json'), exact_record_input_order_preserved=True)
        save(OUT / 'predictions' / f'{branch}-baseline.json', meta)
    save(OUT / 'data-verification.json', dict(all_original_records_unchanged=True, record_order_unchanged=True,
        no_source_group_leakage=True, old_training_retained=5009, new_positives_retained=39,
        new_human_other_selected=39, new_human_other_excluded=242, new_pseudo_other_excluded=278,
        validation_and_test_unsampled=True, xclip_used=False))
    print(json.dumps(protocol['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
