"""Check effective sampler, cache alignment, held-out isolation and available poses."""
from collections import Counter
import numpy as np
import torch
from common import *
from train_models import weights, ShardedWeightedSampler


def main() -> None:
    rows = read(OUT / 'dataset.json')['records']
    cache = np.load(OUT / 'skeleton-inputs.npz')
    assert cache['features'].shape[1] == len(rows)
    valid = cache['valid'][GATES.index(35.)]
    results = {}
    for branch in ['rgb', 'skeleton']:
        ids = [i for i, r in enumerate(rows) if r['train_selected'] and (branch == 'rgb' or valid[i])]
        assert all(rows[i]['split'] == 'train' and rows[i]['label_kind'] != 'pseudo' for i in ids)
        values = np.array(weights(rows, ids))
        probs = values / values.sum()
        summary = Counter()
        for j, i in enumerate(ids):
            summary[f"{rows[i]['scope']}/{rows[i]['label']}"] += float(probs[j])
        results[branch] = dict(n=len(ids), counts=dict(Counter(f"{rows[i]['scope']}/{rows[i]['label']}" for i in ids)),
                               sampler_mass=dict(summary))
        if branch == 'rgb':
            expected = read(OUT / 'sampling-audit.json')['selected_train']['sampling_probability']
            assert all(abs(summary[k] - expected[k]) < 1e-12 for k in summary)
            draws = torch.multinomial(torch.as_tensor(values), 5600, True,
                                     generator=torch.Generator().manual_seed(30)).tolist()
            parts = []
            for rank in range(4):
                sampler = ShardedWeightedSampler(values, 5600, rank, 4, 29)
                sampler.epoch = 1
                parts.append(list(sampler))
            assert [parts[i % 4][i // 4] for i in range(5600)] == draws
            results[branch]['global_draw_equal_across_ddp_world_sizes'] = True
    for branch in ['rgb', 'skeleton']:
        with np.load(OUT / 'predictions' / f'{branch}-baseline.npz') as pred:
            assert pred['ids'].tolist() == [r['id'] for r in rows]
    assert all(Path(r['path']).exists() for r in rows)
    for model in read(OUT / 'protocol.json')['models'].values():
        assert sha(model['path']) == model['sha256']
    save(OUT / 'training-input-verification.json', dict(passed=True, branches=results,
        exact_baseline_input_order=True, all_production_model_hashes_match=True))
    print(json.dumps(results))


if __name__ == '__main__':
    main()
