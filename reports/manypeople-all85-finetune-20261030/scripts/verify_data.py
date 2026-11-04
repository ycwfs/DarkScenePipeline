"""Verify sampling and old holdouts before any optimizer updates."""
from collections import Counter
import numpy as np
import torch
from common import *
from train_models import weights, ShardedWeightedSampler, evaluation_indices


def main() -> None:
    rows = read(OUT/'dataset.json')['records']
    old = read(ROOT/'reports/manypeople-positive-rebalance-20261030/dataset.json')['records']
    assert len(rows) == len(old) == 7020
    assert all({k:v for k,v in a.items() if k not in ('train_selected','evaluation_role')}
               == {k:v for k,v in b.items() if k!='train_selected'} for a,b in zip(rows,old))
    features = np.load(OUT/'skeleton-inputs.npz')
    valid = features['valid'][GATES.index(35.)]
    summary = {}
    for branch in ['rgb','skeleton']:
        ids = [i for i,r in enumerate(rows) if r['train_selected'] and (branch=='rgb' or valid[i])]
        val = evaluation_indices(rows,branch,valid)
        assert not set(ids)&set(val)
        assert all(rows[i]['scope']=='old' and rows[i]['split']=='val' for i in val)
        assert not {rows[i]['group'] for i in ids if rows[i]['scope']=='old'} & {rows[i]['group'] for i in val}
        values=np.array(weights(rows,ids));probs=values/values.sum();mass=Counter()
        for j,i in enumerate(ids):mass[f"{rows[i]['scope']}/{rows[i]['label']}"]+=float(probs[j])
        summary[branch]=dict(train=len(ids),validation=len(val),sampling_probability=dict(mass))
        if branch=='rgb':
            expected=read(OUT/'sampling-audit.json')['selected_train']['sampling_probability']
            assert all(abs(mass[k]-expected[k])<1e-12 for k in mass)
            seen=set()
            for epoch in range(1,7):
                parts=[]
                for rank in range(4):
                    sampler=ShardedWeightedSampler(values,5600,rank,4,29);sampler.epoch=epoch
                    parts.append(list(sampler))
                draw=[parts[i%4][i//4] for i in range(5600)]
                ref=torch.multinomial(torch.as_tensor(values),5600,True,generator=torch.Generator().manual_seed(29+epoch)).tolist()
                assert draw==ref
                seen.update(ids[i] for i in draw)
            positives={i for i in ids if rows[i]['scope']=='new' and rows[i]['label']!='other'}
            assert len(positives)==85 and positives<=seen
            summary[branch]['all_85_scheduled_in_six_epochs']=True
    for i in scored_indices(rows):
        assert Path(rows[i]['path']).is_file() and Path(rows[i]['feature_path']).is_file()
    for model in read(OUT/'protocol.json')['models'].values():assert sha(model['path'])==model['sha256']
    save(OUT/'training-input-verification.json',dict(passed=True,branches=summary,production_hashes_match=True))
    print(json.dumps(summary))


if __name__=='__main__':main()
