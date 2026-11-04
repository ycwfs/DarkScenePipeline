"""Verify the data boundary, real predictions, exports and unchanged releases."""
from collections import Counter
from datetime import datetime, timezone
import csv
import numpy as np
from common import *


def main() -> None:
    data=read(OUT/'dataset.json');rows=data['records'];protocol=read(OUT/'protocol.json')
    assert data['protocol_sha256']==sha(OUT/'protocol.json')
    previous=Path(protocol['previous_experiment'])
    assert sha(previous/'dataset.json')==protocol['previous_dataset_sha256']
    original=read(previous/'dataset.json')['records']
    assert len(rows)==len(original)==7020
    for row,old in zip(rows,original):
        assert {k:v for k,v in row.items() if k not in ('train_selected','evaluation_role')} == {
            k:v for k,v in old.items() if k!='train_selected'}
    train=[i for i,r in enumerate(rows) if r['train_selected']]
    assert len(train)==5179
    assert sum(rows[i]['scope']=='old' for i in train)==5009
    assert Counter(rows[i]['label'] for i in train if rows[i]['scope']=='new') == {
        'other':85,'chase':34,'talk':33,'wave':12,'drink':6}
    assert all(rows[i]['label_kind']!='pseudo' for i in train)
    old_train={rows[i]['group'] for i in train if rows[i]['scope']=='old'}
    old_held={r['group'] for r in rows if r['scope']=='old' and r['split']!='train'}
    assert not old_train & old_held
    for branch,epochs,positive_count in [('rgb',6,85),('skeleton',20,70)]:
        history=read(OUT/branch/'history.json')
        assert [h['epoch'] for h in history]==list(range(epochs+1))
        coverage=read(OUT/branch/'training-coverage.json')
        assert coverage['new_positives_seen']==coverage['new_positives_enrolled']==positive_count
    lock=read(OUT/'selection-lock.json');val=np.load(OUT/'rgb/val-00.npz')['indices']
    assert all(rows[i]['scope']=='old' and rows[i]['split']=='val' and not rows[i]['train_selected'] for i in val)
    assert len(val)==198 and lock['grid_count']==147 and lock['test_used_for_selection'] is False
    assert lock['config']==dict(rgb='selected',skeleton='selected',alpha=.972,gate=35.,action=.925,object=.581)
    assert all(m['epoch']>0 for m in lock['best_trained_models'].values())
    results=read(OUT/'results.json')
    assert results['selection_lock_sha256']==sha(OUT/'selection-lock.json')
    assert results['groups']==dict(old_val=198,old_test=214,new_positive_training_fit=85,
        new_other_training_fit=85,new_other_untrained_same_sources=566)
    with (OUT/'window-results.csv').open() as stream:scored=list(csv.DictReader(stream))
    assert len(scored)==1148 and len({r['id'] for r in scored})==1148
    for which in ['trained','selected','last']:
        exported=read(OUT/f'export-{which}-verification.json')
        assert exported['production_models_unchanged'] and exported['previous_dataset_unchanged']
        for model in exported['models'].values():
            assert sha(model['path'])==model['sha256']
            assert model['tensors_identical'] and model['weights_only_load'] and model['argmax_equal']
            assert model['max_probability_abs_difference']<.003
    parity=read(OUT/'replay-parity.json')
    assert parity['checked']==4592 and parity['mismatched_labels']==0
    production={}
    for branch,model in protocol['models'].items():
        digest=sha(model['path']);assert digest==model['sha256']
        production[branch]=digest
    archives={}
    for name in ['platform/dark-demo-v2.tar.gz','platform/dark-demo-v2.zip','NVIDIA-Box.zip']:
        path=ROOT/name;digest=sha(path)
        assert digest==path.with_name(path.name+'.sha256').read_text().split()[0]
        archives[name]=digest
    for script in (OUT/'scripts').glob('*.py'):compile(script.read_text(),str(script),'exec')
    names=['README.md','RUN_STATE.md','analysis-report.md','stats-appendix.md','figure-catalog.md',
           'window-results.csv','all85-positive-results.csv','selection-lock.json','protocol.json',
           'adoption-decision.json','replay-parity.json','results.json','branch-results.json',
           'export-verification.json']
    names += [str(p.relative_to(OUT)) for p in sorted((OUT/'figures').glob('*'))]
    save(OUT/'final-artifacts.json',{name:dict(sha256=sha(OUT/name),bytes=(OUT/name).stat().st_size) for name in names})
    summary=dict(complete=True,completed_utc=datetime.now(timezone.utc).isoformat(),
        data_boundary_verified=True,rgb_positives_used=85,skeleton_positives_used=70,
        rgb_epochs=6,skeleton_epochs=20,selected_epochs=lock['epochs'],
        trained_epochs={b:m['epoch'] for b,m in lock['best_trained_models'].items()},
        selection_lock_sha256=sha(OUT/'selection-lock.json'),scored_windows=1148,
        replay_comparisons=4592,parity_mismatches=0,exported_models_verified=True,
        production_model_sha256=production,production_archives_sha256=archives,
        production_modified=False,crowded_positive_holdout=False,
        scope='Old source-held-out historical regression plus all-85 crowded training fit; no real-time acceptance')
    save(OUT/'completion.json',summary)
    print(json.dumps(summary))


if __name__=='__main__':main()
