"""Choose using regression/train evidence only; never read the sealed holdout here."""
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reports/feedback-finetune-20261019'


def main() -> None:
    directory=OUT/'evaluation/regression'
    baseline=json.loads((directory/'baseline-summary.json').read_text())
    candidates=[]
    for file in sorted(directory.glob('epoch-*-summary.json')):
        name=file.name.removesuffix('-summary.json')
        metrics=json.loads(file.read_text())
        failures=[]
        for suite in ('A','C'):
            if metrics[suite]['correct']<baseline[suite]['correct']:
                failures.append(f'{suite}: correct {metrics[suite]["correct"]} < {baseline[suite]["correct"]}')
        for suite in ('B','P'):
            if metrics[suite]['fired']>baseline[suite]['fired']:
                failures.append(f'{suite}: fires {metrics[suite]["fired"]} > {baseline[suite]["fired"]}')
        rows=[json.loads(line) for line in (directory/(name+'.jsonl')).read_text().splitlines()]
        q=[r for r in rows if r['suite']=='Qtrain']
        mean=float(np.mean([max(r['decision'][:9]) for r in q]))
        score=[metrics['Qtrain']['fired'],metrics['Qtrain']['rgb_fired'],
               -metrics['A']['correct']-metrics['C']['correct'],mean,name]
        candidates.append(dict(name=name,acceptable=not failures,gate_failures=failures,
                               score=score,metrics=metrics,mean_max_named_qtrain=mean))
    assert len(candidates)>=4,'All planned epochs must be compared before selection'
    passing=sorted((r for r in candidates if r['acceptable']),key=lambda r:r['score'])
    result=dict(baseline=baseline,candidates=candidates,
                selected=passing[0]['name'] if passing else None,
                holdout_used=False,rule='protocol.json',
                reason='Meets all ABCP gates; lexicographic Qtrain ranking' if passing else 'No checkpoint passed all regression gates')
    (OUT/'selection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('baseline','candidates')},ensure_ascii=False,indent=2))
    for row in candidates:
        print(row['name'],'accepted',row['acceptable'],'score',row['score'],'failures',row['gate_failures'])


if __name__=='__main__': main()
