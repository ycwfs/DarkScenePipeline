"""Post-selection sensitivity at the protocol's fixed thresholds; never reselect on holdout."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reports/feedback-finetune-20261019'
sys.path.insert(0,str(OUT/'unpacked/dark-demo-v2'))
from darkpipe.stages.recognize import reject_probs
from darkpipe.constants import BEHAVIORS
selected=json.loads((OUT/'selection.json').read_text())['selected']
result={}
for model in ['baseline',selected]:
    rows=[]
    for group in ['regression','holdout']:
        rows += list(map(json.loads,(OUT/'evaluation'/group/(model+'.jsonl')).read_text().splitlines()))
    result[model]={}
    for tau in json.loads((OUT/'protocol.json').read_text())['thresholds']:
        summaries={}
        for suite in ['A','B','C','P','Qtrain','Qholdout']:
            sub=[r for r in rows if r['suite']==suite]
            def label(r,key):
                return BEHAVIORS[int(reject_probs(np.array(r[key],np.float32),BEHAVIORS,tau).argmax())]
            summaries[suite]={'windows':len(sub),'correct':sum(label(r,'decision')==r['expect'] for r in sub),
                'fired':sum(label(r,'decision')!='other' for r in sub),
                'rgb_fired':sum(label(r,'rgb')!='other' for r in sub)}
        result[model][str(tau)]=summaries
(OUT/'analysis/threshold-sensitivity.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# 固定阈值敏感性（选模型之后的补充分析）','','保持默认 0.75；未用留出集调阈值或重选检查点。全部数字只适用于本轮回放口径。','','| 阈值 | A正确（旧→新） | C正确（旧→新） | B误报 | P误报 | 新训练误报 | 新留出误报 |','|---|---|---|---|---|---|---|']
for tau in result['baseline']:
    a,b=result['baseline'][tau],result[selected][tau]
    values=[tau]+[f"{a[s]['correct']}→{b[s]['correct']}/{a[s]['windows']}" for s in ['A','C']]+[f"{a[s]['fired']}→{b[s]['fired']}/{a[s]['windows']}" for s in ['B','P','Qtrain','Qholdout']]
    lines.append('| '+' | '.join(values)+' |')
(OUT/'analysis/threshold-sensitivity.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
