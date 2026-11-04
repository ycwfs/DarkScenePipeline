"""Freeze zero-shot rejection on historical val only before opening test outputs."""
from pathlib import Path
import json
import numpy as np

OUT=Path(__file__).resolve().parents[1]
LABELS=['wave','throw','chase','fall','fight','talk','drink','pick_up','shake_hands','other']


def metrics(rows, probabilities, tau):
    known=[i for i,r in enumerate(rows)if r.get('truth') is not None]
    y=np.array([LABELS.index(rows[i]['truth'])for i in known])
    pp=probabilities[known];pred=pp.argmax(1);pred[(pred!=9)&(pp.max(1)<tau)]=9
    per={}
    for c in sorted(set(y)):
        tp=int(((pred==c)&(y==c)).sum());fp=int(((pred==c)&(y!=c)).sum());fn=int(((pred!=c)&(y==c)).sum())
        per[LABELS[c]]=dict(total=int((y==c).sum()),tp=tp,fp=fp,fn=fn,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),f1=2*tp/max(1,2*tp+fp+fn))
    return dict(windows=len(y),unknown=len(rows)-len(y),correct=int((pred==y).sum()),accuracy=float((pred==y).mean()),
        macro_f1=float(np.mean([v['f1']for v in per.values()])),other_false=int(((y==9)&(pred!=9)).sum()),
        other_total=int((y==9).sum()),per_class=per)


def main():
    files=list((OUT/'old/val').glob('*.json'));assert len(files)==60,len(files)
    rows=[r for p in sorted(files)for r in json.loads(p.read_text())['events']]
    pp=np.array([r['xclip_probs']for r in rows]);sweep=[]
    for i in range(100):
        t=i/100;m=metrics(rows,pp,t);sweep.append(dict(threshold=t,**m))
    choice=max(sweep,key=lambda r:(r['macro_f1'],-r['other_false'],r['threshold']))
    dest=OUT/'selection.json';assert not dest.exists()
    dest.write_text(json.dumps(dict(threshold=choice['threshold'],metrics=choice,criterion='Maximum val macroF1, ties fewer other false alarms, higher threshold',
        no_test_opened=True,val_sources=len(files),val_windows=len(rows)),ensure_ascii=False,indent=2)+'\n')
    (OUT/'val-threshold-sweep.json').write_text(json.dumps(sweep,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(choice,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
