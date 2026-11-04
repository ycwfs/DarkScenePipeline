from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reports/other-action-review-20261101'
AUDIT=ROOT/'reports/low-conf-candidate-audit-20261101'
OP=ROOT/'platform/dark-demo-v2'
sys.path.insert(0,str(OP))
LABELS=['wave','throw','chase','fall','fight','talk','drink','pick_up','shake_hands','other']
BINS=[(.65,.75),(.75,.85),(.85,.925)]

def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def rows():
    raw=read(AUDIT/'replay-rows.json')
    confirmed={r['id']:r['action'] for r in read(AUDIT/'post-review-20261101/confirmed-windows.json')}
    for r in raw:
        if r['id'] in confirmed:
            r['truth']=confirmed[r['id']];r['truth_source']='user_confirmed_20261101'
        else:r['truth_source']='existing_label' if r['truth'] is not None else None
        r['bin']=next((j for j,(lo,hi)in enumerate(BINS)if r['score']>.65 and lo<=r['score']<hi and r['raw_label']!='other'),None)
    return raw
