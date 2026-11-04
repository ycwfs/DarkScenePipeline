"""Wait for both validation protocols, select once, export, then open heldout tests."""
import json
from pathlib import Path
import subprocess
import sys
import time

from common import OUT, save


def main():
    while not all((OUT/f).exists()for f in ['selection.json','retention-selection.json']):
        time.sleep(10)
    broad=json.loads((OUT/'selection.json').read_text())
    retained=json.loads((OUT/'retention-selection.json').read_text())
    assert len(broad['rows'])==len(retained['rows'])==20
    save(OUT/'broad-selection.json',broad)
    rows=[dict(r,family=family)for family,d in [('broad',broad),('retention',retained)]for r in d['rows']]
    eligible=[r for r in rows if r['eligible']]
    selected=max(eligible or rows,key=lambda r:(r['score'],-r['epoch']))
    save(OUT/'selection.json',dict(baseline=broad['baseline'],rows=rows,selected=selected,
        generic_guard_passed=bool(eligible),criterion=broad['criterion'],
        note='Second conservative family added after broad validation regression, before heldout test; one seed, two validation-selected families'))
    print('LOCKED',selected['family'],selected['epoch'],selected['score'],flush=True)
    for script,args in [('export.py',[]),('validate_select.py',['--test']),('evaluate.py',[]),('cached_windows.py',[])]:
        with (OUT/f'{Path(script).stem}-final.log').open('w')as log:
            subprocess.run([sys.executable,str(OUT/'scripts'/script),*args],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('FINISHED',script,flush=True)
    save(OUT/'model-evaluation-complete.json',dict(passed=True))


if __name__=='__main__':main()
