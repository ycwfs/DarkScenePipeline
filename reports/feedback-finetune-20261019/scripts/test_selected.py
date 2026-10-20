"""Open the reserved sessions only after the regression-only selection is final."""
import json
from pathlib import Path
import subprocess
import sys
import time

OUT=Path(__file__).resolve().parents[1]


def main() -> None:
    start=time.monotonic()
    while True:
        try:
            selection=json.loads((OUT/'selection.json').read_text())
            break
        except (FileNotFoundError,json.JSONDecodeError):
            if time.monotonic()-start>7200:
                raise TimeoutError('No final selection; inspect regression results')
            time.sleep(10)
    assert selection['selected'], 'No checkpoint met regression requirements'
    assert not selection['holdout_used']
    selected=selection['selected']
    print('[selected]',selected,flush=True)
    for name,command in [
        ('holdout-build',[sys.executable,str(OUT/'scripts/replay.py'),'--build','--group','holdout']),
        ('holdout-selected',[sys.executable,str(OUT/'scripts/replay.py'),'--group','holdout',
                             '--checkpoint',str(OUT/'checkpoints'/(selected+'.pth'))]),
    ]:
        with (OUT/(name+'.log')).open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('[finished]',name,flush=True)


if __name__=='__main__': main()
