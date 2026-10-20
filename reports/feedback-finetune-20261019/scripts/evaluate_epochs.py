"""Evaluate completed checkpoints as they arrive, after the baseline cache is complete."""
import json
from pathlib import Path
import subprocess
import sys
import time

OUT=Path(__file__).resolve().parents[1]


def ready(epoch: int) -> bool:
    try:
        history=json.loads((OUT/'training-history.json').read_text())
        baseline=json.loads((OUT/'evaluation/regression/baseline-summary.json').read_text())
        return baseline['Qtrain']['windows']==175 and any(r['epoch']==epoch for r in history)
    except (FileNotFoundError,json.JSONDecodeError):
        return False


def main() -> None:
    for epoch in range(1,5):
        started=time.monotonic()
        while not ready(epoch):
            if time.monotonic()-started>7200:
                raise TimeoutError(f'Checkpoint {epoch} did not become ready; inspect training log')
            time.sleep(10)
        ckpt=OUT/'checkpoints'/f'epoch-{epoch:02d}.pth'
        print('[evaluate]',ckpt.name,flush=True)
        with (OUT/f'eval-epoch-{epoch:02d}.log').open('w') as log:
            subprocess.run([sys.executable,str(OUT/'scripts/replay.py'),'--checkpoint',str(ckpt)],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        summary=json.loads((OUT/'evaluation/regression'/f'epoch-{epoch:02d}-summary.json').read_text())
        print('[result]',epoch,json.dumps({s:{k:r[k] for k in ['windows','correct','fired','rgb_fired']}
                                          for s,r in summary.items()}),flush=True)
    subprocess.run([sys.executable,str(OUT/'scripts/select_checkpoint.py')],check=True)


if __name__=='__main__': main()
