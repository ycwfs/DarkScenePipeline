"""Wait for the two preparation shards, audit data, then run the fixed protocol."""
import os
from pathlib import Path
import subprocess
import sys
import time

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]
start=time.monotonic();last=-1
while True:
    count=sum(1 for _ in (OUT/'data').glob('*/manifest.json'))
    if count!=last:
        print(f'[data] completed={count}/535',flush=True);last=count
    if count==535:break
    if time.monotonic()-start>3600:raise RuntimeError('Preparation did not finish within one hour')
    alive=False
    for pid in [3011349,3011460]:
        try:os.kill(pid,0);alive=True
        except ProcessLookupError:pass
    if not alive:raise RuntimeError('Preparation processes stopped before all videos completed')
    time.sleep(10)
subprocess.run([sys.executable,str(OUT/'scripts/assemble.py'),'--verify-pixels'],cwd=ROOT,check=True)
subprocess.run([sys.executable,str(OUT/'scripts/train.py')],cwd=ROOT,check=True)
print('[done] Fixed-protocol training and validation selection complete.',flush=True)
