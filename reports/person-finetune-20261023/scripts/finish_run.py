"""Evaluate and render only after the fixed training protocol has selected a checkpoint."""
import json
from pathlib import Path
import subprocess
import sys
import time

OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parents[1]
start=time.monotonic()
while not (OUT/'selection.json').exists():
    if time.monotonic()-start>10800:raise RuntimeError('Training selection not ready after three hours')
    time.sleep(20)

def run(script: str,*args: str) -> None:
    print('[run]',script,*args,flush=True)
    subprocess.run([sys.executable,'-u',str(OUT/'scripts'/script),*args],cwd=ROOT,check=True)

run('evaluate_test.py')
run('analyze.py')
run('render_demo.py')
protocol=json.loads((OUT/'protocol.json').read_text())
run('render_demo.py','--checkpoint',str(ROOT/'platform/dark-demo-v2/ckpts'/protocol['init']),
    '--output',str(OUT/'demo-baseline'))
run('compare_demos.py')
run('review_index.py')
print('[done] Paired evaluation, figures, independent demos, and decoding verification complete.',flush=True)
