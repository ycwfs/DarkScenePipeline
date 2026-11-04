"""Summarize actual per-recording progress and errors without touching inference."""
import json
from pathlib import Path
import time

REPORT = Path(__file__).resolve().parents[1]
sources = json.loads((REPORT/'input-manifest.json').read_text())
total = 0
finished = 0
for folder in sorted((REPORT/'jobs').iterdir()):
    number = int(folder.name)
    progress = folder/'progress.json'
    if not progress.exists():
        continue
    row = json.loads(progress.read_text())
    n = row.get('frames',0)
    total += n
    done = row.get('status')=='finished'
    finished += int(done)
    elapsed = row.get('elapsed',0)
    print(f'{number:02d}: {n}/{sources[number-1]["frames"]} frames, '
          f'{n/sources[number-1]["frames"]:.1%}, fps={n/elapsed if elapsed else 0:.2f}, '
          f'people={row.get("people","-")}, events={row.get("events",0)}, '
          f'status={"finished" if done else row.get("status","running")}, '
          f'updated={time.time()-row.get("updated_at",time.time()):.0f}s ago')
    log = folder/'operator.log'
    errors = [s for s in log.read_text(errors='replace').splitlines()
              if any(word in s for word in ('Traceback','Error:', '[failed]', '写入失败'))]
    for error in errors[-3:]:
        print('ERROR',number,error)
print(f'Total: {total}/80316 frames ({total/80316:.1%}); completed={finished}/9')
