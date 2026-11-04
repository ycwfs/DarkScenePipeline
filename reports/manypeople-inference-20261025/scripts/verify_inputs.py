"""Decode every source frame strictly, without modifying the original recordings."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import time


REPORT = Path(__file__).resolve().parents[1]


def decode(path: str) -> dict:
    started = time.monotonic()
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-err_detect', 'explode',
        '-threads', '2', '-i', path, '-map', '0:v:0', '-an', '-vsync', '0',
        '-progress', 'pipe:1', '-f', 'null', '-']
    proc = subprocess.run(command, capture_output=True, text=True)
    values = dict(line.split('=', 1) for line in proc.stdout.splitlines() if '=' in line)
    return dict(path=path, exit_code=proc.returncode, frames=int(values.get('frame', 0)),
        stderr=proc.stderr, seconds=time.monotonic()-started)


def main() -> None:
    rows = json.loads((REPORT / 'input-manifest.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(decode, [r['absolute_path'] for r in rows]))
    for source, result in zip(rows, results):
        result['expected_frames'] = source['frames']
        result['pass'] = result['exit_code'] == 0 and result['frames'] == source['frames']
        print(Path(result['path']).name, result['frames'], result['pass'], flush=True)
    (REPORT / 'input-verification.json').write_text(json.dumps(results, ensure_ascii=False, indent=2)+'\n')
    if not all(r['pass'] for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
