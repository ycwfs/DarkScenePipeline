"""Re-decode every raw field video, retaining every frame whose raw 720-side L < 75."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / 'reports/person-finetune-20261023'
RAW_DIRS = ('real-actiondata-all', 'real-actiondata-0909', 'real-actiondata', 'real-actiondata-cutdrink')


def save(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n')
    tmp.replace(path)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def downscale(frame: np.ndarray, side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if max(h, w) <= side:
        return frame
    factor = side / max(h, w)
    return cv2.resize(frame, (round(w*factor)//2*2, round(h*factor)//2*2), interpolation=cv2.INTER_AREA)


def scan(video: dict) -> dict:
    destination = OUT / 'data' / video['id']
    cache = destination / 'scan.json'
    # Even a resumed scan verifies all source aliases against the frozen digest.
    for alias in video['aliases']:
        if digest(ROOT / alias) != video['sha256']:
            raise ValueError(f'Source content changed: {alias}')
    if cache.exists():
        return json.loads(cache.read_text())
    old_path = PRIOR/'data'/video['id']/'manifest.json'
    old = json.loads(old_path.read_text()) if old_path.exists() else None
    timestamps = None
    if Path(video['source']).suffix.lower() == '.avi':
        probe = subprocess.run(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','frame=best_effort_timestamp_time','-of','json',str(ROOT/video['source'])],
            check=True,capture_output=True,text=True)
        if probe.stderr.strip():
            raise ValueError(f'Decode warnings in {video["source"]}: {probe.stderr}')
        timestamps=[float(f['best_effort_timestamp_time']) for f in json.loads(probe.stdout)['frames']]
    cap = cv2.VideoCapture(str(ROOT / video['source']))
    if not cap.isOpened():
        raise ValueError(video['source'])
    frames, max_delta, started = [], 0., time.monotonic()
    try:
        while True:
            ok, native = cap.read()
            if not ok:
                break
            index = len(frames)
            small = downscale(native, 720)
            b, g, r = cv2.mean(small)[:3]
            luma = .114*b + .587*g + .299*r
            if old is not None:
                max_delta = max(max_delta, abs(luma - old['frames'][index]['luma']))
            record = dict(frame=index, timestamp=timestamps[index] if timestamps is not None else index/video['fps'],
                          luma=luma, eligible=luma < 75)
            if record['eligible']:
                preview = downscale(native, 960)
                image = destination/'raw'/f'{index:06d}.jpg'
                image.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(image), preview, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise IOError(image)
                record.update(image=image.relative_to(OUT).as_posix(),
                              preview_size=list(preview.shape[1::-1]),
                              previous_boxes=old['frames'][index]['tracks'] if old else None,
                              previous_work_size=list(small.shape[1::-1]))
            frames.append(record)
    finally:
        cap.release()
    if timestamps is not None:
        assert len(frames) == len(timestamps), video['source']
        video=dict(video,container_frames=video['frames'],frames=len(frames),
                   decoded_timestamps='ffprobe best_effort_timestamp_time; AVI index may include absent chunks')
    else:
        assert len(frames) == video['frames'], video['source']
    if old is not None:
        assert len(frames) == len(old['frames']), video['source']
    assert max_delta < 1e-8, (video['source'], max_delta)
    result = dict(video=video, frames=frames, dark_frames=sum(f['eligible'] for f in frames),
                  full_decode=True, all_alias_hashes_verified=True, cached_luma_max_delta=max_delta,
                  elapsed=time.monotonic()-started)
    save(cache, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    inventory = json.loads((PRIOR/'source-inventory.json').read_text())
    raw = [v for v in inventory['videos'] if not v['processed']]
    aliases = {p for v in raw for p in v['aliases']}
    actual = {p.relative_to(ROOT).as_posix() for name in RAW_DIRS for p in (ROOT/name).rglob('*')
              if p.suffix.lower() in {'.mp4', '.avi', '.mkv', '.mov', '.m4v'}}
    assert not aliases-actual, {'missing': sorted(aliases-actual)}
    # The earlier MP4-only inventory omitted AVI originals; include every video extension.
    for source in sorted(actual-aliases):
        path = ROOT/source
        sha = digest(path)
        duplicate = next((v for v in raw if v['sha256'] == sha), None)
        if duplicate is not None:
            duplicate['aliases'].append(source)
            continue
        cap = cv2.VideoCapture(str(path))
        video = dict(source=source, id=sha[:16], sha256=sha, aliases=[source], processed=False,
            width=int(cap.get(3)), height=int(cap.get(4)), frames=int(cap.get(7)), fps=float(cap.get(5)),
            bytes=path.stat().st_size, added_by_full_extension_scan=True)
        cap.release()
        assert min(video['width'], video['height'], video['frames'], video['fps']) > 0
        raw.append(video)
    aliases = actual
    splits = {r['video_id']: r for r in json.loads((PRIOR/'splits.json').read_text())['records']}
    for v in raw:
        v['frozen_split'] = splits.get(v['id'], dict(video_id=v['id'], split='unassigned',
            group=None, reason='Previously unlisted AVI; original/cut recording family must be reconciled before training.'))
    save(OUT/'inventory.json', dict(raw_videos=raw,
        excluded_processed=[v for v in inventory['videos'] if v['processed']],
        exclusion_reason='Processed platform videos have no recoverable pre-enhancement luma.',
        raw_directories=list(RAW_DIRS), screening='Every frame; BGR mean on unenhanced longest-side-720 image; strict L<75',
        preview_only='960-side JPEG is for review only. Future training must decode original source frames.',
        include_in_training=False))
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, result in enumerate(pool.map(scan, raw), 1):
            results.append(result)
            print(f'[scan] {index}/{len(raw)} {result["video"]["source"]} dark={result["dark_frames"]}', flush=True)
    summary = dict(raw_videos=len(results), raw_files=len(aliases),
        source_frames=sum(len(v['frames']) for v in results),
        dark_videos=sum(v['dark_frames'] > 0 for v in results),
        dark_frames=sum(v['dark_frames'] for v in results),
        previous_no_box_dark_frames=sum(f['previous_boxes'] == [] for v in results for f in v['frames'] if f['eligible']),
        newly_included_raw_videos=sum(bool(v['video'].get('added_by_full_extension_scan')) for v in results),
        full_decode=True, all_alias_hashes_verified=True,
        cached_luma_max_delta=max(v['cached_luma_max_delta'] for v in results))
    save(OUT/'scan-summary.json', summary)
    frozen=json.loads((OUT/'inventory.json').read_text())
    frozen['raw_videos']=[r['video'] for r in results]
    save(OUT/'inventory.json',frozen)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
