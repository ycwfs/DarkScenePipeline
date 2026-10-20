"""Decode every generated video and compare its header/frame count with metadata."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
files = []
for group in ('live-quality', 'live-off-capped', 'live-quality-capped-ort'):
    files.extend((root / group).rglob('*.mp4'))
    files.extend((root / group).rglob('*.flv'))
for group in ('offline-batch', 'offline-dark', 'offline-shard'):
    files.extend((root / group / 'output').rglob('*.mp4'))

def verify(path):
    probe = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name,width,height,avg_frame_rate,nb_frames,nb_read_frames,duration',
            '-show_entries', 'format=duration,size', '-of', 'json', str(path)], capture_output=True, text=True)
    decode = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-threads', '1',
                '-i', str(path), '-f', 'null', '-'], capture_output=True, text=True)
    result = {'file': str(path.relative_to(root)), 'ffprobe_exit': probe.returncode,
              'decode_exit': decode.returncode, 'decode_error': decode.stderr,
              'probe': json.loads(probe.stdout) if probe.returncode == 0 else probe.stderr}
    sidecar = path.with_suffix('.json')
    if sidecar.exists():
        meta = json.loads(sidecar.read_text())
        stream = result['probe']['streams'][0]
        fps = float(meta['fps'])
        result['metadata'] = meta
        result['metadata_frames_match'] = int(stream['nb_read_frames']) == meta['frames_written']
        result['metadata_duration_error'] = abs(float(result['probe']['format']['duration']) - meta['duration_seconds'])
        result['duration_within_one_frame'] = result['metadata_duration_error'] <= 1 / fps + .02
    return result

with ThreadPoolExecutor(max_workers=2) as pool:
    videos = list(pool.map(verify, sorted(files)))
copies = []
for group in ('offline-batch', 'offline-dark'):
    base = root / group
    for p in sorted((base / 'output').rglob('*')):
        if p.is_file():
            q = base / 'copy' / p.relative_to(base / 'output')
            copies.append({'file': str(p.relative_to(base)),
                           'identical': q.exists() and hashlib.sha256(p.read_bytes()).digest() == hashlib.sha256(q.read_bytes()).digest()})

whole = []
for group in ('offline-batch', 'offline-dark'):
    data = json.loads((root / group / 'session.json').read_text())
    items = data.get('items') or [data]
    for item in items:
        src = Path(item.get('file') or item['local_input'])
        if str(src).startswith('/audit/'):
            src = root / str(src).removeprefix('/audit/')
        path = next(root / p.removeprefix('/audit/') for p in item['output_video'] if '/output/' in p)
        record = next(v for v in videos if v['file'] == str(path.relative_to(root)))
        source_probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                      '-show_entries', 'stream=nb_frames,avg_frame_rate,duration',
                                      '-of', 'json', str(src)], capture_output=True, text=True, check=True)
        sp = json.loads(source_probe.stdout)['streams'][0]
        vp = record['probe']['streams'][0]
        whole.append({'source': str(src), 'output': str(path.relative_to(root)),
                      'input_stream': sp, 'output_stream': vp,
                      'frames_match': int(sp['nb_frames']) == int(vp['nb_read_frames']),
                      'fps_match': sp['avg_frame_rate'] == vp['avg_frame_rate']})
report = {'videos_checked': len(videos), 'decode_failures': [v['file'] for v in videos if v['decode_exit'] or v['decode_error']],
          'metadata_frame_failures': [v['file'] for v in videos if v.get('metadata_frames_match') is False],
          'metadata_duration_failures': [v['file'] for v in videos if v.get('duration_within_one_frame') is False],
          'copy_failures': [c for c in copies if not c['identical']],
          'whole_video_checks': whole, 'copies': copies, 'videos': videos}
(root / 'media-verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps({k: v for k, v in report.items() if k not in ('videos', 'copies')}, ensure_ascii=False, indent=2))
