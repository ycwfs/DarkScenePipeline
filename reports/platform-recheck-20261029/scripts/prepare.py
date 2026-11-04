"""Freeze current platform archive and reuse the exact historical inputs for retesting."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'reports/full-prior-release-20261027'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main():
    archive = ROOT / 'platform/dark-demo-v2.tar.gz'
    manifest = json.loads(Path(str(archive) + '.manifest.json').read_text())
    archive_sha = sha(archive)
    historical = json.loads((OLD / 'archive-verification.json').read_text())
    assert archive_sha == manifest['sha256'] == historical['archives'][0]['sha256']
    assert sha(ROOT / 'platform/dark-demo-v2_full_priors_20261027.tar.gz') == archive_sha
    extracted = OUT / 'release-extracted'
    extracted.mkdir(exist_ok=True)
    with tarfile.open(archive) as source:
        source.extractall(extracted, filter='data')
    operator = extracted / 'dark-demo-v2'
    for rel, record in manifest['files'].items():
        assert sha(operator / rel) == sha(ROOT / 'platform/dark-demo-v2' / rel) == record['sha256'], rel
        assert sha(OLD / 'release-extracted/dark-demo-v2' / rel) == record['sha256'], rel
    save(OUT / 'package-verification.json', dict(passed=True, archive=str(archive.relative_to(ROOT)),
         sha256=archive_sha, files=len(manifest['files']), matches_historical_tested_extraction=True,
         matches_current_source=True, package_modified=False))
    for name in ('run_live.py', 'run_faults.py', 'analyze_live.py', 'analyze_faults.py',
                 'instrumented_operator.py', 'demo.py', 'trim_stream.py'):
        text = (OLD / 'scripts' / name).read_text()
        if name == 'run_live.py':
            text = text.replace("for checkpoint in (10,120,250):", "for checkpoint in (10,120,250,410,610,850):")
            text = text.replace("print('PLAYING: 300-second encoded source at original speed.',flush=True)",
                                "print(f'PLAYING: {manifest[\"seconds\"]:g}-second encoded source at original speed.',flush=True)")
        if name == 'trim_stream.py':
            text = text.replace('start+290', 'start+890').replace('295<=end-start<=305', '895<=end-start<=905')
            text = text.replace('full-output-5min.mp4', 'full-output-15min.mp4')
        (OUT / 'scripts' / name).write_text(text)
    for name in ('inputs.json', 'coverage-limitations.json'):
        shutil.copyfile(OLD / name, OUT / name)
    for item in json.loads((OUT / 'inputs.json').read_text()):
        assert sha(ROOT / item['input']) == item['input_sha256'], item['id']
    for name in ('transport', 'input-live-300s.mp4'):
        path = OUT / name
        if not path.exists():
            path.symlink_to(OLD / name, target_is_directory=name == 'transport')
    original = json.loads((OLD / 'input-manifest.json').read_text())
    assert sha(OUT / 'input-live-300s.mp4') == original['sha256']
    playlist = OUT / 'three-cycles.txt'
    playlist.write_text("file 'input-live-300s.mp4'\n" * 3)
    combined = OUT / 'input-live-900s.mp4'
    if not combined.exists():
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'concat', '-safe', '0',
                        '-i', str(playlist), '-c', 'copy', '-movflags', '+faststart', str(combined)], check=True)
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_packets',
        '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_packets:format=duration', '-of', 'json', str(combined)]))
    assert int(probe['streams'][0]['nb_read_packets']) == 22500
    assert float(probe['format']['duration']) == 900.
    manifest900 = dict(original, input=combined.name, seconds=900., frames=22500, sha256=sha(combined),
                      cycles=3, base_input_sha256=original['sha256'],
                      segments=[dict(s, start=s['start']+cycle*300, cycle=cycle+1)
                                for cycle in range(3) for s in original['segments']],
                      selection='Exactly three copies of historical five-minute input, remuxed without re-encoding')
    save(OUT / 'input-manifest.json', manifest900)
    save(OUT / 'protocol.json', dict(platform_version='full_priors_20261027', archive_sha256=archive_sha,
         stream_seconds=900, fault_seconds=110, same_historical_inputs=23,
         real_image='darkpipe-operator:0.3.0', cards='0,1,2', storage='local', network='localhost RTSP/RTMP',
         goals=['final labels and prior evidence', 'all output formats and viewer reconnection',
                'per-person overlapping clips and graceful draining', 'capture/output fault recovery',
                'cycle throughput, action age, bounded caches and memory observations'],
         limitations=['No connection to onsite platform/vGPU/NFS/HDFS', 'No independent accuracy benchmark',
                      'No source watermark: latency starts at decoded frame delivery, not camera capture']))
    print('Prepared exact current archive, 23 frozen inputs, and 900-second stream', flush=True)


if __name__ == '__main__':
    main()
