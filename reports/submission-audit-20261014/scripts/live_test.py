"""Drive the original operator with argv from its manifest and a paced local HTTP stream."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import threading
import time
import urllib.request

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
PKG = ROOT / 'package/dark-demo-v2'
SOURCE = Path('/ssd1/wfs/project/lowlight/DarkScenePipeline/real-actiondata-0909/喝水16.mp4')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip-denoise', default='quality')
    ap.add_argument('--seconds', default=75, type=int)
    ap.add_argument('--thread-caps', action='store_true')
    ap.add_argument('--ort-tuned', action='store_true')
    ap.add_argument('--sse-seconds', type=float, default=35,
                    help='0 keeps the SSE connection open until shutdown (exit-bug reproduction)')
    a = ap.parse_args()
    tag = 'live-' + a.clip_denoise + ('-capped' if a.thread_caps else '')
    if a.ort_tuned:
        tag += '-ort'
    if a.sse_seconds <= 0:
        tag += '-persistent-sse'
    out = ROOT / tag
    out.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(SOURCE))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        good, enc = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        assert good
        frames.append(enc.tobytes())
    cap.release()
    assert len(frames) == 75
    stop = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=sourceframe')
            self.end_headers()
            i, start = 0, time.monotonic()
            try:
                while not stop.is_set():
                    jpg = frames[i % len(frames)]
                    header = (b'--sourceframe\r\nContent-Type: image/jpeg\r\nContent-Length: '
                              + str(len(jpg)).encode() + b'\r\n\r\n')
                    self.wfile.write(header + jpg + b'\r\n')
                    self.wfile.flush()
                    i += 1
                    stop.wait(max(0, start + i / 25 - time.monotonic()))
            except (BrokenPipeError, ConnectionResetError):
                pass

    source = ThreadingHTTPServer(('127.0.0.1', 18764), Handler)
    threading.Thread(target=source.serve_forever, daemon=True).start()
    doc = json.loads((PKG / 'suanzi.json').read_text())
    values = {i['name']: i.get('default', '') for i in doc['inputs']}
    values.update(video_path='http://127.0.0.1:18764/live.mjpg',
                  clip_dir='/audit/' + tag + '/clips', session_json='/audit/' + tag + '/session.json',
                  serve_port=18043, run_seconds=a.seconds, extra_hosts='', clip_denoise=a.clip_denoise)
    argv = []
    for item in doc['implementation']['container']['args']:
        if isinstance(item, str):
            argv.append(item)
        else:
            value = values[list(item.values())[0]]
            argv.append(str(value).lower() if isinstance(value, bool) else str(value))
    name = 'darkpipe-audit-' + tag + '-20261014'
    cmd = ['docker', 'run', '--rm', '--name', name, '--gpus', '"device=0,1"',
           '--network', 'host', '--cpus', '8', '--memory', '12g',
           '-e', 'PYTHONDONTWRITEBYTECODE=1',
           '-v', str(PKG) + ':/opt/darkpipe/op:ro', '-v', str(ROOT) + ':/audit',
           '-w', '/opt/darkpipe/op', doc['implementation']['container']['image'],
           *doc['implementation']['container']['command'], *argv]
    if a.thread_caps:
        where = cmd.index('-w')
        cmd[where:where] = ['-e', 'OMP_NUM_THREADS=4', '-e', 'MKL_NUM_THREADS=4',
                            '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'OPENCV_FOR_THREADS_NUM=2']
    if a.ort_tuned:
        cmd[cmd.index('main.py')] = '/audit/ort_launcher.py'
    (out / 'command.json').write_text(json.dumps(cmd, ensure_ascii=False, indent=2))
    log = (out / 'container.log').open('w')
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    health, events = [], []
    started = time.monotonic()
    consumers = []

    def sse():
        try:
            with urllib.request.urlopen('http://127.0.0.1:18043/events', timeout=15) as r:
                deadline = time.monotonic() + a.sse_seconds if a.sse_seconds > 0 else float('inf')
                for line in r:
                    if line.startswith(b'data: '):
                        events.append({'received_wall': time.time(), 'event': json.loads(line[6:])})
                    if time.monotonic() >= deadline:
                        break
        except Exception as e:
            (out / 'sse-error.txt').write_text(str(e))

    def mjpeg():
        try:
            with urllib.request.urlopen('http://127.0.0.1:18043/stream', timeout=15) as r:
                buf = b''
                while len(buf) < 5_000_000:
                    buf += r.read(16384)
                    begin, end = buf.find(b'\xff\xd8'), buf.find(b'\xff\xd9')
                    if begin >= 0 and end > begin:
                        (out / 'frame.jpg').write_bytes(buf[begin:end + 2])
                        return
        except Exception as e:
            (out / 'mjpeg-error.txt').write_text(str(e))

    try:
        clients_started = False
        with (out / 'health.jsonl').open('w') as hp:
            while proc.poll() is None and time.monotonic() - started < a.seconds + 120:
                before = time.time()
                try:
                    with urllib.request.urlopen('http://127.0.0.1:18043/health', timeout=3) as r:
                        status = r.status
                        h = json.load(r)
                    h.update(_received_wall=time.time(), _request_wall=before, _http_status=status)
                    health.append(h)
                    hp.write(json.dumps(h, ensure_ascii=False) + '\n')
                    hp.flush()
                    if h['uptime_s'] >= 40 and not (out / 'cpu-stat.txt').exists():
                        stats = subprocess.run(['docker', 'exec', name, 'cat', '/sys/fs/cgroup/cpu.stat'],
                                               capture_output=True, text=True)
                        (out / 'cpu-stat.txt').write_text(stats.stdout)
                    if not clients_started and h['frames_dropped'] >= 0 and h['events_total'] > 0:
                        clients_started = True
                        with urllib.request.urlopen('http://127.0.0.1:18043/config', timeout=3) as r:
                            (out / 'resolved-config.json').write_text(json.dumps(json.load(r), indent=2))
                        threading.Thread(target=sse, daemon=True).start()
                        threading.Thread(target=mjpeg, daemon=True).start()
                        flv_log = (out / 'flv-consumer.log').open('w')
                        player = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error',
                            '-y', '-i', 'http://127.0.0.1:18043/live.flv', '-t', '10',
                            '-c:v', 'copy', '-an', str(out / 'sample.flv')],
                            stdout=flv_log, stderr=subprocess.STDOUT)
                        consumers.append((player, flv_log))
                except Exception:
                    pass
                time.sleep(1)
        if proc.poll() is None:
            subprocess.run(['docker', 'stop', '-t', '20', name], capture_output=True)
        rc = proc.wait(timeout=40)
    finally:
        stop.set()
        source.shutdown()
        source.server_close()
        log.close()
        for player, player_log in consumers:
            if player.poll() is None:
                player.terminate()
                player.wait(timeout=10)
            player_log.close()
    (out / 'sse-events.json').write_text(json.dumps(events, ensure_ascii=False, indent=2))
    stable = [h for h in health if h['uptime_s'] >= 20 and h['fps_proc'] > 0]
    tstarts = [(h['_received_wall'] + h['_request_wall']) / 2 - h['uptime_s'] for h in stable]
    reference_start = float(np.median(tstarts)) if tstarts else 0
    event_ages = [(e['received_wall'] - reference_start - e['event']['timestamp']) * 1000
                  for e in events if e['event']['timestamp'] >= 20]
    report = {'exit_code': rc, 'seconds_requested': a.seconds, 'source': str(SOURCE),
              'thread_caps': a.thread_caps,
              'ort_tuned': a.ort_tuned,
              'sse_seconds': a.sse_seconds,
              'source_protocol': 'HTTP multipart MJPEG paced at 25fps; looping real 1080p take',
              'physical_gpus': [0, 1], 'cpu_limit': 8, 'memory_limit_gib': 12,
              'steady_health_samples': len(stable), 'sse_events': len(events),
              'health_last': health[-1] if health else None}
    def describe(nums):
        return dict(zip(['min', 'p50', 'p95', 'max'], map(float, np.percentile(nums, [0, 50, 95, 100])))) if nums else None
    report['fps_proc'] = describe([h['fps_proc'] for h in stable])
    report['frame_latency_ms'] = describe([h['latency_ms_last'] for h in stable])
    report['recognition_event_age_ms'] = describe(event_ages)
    if (out / 'session.json').exists():
        report['session'] = json.loads((out / 'session.json').read_text())
    report['saved_mp4'] = [str(p.relative_to(ROOT)) for p in (out / 'clips').rglob('*.mp4')]
    report['log_errors'] = [line for line in (out / 'container.log').read_text().splitlines()
                            if 'Traceback' in line or 'died:' in line or '骨架分支关闭' in line]
    (out / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
