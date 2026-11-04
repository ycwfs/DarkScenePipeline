"""Real native GPU verification: isolated Box, file input, live I/O and graceful shutdown."""
import argparse
import json
import os
import signal
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import cv2


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def probe(path):
    cmd = ['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
           '-show_entries', 'stream=codec_name,width,height,nb_read_frames,duration', '-of', 'json', str(path)]
    result = json.loads(subprocess.check_output(cmd))['streams'][0]
    assert result['codec_name'] == 'h264', (path, result)
    assert int(result['nb_read_frames']) > 0, (path, result)
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'null', '-'],
                   capture_output=True, check=True, timeout=60)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--box', type=Path, required=True)
    ap.add_argument('--python', type=Path, required=True)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--gpu', default='1')
    args = ap.parse_args()
    box, source, out = [p.resolve() for p in (args.box, args.source, args.output)]
    # Resolving a venv's python symlink switches back to the base environment.
    python = args.python.absolute()
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED='1')
    command = [str(python), str(box / 'run.py')]
    report = {'box': str(box), 'source': str(source), 'physical_gpu': args.gpu,
              'native': True, 'container_used': False}
    offline = out / 'offline'
    session_file = offline / 'session.json'
    cmd = command + ['--video_path', str(source), '--output-dir', str(offline),
                     '--session_json', str(session_file)]
    with (out / 'offline.log').open('w') as log:
        result = subprocess.run(cmd, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT,
                                timeout=420, check=False)
    assert result.returncode == 0, (out / 'offline.log')
    session = json.loads(session_file.read_text())
    events_path = Path(session['events_json'][0])
    events = json.loads(events_path.read_text())
    if isinstance(events, dict):
        events = events['events']
    videos = {str(p.relative_to(offline)): probe(p) for p in offline.rglob('*.mp4')}
    assert len(videos) >= 2, videos
    whole = next(v for k, v in videos.items() if k.endswith('-处理后.mp4'))
    cap = cv2.VideoCapture(str(source))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert int(whole['nb_read_frames']) == count, (whole, count)
    jpgs = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        assert ok
        jpgs.append(jpg.tobytes())
    cap.release()
    assert any(e['skeleton_active'] and e['actor_poses'] for e in events), events
    report['offline'] = {'session': session, 'videos': videos, 'events': len(events),
                         'events_with_actor_poses': sum(bool(e['actor_poses']) for e in events)}
    (out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))

    source_port, service_port, push_port = free_port(), free_port(), free_port()
    stop_source = threading.Event()

    class Source(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            i, start = 0, time.monotonic()
            try:
                while not stop_source.is_set():
                    data = jpgs[i % len(jpgs)]
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' +
                                     str(len(data)).encode() + b'\r\n\r\n' + data + b'\r\n')
                    self.wfile.flush()
                    i += 1
                    stop_source.wait(max(0, start + i / 25 - time.monotonic()))
            except (BrokenPipeError, ConnectionResetError):
                pass

    http = ThreadingHTTPServer(('127.0.0.1', source_port), Source)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    online = out / 'online'
    online.mkdir()
    push_url = f'rtmp://127.0.0.1:{push_port}/live/box'
    push_log = (out / 'push-receiver.log').open('w')
    receiver = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                                 '-listen', '1', '-i', push_url, '-c:v', 'copy', '-an',
                                 str(online / 'rtmp.flv')], stdout=push_log, stderr=subprocess.STDOUT)
    cmd = command + ['--video_path', f'http://127.0.0.1:{source_port}/live.mjpg',
                     '--output-dir', str(online / 'saved'), '--session_json', str(online / 'session.json'),
                     '--serve_port', str(service_port), '--stream_formats', 'mjpeg,flv,hls',
                     '--rtmp_push_url', push_url, '--run_seconds', '90']
    (out / 'online-command.json').write_text(json.dumps(cmd, ensure_ascii=False, indent=2))
    log = (out / 'online.log').open('w')
    proc = subprocess.Popen(cmd, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT)
    base = f'http://127.0.0.1:{service_port}'
    consumers, threads, health, sse_events, errors = [], [], [], [], []
    sent_signal = False
    started = time.monotonic()

    def sse():
        try:
            with urlopen(base + '/events', timeout=20) as stream:
                for line in stream:
                    if line.startswith(b'data: '):
                        sse_events.append(json.loads(line[6:]))
        except Exception as exc:  # noqa: BLE001 -- propagate client thread failures to assertions
            errors.append(f'SSE: {exc}')

    def mjpeg():
        try:
            with urlopen(base + '/stream', timeout=20) as stream:
                data = b''
                while len(data) < 10_000_000:
                    data += stream.read(16384)
                    a, b = data.find(b'\xff\xd8'), data.find(b'\xff\xd9')
                    if 0 <= a < b:
                        (online / 'frame.jpg').write_bytes(data[a:b+2])
                        return
                raise RuntimeError('no JPEG')
        except Exception as exc:  # noqa: BLE001 -- propagate client thread failures to assertions
            errors.append(f'MJPEG: {exc}')

    last_poll_error = None
    try:
        consumers_started = False
        while proc.poll() is None and time.monotonic() - started < 300:
            try:
                try:
                    response = urlopen(base + '/health', timeout=3)
                except HTTPError as exc:
                    response = exc
                with response:
                    h = json.load(response)
                    h['_http_status'] = response.status
                health.append(h)
                if not consumers_started and h.get('events_total', 0) > 0:
                    consumers_started = True
                    with urlopen(base + '/config', timeout=3) as response:
                        (out / 'resolved-config.json').write_text(json.dumps(json.load(response), indent=2))
                    for target in (sse, mjpeg):
                        thread = threading.Thread(target=target, daemon=True)
                        thread.start()
                        threads.append(thread)
                    for name, endpoint in (('flv', '/live.flv'), ('hls', '/hls/index.m3u8')):
                        client_log = (out / f'{name}-client.log').open('w')
                        client = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error',
                            '-y', '-i', base + endpoint, '-t', '6', '-an', '-c:v', 'copy',
                            str(online / f'{name}.mp4')], stdout=client_log, stderr=subprocess.STDOUT)
                        consumers.append((name, client, client_log))
                if h.get('uptime_s', 0) >= 65 and consumers_started and not sent_signal:
                    proc.send_signal(signal.SIGTERM)
                    sent_signal = True
            except (OSError, ValueError) as exc:
                last_poll_error = str(exc)  # connection refused is expected during startup/shutdown
            time.sleep(1)
        if proc.poll() is None:
            raise RuntimeError('native service did not exit within 300 seconds')
        rc = proc.wait()
        for _, player, _ in consumers:
            player.wait(timeout=30)
        receiver.wait(timeout=30)
        for thread in threads:
            thread.join(timeout=10)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for player in [receiver] + [p for _, p, _ in consumers]:
            if player.poll() is None:
                player.terminate()
                try:
                    player.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    player.kill()
                    player.wait()
        for _, _, handle in consumers:
            handle.close()
        stop_source.set()
        http.shutdown()
        http.server_close()
        log.close()
        push_log.close()
    session = json.loads((online / 'session.json').read_text())
    videos = {str(p.relative_to(online)): probe(p) for p in online.rglob('*.mp4')}
    videos['rtmp.flv'] = probe(online / 'rtmp.flv')
    (out / 'health.json').write_text(json.dumps(health, ensure_ascii=False, indent=2))
    (out / 'sse-events.json').write_text(json.dumps(sse_events, ensure_ascii=False, indent=2))
    report['online'] = {'exit_code': rc, 'sent_sigterm': sent_signal, 'session': session,
                        'videos': videos, 'sse_events': len(sse_events),
                        'health_http_200': sum(h['_http_status'] == 200 for h in health),
                        'client_errors': errors, 'last_poll_error': last_poll_error,
                        'clients': {n:p.returncode for n,p,_ in consumers}}
    (out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    assert rc == 0 and session['status'] == 'stopped' and sent_signal, report['online']
    assert all(session['clips'][key] == 0 for key in
               ('clips_forced', 'clips_failed', 'clips_unfinished', 'write_errors')), session['clips']
    assert session['event_log_stats']['events_written'] == session['events_total']
    assert sse_events and (online / 'frame.jpg').is_file() and not errors, report['online']
    assert len(videos) >= 4 and all(p.returncode == 0 for _,p,_ in consumers), report['online']
    assert report['online']['health_http_200'] > 0, report['online']
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
