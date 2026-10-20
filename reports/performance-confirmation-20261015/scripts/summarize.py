"""Summarize archived runs using the same 20..120 s measurement window."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    results = {}
    for tag in ('single-default-r1-persistent-sse', 'single-fast-diagnostic-persistent-sse'):
        path = ROOT / tag
        original = json.loads((path / 'summary.json').read_text())
        rows = [json.loads(line) for line in (path / 'health.jsonl').read_text().splitlines()]
        events = json.loads((path / 'sse-events.json').read_text())
        active = [h for h in rows if 20 <= h['uptime_s'] <= 120 and h['fps_proc'] > 0]
        clock = float(np.median([(h['_received_wall'] + h['_request_wall']) / 2 - h['uptime_s'] for h in active]))
        ages = [(e['received_wall'] - clock - e['event']['timestamp']) * 1000
                for e in events if 20 <= e['event']['timestamp'] <= 120]
        session = json.loads((path / 'session.json').read_text())
        inspect = json.loads((path / 'container-inspect.json').read_text())[0]
        cfg = json.loads((path / 'resolved-config.json').read_text())
        assert inspect['HostConfig']['NanoCpus'] == 8_000_000_000
        assert inspect['HostConfig']['Memory'] == 12 * 1024**3
        assert inspect['HostConfig']['DeviceRequests'][0]['DeviceIDs'] == ['1']
        assert cfg['device'] == 'cuda:0' and cfg['gpus'] == '0'
        results[tag] = {
            'exit_code': original['exit_code'], 'session_status': session['status'],
            'measurement_window': [20, 120], 'health_samples': len(active),
            'recognition_samples': len(ages), 'processing_fps': original['fps_proc'],
            'recognition_delay_ms': original['recognition_event_age_ms'],
            'frame_latency_ms': original['frame_latency_ms'],
            'fps_ge_15_samples': sum(h['fps_proc'] >= 15 for h in active),
            'recognition_le_1000ms': sum(x <= 1000 for x in ages),
            'clips': session['clips'], 'wall_seconds': original['wall_seconds_to_exit'],
            'image_id': inspect['Image'], 'gpu_physical': [1], 'cpu_limit': 8,
            'memory_limit_gib': 12, 'clip_denoise': cfg['clip_denoise'],
            'skeleton_enabled': '骨架分支已启用' in (path / 'container.log').read_text(),
            'skeleton_disabled': '骨架分支关闭' in (path / 'container.log').read_text(),
            'oom_killed': inspect['State']['OOMKilled'],
        }
    (ROOT / 'comparison.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
