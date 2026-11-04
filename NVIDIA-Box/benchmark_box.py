#!/usr/bin/env python3
"""Run the real Box entry and retain device, memory, session and video evidence."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time

from run import ROOT, runtime_environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True, help='本地视频或 RTSP/RTMP URL')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=300, help='实时流运行秒数；文件按完整视频运行')
    parser.add_argument('--backend', choices=['auto', 'cuda', 'tensorrt'], default='auto')
    parser.add_argument('--profile', choices=['balanced', 'reference'], default='balanced')
    args, extra = parser.parse_known_args()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    session = out/'session.json'
    command = [sys.executable, '-u', str(ROOT/'run.py'), '--video_path', args.video,
        '--output-dir', str(out/'outputs'), '--session_json', str(session),
        '--run_seconds', str(args.duration), '--box-backend', args.backend,
        '--box-profile', args.profile, *extra]
    evidence = dict(architecture=platform.machine(), python=sys.version, uname=list(platform.uname()),
        backend=args.backend, profile=args.profile,
        mode='live stream' if args.video.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')) else 'offline file',
        timing_note='File FPS is offline throughput. Stream FPS/age/drops are in session/operator log; this is not an external network receiver test.')
    for name in ('/etc/nv_tegra_release', '/etc/os-release'):
        if Path(name).is_file():
            evidence[name] = Path(name).read_text()
    samples = []
    tegra, tegra_log = None, None
    if shutil.which('tegrastats'):
        tegra_log = (out/'tegrastats.log').open('w')
        tegra = subprocess.Popen(['tegrastats', '--interval', '1000'], stdout=tegra_log, stderr=subprocess.STDOUT)
    started = time.monotonic()
    process = None
    try:
        with (out/'operator.log').open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=runtime_environment())
            while process.poll() is None:
                row = dict(seconds=time.monotonic()-started)
                try:
                    status = Path(f'/proc/{process.pid}/status').read_text().splitlines()
                    for line in status:
                        if line.startswith(('VmRSS:', 'VmHWM:')):
                            key, value = line.split(':', 1)
                            row[key+'_KiB'] = int(value.split()[0])
                except FileNotFoundError:
                    pass
                for line in Path('/proc/meminfo').read_text().splitlines():
                    key, value = line.split(':', 1)
                    if key in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
                        row[key+'_KiB'] = int(value.split()[0])
                samples.append(row)
                time.sleep(1)
        evidence.update(returncode=process.returncode, elapsed_seconds=time.monotonic()-started)
    finally:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=5)
        if tegra is not None:
            tegra.terminate()
            try:
                tegra.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tegra.kill(); tegra.wait(timeout=5)
            tegra_log.close()
    if session.is_file():
        evidence['session'] = json.loads(session.read_text())
    evidence['samples'] = samples
    (out/'benchmark.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n')
    print(f"结果：{out/'benchmark.json'}；退出码 {evidence['returncode']}")
    raise SystemExit(evidence['returncode'])


if __name__ == '__main__':
    main()
