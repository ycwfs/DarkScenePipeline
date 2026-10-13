"""Measure the pipeline against the spec: offline >= 15 fps (floor 10), real-time <= 1 s.

Both numbers are taken from the shipped entry point, not from a reimplementation, so what is
reported is what a user gets:

  offline   `darkpipe --mode offline` -> its own `[done] N frames in Ts = F fps` line, which
            is timed after model load, i.e. steady-state throughput.
  realtime  `darkpipe --mode serve` -> `/health.latency_ms_last`, which the server measures
            from the capture timestamp to after JPEG encode -- the full capture-to-output
            path, the thing the 1 s budget is actually about.

Resolution is a parameter because it dominates: the README's numbers are 320x240, while a
deployed camera is 720p+. `--res` re-encodes the source clip so a run is reproducible.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DP = os.path.normpath(os.path.join(HERE, "..", ".."))
PY = os.path.join(DP, ".venv", "bin", "python")
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))
BENCH_DIR = os.path.normpath(os.path.join(HERE, "..", "behavior_data", "bench"))

# (enhance, sr, recognize) -- the combinations worth shipping
COMBOS = [
    ("retinexformer", "off", "off"),
    ("retinexformer", "off", "r3d"),
    ("retinexformer", "off", "videomamba"),
    ("retinexformer", "off", "behavior"),
    ("retinexformer", "off", "xclip"),
    ("cidnet", "off", "behavior"),
    ("retinexformer", "bicubic_x2", "off"),
    ("retinexformer", "bicubic_x2", "behavior"),
    ("cidnet", "bicubic_x2", "behavior"),
    ("retinexformer", "lightsr_x2", "off"),
    ("retinexformer", "lightsr_x2", "behavior"),
    ("cidnet", "lightsr_x2", "behavior"),
    ("retinexformer", "catanet_x2", "off"),
    ("retinexformer", "catanet_x2", "behavior"),
    ("cidnet", "catanet_x2", "behavior"),
]
FPS_TARGET, FPS_FLOOR, LATENCY_BUDGET_S = 15.0, 10.0, 1.0


def make_clip(src, res, n_frames, out):
    """Re-encode `src` to a fixed resolution / length so every combo sees identical input."""
    import cv2
    if os.path.exists(out):
        return out
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = res
    os.makedirs(os.path.dirname(out), exist_ok=True)
    wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    written = 0
    while written < n_frames:
        ok, f = cap.read()
        if not ok:                       # loop the source until we have enough frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, f = cap.read()
            if not ok:
                break
        wr.write(cv2.resize(f, (w, h), interpolation=cv2.INTER_CUBIC))
        written += 1
    cap.release()
    wr.release()
    print(f"[bench] wrote {written} frames @ {w}x{h} -> {out}")
    return out


def run_offline(clip, enh, sr, reco, device, sr_chunk, extra=()):
    out = os.path.join(BENCH_DIR, "out.mp4")
    cmd = [PY, "-m", "darkpipe.cli", "--mode", "offline", "--input", clip, "--output", out,
           "--enhance", enh, "--sr", sr, "--recognize", reco, "--device", device,
           "--ckpt-dir", os.path.join(DP, "ckpts"), "--sr-chunk", str(sr_chunk), *extra]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=DP, capture_output=True, text=True)
    if p.returncode != 0:
        return None, (p.stderr or p.stdout).strip().splitlines()[-1][:160]
    m = re.search(r"= ([\d.]+) fps", p.stdout)
    return (float(m.group(1)) if m else None), f"wall {time.time()-t0:.0f}s"


def run_serve(clip, enh, sr, reco, device, port, warmup=25.0, sample=10.0):
    """Start the server on the clip, let it reach steady state, then read /health."""
    cmd = [PY, "-m", "darkpipe.cli", "--mode", "serve", "--input", clip, "--enhance", enh,
           "--sr", sr, "--recognize", reco, "--device", device, "--port", str(port),
           "--ckpt-dir", os.path.join(DP, "ckpts")]
    p = subprocess.Popen(cmd, cwd=DP, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}/health"
    try:
        deadline = time.time() + warmup + 60
        health = None
        while time.time() < deadline:
            time.sleep(2.0)
            try:
                health = json.load(urllib.request.urlopen(url, timeout=5))
            except Exception:
                continue
            if health.get("frames_proc", 0) > 40:      # past load + buffer fill
                break
        if not health:
            return None
        time.sleep(sample)
        samples = []
        for _ in range(5):
            samples.append(json.load(urllib.request.urlopen(url, timeout=5)))
            time.sleep(1.0)
        lat = sorted(s["latency_ms_last"] for s in samples)
        return dict(latency_ms_med=lat[len(lat) // 2], latency_ms_max=lat[-1],
                    fps_proc=samples[-1]["fps_proc"], fps_in=samples[-1]["fps_in"],
                    dropped=samples[-1].get("frames_dropped"))
    finally:
        p.terminate()
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.normpath(
        os.path.join(HERE, "..", "arid", "test_clips", "Drink", "Drink_11_1.mp4")))
    ap.add_argument("--res", default="640x480", help="WxH, or comma-separated list")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sr-chunk", type=int, default=8)
    ap.add_argument("--port", type=int, default=8811)
    ap.add_argument("--serve", action="store_true", help="also measure serve latency")
    ap.add_argument("--only", default="",
                    help="comma-separated substring filters on 'enh+sr+reco'; a combo runs if "
                         "ANY matches. Several filters in one invocation keep the rows "
                         "comparable — machine load moves these numbers by 2-3x, so a "
                         "reference row measured in a different run is not a reference")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    rows = []
    only = [s.strip() for s in a.only.split(",") if s.strip()]
    for res_s in a.res.split(","):
        w, h = (int(v) for v in res_s.lower().split("x"))
        clip = make_clip(a.src, (w, h), a.frames,
                         os.path.join(BENCH_DIR, f"bench_{w}x{h}_{a.frames}.mp4"))
        for enh, sr, reco in COMBOS:
            name = f"{enh}+{sr}+{reco}"
            if only and not any(f in name for f in only):
                continue
            fps, note = run_offline(clip, enh, sr, reco, a.device, a.sr_chunk)
            r = dict(res=f"{w}x{h}", enhance=enh, sr=sr, recognize=reco,
                     offline_fps=fps, note=note, sr_chunk=a.sr_chunk)
            if a.serve:
                r["serve"] = run_serve(clip, enh, sr, reco, a.device, a.port)
            rows.append(r)
            lat = (r.get("serve") or {}).get("latency_ms_med")
            print(f"  {w}x{h:<4} {name:<44} "
                  f"{'FAIL ' + note if fps is None else f'{fps:7.2f} fps'}"
                  f"{'' if lat is None else f'   latency {lat:7.1f} ms'}", flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"behavior_perf{a.tag}.json")
    json.dump(rows, open(out, "w"), indent=1)

    print(f"\n  {'config':<56}{'fps':>9} {'>=15':>5}{'>=10':>6}   realtime latency")
    for r in rows:
        f, lat = r["offline_fps"], (r.get("serve") or {}).get("latency_ms_med")
        cfg = f"{r['res']} {r['enhance']}+{r['sr']}+{r['recognize']}"
        fps_s = "      n/a" if f is None else f"{f:9.2f}"
        t15 = "ok" if f and f >= FPS_TARGET else "--"
        t10 = "ok" if f and f >= FPS_FLOOR else "--"
        lat_s = "" if lat is None else \
            f"   {lat:7.0f} ms {'ok' if lat <= LATENCY_BUDGET_S * 1000 else 'OVER'}"
        print(f"  {cfg:<56}{fps_s}{t15:>6}{t10:>6}{lat_s}")
    print(f"\n[bench] -> {out}")


if __name__ == "__main__":
    main()
