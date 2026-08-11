"""Single entry point: `darkpipe` (or python main.py). Flow is fully parameter-driven."""
import argparse

from .config import PipelineConfig, validate


def build_parser():
    p = argparse.ArgumentParser(
        prog="darkpipe",
        description="Dark complex scene algorithm: low-light enhancement, super-resolution "
                    "and action recognition — each independently switchable; offline file "
                    "processing or online streaming-inference server.")
    p.add_argument("--mode", choices=["offline", "serve"], default="offline")
    p.add_argument("--input", required=True,
                   help="video file | rtsp:// | http(s):// | webcam index")
    p.add_argument("--output", default="", help="output video path (offline)")
    p.add_argument("--enhance", choices=["off", "retinexformer", "cidnet", "realrestorer"],
                   default="retinexformer")
    p.add_argument("--sr", choices=["off", "bicubic", "lightsr", "catanet",
                                    "bicubic_x2", "lightsr_x2", "catanet_x2"],
                   default="off",
                   help="super-resolution backend; the factor is --sr-scale (default 2). "
                        "The `_x2` spellings are the older names and pin x2. bicubic is the "
                        "only one that meets the "
                        "performance spec (~1-5 ms/frame) and it also scores highest on "
                        "in-domain PSNR/SSIM for 240p dark video — use it unless you "
                        "specifically want learned texture. lightsr / catanet run "
                        "~1.1-1.2 fps at 640x480 on an RTX 3090 and are offline quality "
                        "options for short clips; of the two, catanet (CVPR2025) has the "
                        "better quality and is the only neural one under 1 s serve latency, "
                        "while lightsr needs 3.2 GiB instead of ~10. See README "
                        "'Performance' and compare/results/SR_REPORT.md.")
    p.add_argument("--sr-scale", type=int, choices=[2, 3, 4], default=None,
                   help="super-resolution factor (default 2). bicubic needs no weights at any "
                        "scale; lightsr/catanet need the checkpoint trained for that factor "
                        "(mambairv2_lightSR_x<N>.pth / catanet_x<N>.pth — "
                        "scripts/download_ckpts.sh --sr-scale <N>)")
    p.add_argument("--recognize",
                   choices=["off", "r3d", "videomamba", "behavior", "xclip"],
                   default="videomamba",
                   help="r3d/videomamba: ARID-11 actions. behavior: trained 9-behavior head. "
                        "xclip: open-vocabulary, any --labels, no retraining")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gpus", default="",
                   help="offline only: comma-separated GPU ids to split the video across, "
                        "e.g. '0,1,2,3'. One segment per GPU, concatenated afterwards; the "
                        "recognition window restarts at each cut, so short clips are left "
                        "on a single GPU")
    p.add_argument("--ckpt-dir", default="./ckpts")
    t = p.add_argument_group("tuning")
    t.add_argument("--enhance-chunk", type=int, default=32)
    t.add_argument("--sr-chunk", type=int, default=None,
                   help="frames per SR batch (default: 4 for lightsr_x2, 1 for catanet_x2, "
                        "which needs ~10 GiB per 640x480 frame); halves itself on OOM")
    t.add_argument("--sr-fp32", action="store_true")
    t.add_argument("--reco-stride", type=int, default=None,
                   help="frames between recognition updates (default: window/2)")
    t.add_argument("--reco-span-sec", type=float, default=None,
                   help="cap the recognition window to the last N seconds, resampling its T "
                        "frames from them (default: off offline, 1.0 in serve mode)")
    t.add_argument("--reco-ckpt", default="",
                   help="weights for r3d/videomamba/behavior (default: <ckpt-dir>/ the file "
                        "in CKPT_FILES) — e.g. a behavior head trained for another enhancer")
    t.add_argument("--labels", default="",
                   help="comma-separated open-vocabulary labels for --recognize xclip "
                        "(default: the nine behaviors + other)")
    t.add_argument("--xclip-model", default="",
                   help="X-CLIP snapshot dir (default: <ckpt-dir>/xclip-base-patch16-zero-shot)")
    t.add_argument("--xclip-reject-tau", type=float, default=None,
                   help="report 'other' unless a named behavior reaches this probability "
                        "(default 0.4, calibrated on validation; 0 disables)")
    t.add_argument("--no-label-bar", action="store_true")
    t.add_argument("--events-json", default="")
    t.add_argument("--max-frames", type=int, default=None)
    t.add_argument("--start-frame", type=int, default=0,
                   help="offline: begin at this frame (set per segment by --gpus)")
    s = p.add_argument_group("serve")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--jpeg-quality", type=int, default=85)
    s.add_argument("--max-stream-fps", type=float, default=15.0)
    s.add_argument("--record", default="")
    s.add_argument("--clip-dir", default="",
                   help="serve: save one mp4 per recognised behavior under this directory "
                        "(empty = off). Clips are cut from the live stream around each "
                        "event; see --clip-skip-labels")
    s.add_argument("--clip-pre-sec", type=float, default=2.0,
                   help="seconds kept from BEFORE the trigger (the window that caused the "
                        "event is already past when it fires)")
    s.add_argument("--clip-post-sec", type=float, default=2.0,
                   help="a clip closes this long after its last qualifying event")
    s.add_argument("--clip-max-sec", type=float, default=30.0,
                   help="hard cap so a permanently active scene cannot grow one file forever")
    s.add_argument("--clip-skip-labels", default="other",
                   help="comma-separated labels that never start a clip (default: other)")
    s.add_argument("--clip-min-conf", type=float, default=0.0,
                   help="ignore events below this probability (0 = keep all)")
    s.add_argument("--stream-formats", default="mjpeg",
                   help="comma-separated live outputs: mjpeg (native), flv (/live.flv), "
                        "hls (/hls/index.m3u8). Anything but mjpeg needs ffmpeg in the image")
    s.add_argument("--hls-dir", default="",
                   help="where HLS segments are written (default: a temp dir)")
    s.add_argument("--rtmp-push-url", default="",
                   help="also push the processed stream to an external rtmp:// / rtsp:// server")
    s.add_argument("--stream-bitrate", default="4M",
                   help="cap for every H.264 output, e.g. 4M/8M; empty disables the cap "
                        "(uncapped measured 44 Mbit/s at 1080p on noisy enhanced footage)")
    s.add_argument("--max-flv-clients", type=int, default=4,
                   help="concurrent /live.flv viewers; each one costs its own encoder")
    r = p.add_argument_group("realrestorer")
    r.add_argument("--rr-bundle", default="")
    r.add_argument("--rr-steps", type=int, default=28)
    r.add_argument("--rr-cfg-scale", type=float, default=3.0)
    r.add_argument("--rr-chunk", type=int, default=8)
    r.add_argument("--rr-prompt", default="")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = validate(PipelineConfig(**vars(args)))
    for w in cfg.warnings:
        print(f"[warn] {w}")
    print(f"[config] mode={cfg.mode} enhance={cfg.enhance} sr={cfg.sr_name()} "
          f"recognize={cfg.recognize} device={cfg.device}")
    if cfg.mode == "offline":
        gpus = [g.strip() for g in cfg.gpus.split(",") if g.strip()]
        if len(gpus) > 1:
            from .shard import run_offline_sharded
            run_offline_sharded(cfg, gpus)
        else:
            from .pipeline import run_offline
            run_offline(cfg)
    else:
        from .server import run_server
        run_server(cfg)


if __name__ == "__main__":
    main()
