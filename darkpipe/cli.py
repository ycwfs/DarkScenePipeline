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
    p.add_argument("--enhance", choices=["off", "retinexformer", "realrestorer"],
                   default="retinexformer")
    p.add_argument("--sr", choices=["off", "lightsr_x2"], default="lightsr_x2")
    p.add_argument("--recognize", choices=["off", "r3d", "videomamba"], default="videomamba")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ckpt-dir", default="./ckpts")
    t = p.add_argument_group("tuning")
    t.add_argument("--enhance-chunk", type=int, default=32)
    t.add_argument("--sr-chunk", type=int, default=8)
    t.add_argument("--sr-fp32", action="store_true")
    t.add_argument("--reco-stride", type=int, default=None,
                   help="frames between recognition updates (default: window/2)")
    t.add_argument("--no-label-bar", action="store_true")
    t.add_argument("--events-json", default="")
    t.add_argument("--max-frames", type=int, default=None)
    s = p.add_argument_group("serve")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--jpeg-quality", type=int, default=85)
    s.add_argument("--max-stream-fps", type=float, default=15.0)
    s.add_argument("--record", default="")
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
    print(f"[config] mode={cfg.mode} enhance={cfg.enhance} sr={cfg.sr} "
          f"recognize={cfg.recognize} device={cfg.device}")
    if cfg.mode == "offline":
        from .pipeline import run_offline
        run_offline(cfg)
    else:
        from .server import run_server
        run_server(cfg)


if __name__ == "__main__":
    main()
