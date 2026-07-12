"""PipelineConfig + validation rules (every toggle combination is legal except where noted)."""
import os
import sys
from dataclasses import dataclass, field

from .constants import CKPT_FILES, EXPECTED_FPS, RECO_CFG


@dataclass
class PipelineConfig:
    mode: str = "offline"
    input: str = ""
    output: str = ""
    enhance: str = "retinexformer"   # off | retinexformer | realrestorer
    sr: str = "lightsr_x2"           # off | lightsr_x2
    recognize: str = "videomamba"    # off | r3d | videomamba
    device: str = "cuda:0"
    ckpt_dir: str = "./ckpts"
    # tuning
    enhance_chunk: int = 32
    sr_chunk: int = 8
    sr_fp32: bool = False
    reco_stride: int | None = None
    no_label_bar: bool = False
    events_json: str = ""
    max_frames: int | None = None
    # serve
    host: str = "0.0.0.0"
    port: int = 8000
    jpeg_quality: int = 85
    max_stream_fps: float = 15.0
    record: str = ""
    # realrestorer
    rr_bundle: str = ""
    rr_steps: int = 28
    rr_cfg_scale: float = 3.0
    rr_chunk: int = 8
    rr_prompt: str = ""
    warnings: list = field(default_factory=list)


def _die(msg):
    sys.exit(f"error: {msg}")


def validate(cfg: PipelineConfig) -> PipelineConfig:
    if not cfg.input:
        _die("--input is required")

    if cfg.enhance == "realrestorer" and cfg.mode == "serve":
        _die("RealRestorer runs at ~45 s/frame (batched diffusion, sequential offload) and is "
             "offline-only. Use --enhance retinexformer for serve mode.")

    if cfg.device.startswith("cpu"):
        if cfg.sr != "off" or cfg.recognize == "videomamba" or cfg.enhance == "realrestorer":
            _die("lightSR / VideoMamba / RealRestorer need CUDA (mamba-ssm kernels or "
                 "diffusion offload). Only --enhance retinexformer --sr off --recognize "
                 "{off,r3d} can run on CPU (slowly).")

    if cfg.enhance == "off" and cfg.sr == "off" and cfg.recognize == "off":
        cfg.warnings.append("all functions disabled -> passthrough copy")

    # per-stage checkpoint existence
    need = []
    if cfg.enhance == "retinexformer":
        need.append(CKPT_FILES["retinexformer"])
    if cfg.enhance == "realrestorer":
        bundle = cfg.rr_bundle or os.path.join(cfg.ckpt_dir, CKPT_FILES["realrestorer"])
        if not os.path.isdir(os.path.join(bundle, "transformer")):
            _die(f"RealRestorer bundle not found at {bundle} — see README 'Checkpoint "
                 f"preparation' or run scripts/download_ckpts.sh")
        cfg.rr_bundle = bundle
    if cfg.sr == "lightsr_x2":
        need.append(CKPT_FILES["lightsr_x2"])
    if cfg.recognize in ("r3d", "videomamba"):
        need.append(CKPT_FILES[cfg.recognize])
    for f in need:
        p = os.path.join(cfg.ckpt_dir, f)
        if not os.path.exists(p):
            _die(f"missing checkpoint {p} — see README 'Checkpoint preparation' or run "
                 f"scripts/download_ckpts.sh")

    if cfg.recognize != "off":
        window = RECO_CFG[cfg.recognize]["T"]
        if cfg.reco_stride is not None and cfg.reco_stride > window:
            _die(f"--reco-stride {cfg.reco_stride} > recognition window {window}")

    if cfg.mode == "offline":
        if not cfg.output:
            stem, _ = os.path.splitext(os.path.basename(str(cfg.input)))
            cfg.output = f"{stem}_out.mp4"
        exp = EXPECTED_FPS.get((cfg.enhance, cfg.sr))
        if exp:
            cfg.warnings.append(f"expected throughput ~{exp:.1f} fps on a single GPU "
                                f"for enhance={cfg.enhance} sr={cfg.sr} (see README)")
    else:
        if cfg.output:
            cfg.warnings.append("--output is ignored in serve mode (use --record)")
        if cfg.events_json:
            cfg.warnings.append("--events-json is ignored in serve mode (use /events SSE)")

    if cfg.enhance != "realrestorer" and (cfg.rr_prompt or cfg.rr_bundle):
        cfg.warnings.append("--rr-* flags ignored (enhance != realrestorer)")

    return cfg
