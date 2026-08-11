"""PipelineConfig + validation rules (every toggle combination is legal except where noted)."""
import os
import sys
from dataclasses import dataclass, field

from .constants import (CKPT_FILES, EXPECTED_FPS, RECO_CFG, SR_ALIASES, SR_SCALES,
                        sr_ckpt_file)


@dataclass
class PipelineConfig:
    mode: str = "offline"
    input: str = ""
    output: str = ""
    enhance: str = "retinexformer"   # off | retinexformer | cidnet | realrestorer
    sr: str = "off"                  # off | bicubic | lightsr | catanet (+ legacy _x2 names)
    sr_scale: int | None = None      # 2 | 3 | 4; None -> 2 (pinned to 2 by an _x2 alias)
    recognize: str = "videomamba"    # off | r3d | videomamba | behavior | xclip
    device: str = "cuda:0"
    ckpt_dir: str = "./ckpts"
    # tuning
    enhance_chunk: int = 32
    sr_chunk: int | None = None      # None -> per-backend default (see validate); halves on OOM
    gpus: str = ""                   # comma-separated ids: shard offline work across GPUs
    start_frame: int = 0             # offline: skip this many frames (used by the sharder)
    sr_fp32: bool = False
    reco_stride: int | None = None
    reco_span_sec: float | None = None
    reco_ckpt: str = ""              # override the r3d/videomamba/behavior weights
    labels: str = ""                 # comma-separated, xclip only (open vocabulary)
    xclip_model: str = ""            # override the X-CLIP snapshot directory
    xclip_reject_tau: float | None = None
    no_label_bar: bool = False
    events_json: str = ""
    max_frames: int | None = None
    # serve
    host: str = "0.0.0.0"
    port: int = 8000
    jpeg_quality: int = 85
    max_stream_fps: float = 15.0
    record: str = ""
    # serve: per-event clip recording (see darkpipe/clips.py). Off unless clip_dir is set.
    clip_dir: str = ""
    clip_pre_sec: float = 2.0
    clip_post_sec: float = 2.0
    clip_max_sec: float = 30.0
    clip_skip_labels: str = "other"
    clip_min_conf: float = 0.0
    # Caller-supplied name for this run's clip subdirectory. Left empty the recorder invents
    # one, which is fine standalone but drifts by a second from a name the caller generated
    # separately -- and then the reported paths and the uploaded copies disagree.
    clip_session: str = ""
    # serve: which live output formats to expose. mjpeg is native; flv/hls need ffmpeg.
    stream_formats: str = "mjpeg"
    hls_dir: str = ""                # empty -> a temp dir under /tmp
    rtmp_push_url: str = ""          # push to someone else's rtmp:// / rtsp:// server
    max_flv_clients: int = 4
    # Bitrate cap for every H.264 output. Empty = uncapped, which on noisy enhanced footage
    # measured 44 Mbit/s at 1080p and 179 Mbit/s at 4K.
    stream_bitrate: str = "4M"
    # realrestorer
    rr_bundle: str = ""
    rr_steps: int = 28
    rr_cfg_scale: float = 3.0
    rr_chunk: int = 8
    rr_prompt: str = ""
    warnings: list = field(default_factory=list)
    # Filled by run_offline / run_offline_sharded: frames, seconds, fps of the PROCESSING
    # loop (model loading excluded). Callers that need throughput -- the platform operator
    # writes it into summary_json -- read it here instead of re-timing around the call and
    # charging one-off startup to the frame rate.
    stats: dict = field(default_factory=dict)

    def label_list(self):
        """Open-vocabulary labels for --recognize xclip; None -> the default BEHAVIORS."""
        v = [s.strip() for s in self.labels.split(",") if s.strip()]
        return v or None

    def sr_name(self):
        """Display/lookup key: "off" or "<backend>_x<scale>" — the old CLI spelling."""
        return "off" if self.sr == "off" else f"{self.sr}_x{self.sr_scale}"


def _die(msg):
    sys.exit(f"error: {msg}")


def validate(cfg: PipelineConfig) -> PipelineConfig:
    if not cfg.input:
        _die("--input is required")

    # --sr normalization first: everything below reasons about (backend, scale), not the
    # legacy `<backend>_x2` spellings.
    if cfg.sr in SR_ALIASES:
        backend, pinned = SR_ALIASES[cfg.sr]
        if cfg.sr_scale not in (None, pinned):
            _die(f"--sr {cfg.sr} pins x{pinned}; for another factor use "
                 f"--sr {backend} --sr-scale {cfg.sr_scale}")
        cfg.sr, cfg.sr_scale = backend, pinned
    elif cfg.sr == "off":
        if cfg.sr_scale is not None:
            cfg.warnings.append(f"--sr-scale {cfg.sr_scale} ignored: --sr is off")
        cfg.sr_scale = None
    else:
        cfg.sr_scale = 2 if cfg.sr_scale is None else cfg.sr_scale
        if cfg.sr_scale not in SR_SCALES:
            _die(f"--sr-scale must be one of {SR_SCALES} (got {cfg.sr_scale})")

    if cfg.enhance == "realrestorer" and cfg.mode == "serve":
        _die("RealRestorer runs at ~45 s/frame (batched diffusion, sequential offload) and is "
             "offline-only. Use --enhance retinexformer for serve mode.")

    if cfg.device.startswith("cpu"):
        if cfg.sr in ("lightsr", "catanet") \
                or cfg.recognize in ("videomamba", "behavior", "xclip") \
                or cfg.enhance == "realrestorer":
            _die("lightSR / CATANet / VideoMamba / X-CLIP / RealRestorer need CUDA (mamba-ssm "
                 "kernels, fp16 autocast or diffusion offload). Only --enhance "
                 "{retinexformer,cidnet} --sr {off,bicubic} --recognize {off,r3d} can run "
                 "on CPU (slowly).")

    if cfg.gpus:
        ids = [g.strip() for g in cfg.gpus.split(",") if g.strip()]
        if not all(g.isdigit() for g in ids):
            _die(f"--gpus takes comma-separated device ids, e.g. '0,1,2,3' (got {cfg.gpus!r})")
        if len(set(ids)) != len(ids):
            _die(f"--gpus has repeated ids ({cfg.gpus!r}); each segment needs its own GPU")
        if cfg.mode != "offline":
            _die("--gpus splits a file by frame range and is offline-only; a live stream has "
                 "no future frames to shard. Use --device to pick one GPU for serve mode.")
        if cfg.enhance == "realrestorer":
            _die("--gpus cannot shard RealRestorer: it restores the whole video in one pass "
                 "(a whole_video stage), so there are no independent segments.")

    if cfg.enhance == "off" and cfg.sr == "off" and cfg.recognize == "off":
        cfg.warnings.append("all functions disabled -> passthrough copy")

    # Per-backend chunk default. lightSR's sweet spot is 4 (11.9 GiB at 640x480); CATANet
    # needs ~10.3 GiB for a SINGLE frame at the same size, so 4 would OOM before the
    # halving loop could help. An explicit --sr-chunk always wins. (bicubic ignores it: it
    # is a CPU cv2.resize with no batch dimension.)
    if cfg.sr_chunk is None:
        cfg.sr_chunk = 1 if cfg.sr == "catanet" else 4
    elif cfg.sr_chunk < 1:
        _die(f"--sr-chunk must be >= 1 (got {cfg.sr_chunk})")

    if cfg.sr in ("lightsr", "catanet"):
        rate = {"lightsr": "~1.2 fps at 640x480 on an RTX 3090 (738 ms/frame at chunk 4), "
                           "with 1082-1123 ms serve latency — over the 1 s real-time "
                           "budget, so prefer --sr catanet in serve mode",
                "catanet": "~1.1 fps at 640x480 on an RTX 3090 (828 ms/frame at chunk 1), "
                           "with 825-903 ms serve latency — inside the 1 s budget"}[cfg.sr]
        scale_note = "" if cfg.sr_scale == 2 else (
            f" Those are the x2 numbers; x{cfg.sr_scale} was not benchmarked and produces "
            f"{cfg.sr_scale ** 2 / 4:.2f}x the output pixels, so expect it to be slower.")
        cfg.warnings.append(f"--sr {cfg.sr} measured {rate}. Offline that is below the 10 fps "
                            "floor: SR is a quality option for short clips, and every other "
                            f"configuration meets the spec without it.{scale_note}")

    # per-stage checkpoint existence
    need = []
    if cfg.enhance == "retinexformer":
        need.append(CKPT_FILES["retinexformer"])
    if cfg.enhance == "cidnet":
        need.append(CKPT_FILES["cidnet"])
    if cfg.enhance == "realrestorer":
        bundle = cfg.rr_bundle or os.path.join(cfg.ckpt_dir, CKPT_FILES["realrestorer"])
        if not os.path.isdir(os.path.join(bundle, "transformer")):
            _die(f"RealRestorer bundle not found at {bundle} — see README 'Checkpoint "
                 f"preparation' or run scripts/download_ckpts.sh")
        cfg.rr_bundle = bundle
    if cfg.sr in ("lightsr", "catanet"):     # bicubic has no weights, at any scale
        need.append(sr_ckpt_file(cfg.sr, cfg.sr_scale))
    if cfg.recognize in ("r3d", "videomamba", "behavior"):
        if cfg.reco_ckpt:
            if not os.path.exists(cfg.reco_ckpt):
                _die(f"--reco-ckpt {cfg.reco_ckpt} not found")
        else:
            need.append(CKPT_FILES[cfg.recognize])
    elif cfg.reco_ckpt:
        cfg.warnings.append(f"--reco-ckpt ignored for recognize={cfg.recognize} "
                            f"(use --xclip-model for X-CLIP snapshots)")
    for f in need:
        p = os.path.join(cfg.ckpt_dir, f)
        if not os.path.exists(p):
            _die(f"missing checkpoint {p} — see README 'Checkpoint preparation' or run "
                 f"scripts/download_ckpts.sh")

    if cfg.recognize == "xclip":
        snap = cfg.xclip_model or os.path.join(cfg.ckpt_dir, CKPT_FILES["xclip"])
        if not any(os.path.exists(os.path.join(snap, w))
                   for w in ("pytorch_model.bin", "model.safetensors")):
            _die(f"X-CLIP weights not found in {snap} — see README 'Checkpoint preparation' "
                 f"or run scripts/download_ckpts.sh")
        cfg.xclip_model = snap

    if cfg.labels and cfg.recognize != "xclip":
        cfg.warnings.append(f"--labels ignored: only --recognize xclip is open-vocabulary "
                            f"(recognize={cfg.recognize} has a fixed trained head)")

    if cfg.recognize != "off":
        window = RECO_CFG[cfg.recognize]["T"]
        if cfg.reco_stride is not None and cfg.reco_stride > window:
            _die(f"--reco-stride {cfg.reco_stride} > recognition window {window}")
        if cfg.reco_span_sec is not None and cfg.reco_span_sec <= 0:
            _die("--reco-span-sec must be > 0")
        # Offline processes every frame, so the last T frames already span T/fps ~ 1 s of
        # video. Serve mode drops frames to keep up, so without a span cap the window would
        # describe the last T/pipeline_fps seconds (~4-5 s) — past the 1 s freshness budget.
        if cfg.mode == "serve" and cfg.reco_span_sec is None:
            cfg.reco_span_sec = 1.0
            cfg.warnings.append("--reco-span-sec defaulted to 1.0 s in serve mode "
                                "(recognition window resampled from the last 1 s of stream)")

    if cfg.mode == "offline":
        if not cfg.output:
            stem, _ = os.path.splitext(os.path.basename(str(cfg.input)))
            cfg.output = f"{stem}_out.mp4"
        # Keyed by the x2 spellings: those are the configurations that were benchmarked, so
        # x3/x4 simply gets no prediction rather than a wrong one.
        exp = EXPECTED_FPS.get((cfg.enhance, cfg.sr_name()))
        if exp:
            # Resolution scaling differs by configuration: with a neural SR backend, cost
            # tracks pixel count almost exactly (measured 1.1 -> 5.1 fps going 640x480 ->
            # 320x240, a 4x pixel drop); otherwise fixed per-call overhead flattens it
            # (one paired 150-frame run: 22.2 -> 47.0 at 320x240, 8.1 at 720p, and bicubic
            # tracks it within 2% at every resolution).
            lo, hi = (4.5, 0.3) if cfg.sr in ("lightsr", "catanet") else (2.1, 0.36)
            cfg.warnings.append(f"expected throughput ~{exp:.1f} fps on a single RTX 3090 at "
                                f"640x480 for enhance={cfg.enhance} sr={cfg.sr_name()}; roughly "
                                f"{exp * lo:.0f} fps at 320x240 and {exp * hi:.1f} fps at "
                                f"1280x720 (see README)")
    else:
        if cfg.output:
            cfg.warnings.append("--output is ignored in serve mode (use --record)")
        if cfg.events_json:
            cfg.warnings.append("--events-json is ignored in serve mode (use /events SSE)")

    # Clip recording is driven by recognition events, so it is only ever a serve-mode
    # feature and only ever does anything with a recognizer attached. Saying so here beats
    # letting the user find an empty clip directory an hour into a run.
    if cfg.clip_dir:
        if cfg.mode != "serve":
            _die("--clip-dir cuts clips out of a live stream on recognition events and is "
                 "serve-only; offline already writes the whole processed video to --output.")
        if cfg.recognize == "off":
            _die("--clip-dir needs recognition events to know what to cut; "
                 "--recognize off would never write a clip.")
        for name, v in (("--clip-pre-sec", cfg.clip_pre_sec),
                        ("--clip-post-sec", cfg.clip_post_sec)):
            if v < 0:
                _die(f"{name} must be >= 0 (got {v})")
        if cfg.clip_max_sec <= 0:
            _die(f"--clip-max-sec must be > 0 (got {cfg.clip_max_sec})")
        if cfg.clip_max_sec < cfg.clip_pre_sec + cfg.clip_post_sec:
            cfg.warnings.append(
                f"--clip-max-sec {cfg.clip_max_sec}s is shorter than pre+post "
                f"({cfg.clip_pre_sec}+{cfg.clip_post_sec}s), so every clip is cut off at the "
                f"cap and the post-roll never lands")
        if not 0.0 <= cfg.clip_min_conf <= 1.0:
            _die(f"--clip-min-conf is a probability in [0,1] (got {cfg.clip_min_conf})")

    # Everything except mjpeg is muxed by an ffmpeg subprocess. Checking for it here turns
    # "the /live.flv endpoint quietly 503s" into a startup error naming the missing binary.
    if cfg.mode == "serve":
        from .streams import ffmpeg_path, parse_formats
        try:
            fmts = parse_formats(cfg.stream_formats)
        except ValueError as e:
            _die(str(e))
        needs_ffmpeg = [f for f in fmts if f != "mjpeg"] or ([] if not cfg.rtmp_push_url
                                                             else ["rtmp_push_url"])
        if needs_ffmpeg and not ffmpeg_path():
            _die(f"stream formats {needs_ffmpeg} need the ffmpeg binary, which is not in this "
                 f"image. Use --stream-formats mjpeg, or rebuild the image (platform/"
                 f"Dockerfile installs it).")

    if cfg.enhance != "realrestorer" and (cfg.rr_prompt or cfg.rr_bundle):
        cfg.warnings.append("--rr-* flags ignored (enhance != realrestorer)")

    return cfg
