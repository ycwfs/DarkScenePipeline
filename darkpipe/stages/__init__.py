"""Stage factory: the single place CLI enums map to classes. Imports are lazy so the
RealRestorer stack (transformers + 39 GiB bundle) is only touched when selected."""


def build_stages(cfg):
    """Returns (frame_stages: list[FrameStage], recognizer: Recognizer | None)."""
    frame_stages = []
    if cfg.enhance == "retinexformer":
        from .enhance_retinexformer import RetinexformerStage
        frame_stages.append(RetinexformerStage(cfg.ckpt_dir, chunk=cfg.enhance_chunk))
    elif cfg.enhance == "cidnet":
        from .enhance_cidnet import CIDNetStage
        frame_stages.append(CIDNetStage(cfg.ckpt_dir, chunk=cfg.enhance_chunk))
    elif cfg.enhance == "realrestorer":
        from .enhance_realrestorer import RealRestorerStage
        frame_stages.append(RealRestorerStage(
            cfg.ckpt_dir, bundle=cfg.rr_bundle, steps=cfg.rr_steps,
            cfg_scale=cfg.rr_cfg_scale, chunk=cfg.rr_chunk, prompt=cfg.rr_prompt))

    # cfg.sr is the normalized backend name and cfg.sr_scale the factor (config.validate
    # resolves the legacy `<backend>_x2` spellings into that pair).
    if cfg.sr == "bicubic":
        from .sr_bicubic import BicubicStage
        frame_stages.append(BicubicStage(scale=cfg.sr_scale))
    elif cfg.sr == "lightsr":
        from .sr_lightsr import LightSRStage
        frame_stages.append(LightSRStage(cfg.ckpt_dir, chunk=cfg.sr_chunk,
                                         force_fp32=cfg.sr_fp32, scale=cfg.sr_scale))
    elif cfg.sr == "catanet":
        from .sr_catanet import CATANetStage
        # CATANet needs ~10 GiB for one 640x480 frame; the shared sr_chunk default of 4 would
        # OOM immediately, so unless the user asked for a specific chunk it starts at 1.
        frame_stages.append(CATANetStage(cfg.ckpt_dir, chunk=cfg.sr_chunk,
                                         force_fp32=cfg.sr_fp32, scale=cfg.sr_scale))

    recognizer = None
    kw = dict(stride=cfg.reco_stride, span_sec=cfg.reco_span_sec)
    if cfg.recognize == "r3d":
        from .recognize import R3DRecognizer
        recognizer = R3DRecognizer(cfg.ckpt_dir, **kw)
    elif cfg.recognize == "videomamba":
        from .recognize import VideoMambaRecognizer
        recognizer = VideoMambaRecognizer(cfg.ckpt_dir, **kw)
    elif cfg.recognize == "behavior":
        from .recognize import BehaviorRecognizer
        recognizer = BehaviorRecognizer(cfg.ckpt_dir, **kw)
    if recognizer is not None and cfg.reco_ckpt:
        recognizer.ckpt = cfg.reco_ckpt        # user-supplied head (validated in config.py)
    if cfg.recognize == "xclip":
        from ..constants import XCLIP_REJECT_TAU
        from .recognize_xclip import XCLIPRecognizer
        tau = XCLIP_REJECT_TAU if cfg.xclip_reject_tau is None else cfg.xclip_reject_tau
        recognizer = XCLIPRecognizer(cfg.ckpt_dir, labels=cfg.label_list(),
                                     model_dir=cfg.xclip_model, reject_tau=tau, **kw)

    return frame_stages, recognizer
