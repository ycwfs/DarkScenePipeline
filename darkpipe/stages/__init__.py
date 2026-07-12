"""Stage factory: the single place CLI enums map to classes. Imports are lazy so the
RealRestorer stack (transformers + 39 GiB bundle) is only touched when selected."""


def build_stages(cfg):
    """Returns (frame_stages: list[FrameStage], recognizer: Recognizer | None)."""
    frame_stages = []
    if cfg.enhance == "retinexformer":
        from .enhance_retinexformer import RetinexformerStage
        frame_stages.append(RetinexformerStage(cfg.ckpt_dir, chunk=cfg.enhance_chunk))
    elif cfg.enhance == "realrestorer":
        from .enhance_realrestorer import RealRestorerStage
        frame_stages.append(RealRestorerStage(
            cfg.ckpt_dir, bundle=cfg.rr_bundle, steps=cfg.rr_steps,
            cfg_scale=cfg.rr_cfg_scale, chunk=cfg.rr_chunk, prompt=cfg.rr_prompt))

    if cfg.sr == "lightsr_x2":
        from .sr_lightsr import LightSRStage
        frame_stages.append(LightSRStage(cfg.ckpt_dir, chunk=cfg.sr_chunk,
                                         force_fp32=cfg.sr_fp32))

    recognizer = None
    if cfg.recognize == "r3d":
        from .recognize import R3DRecognizer
        recognizer = R3DRecognizer(cfg.ckpt_dir, stride=cfg.reco_stride)
    elif cfg.recognize == "videomamba":
        from .recognize import VideoMambaRecognizer
        recognizer = VideoMambaRecognizer(cfg.ckpt_dir, stride=cfg.reco_stride)

    return frame_stages, recognizer
