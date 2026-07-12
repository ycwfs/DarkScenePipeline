"""Build RealRestorer components from an HF-layout bundle (transformer/vae/text_encoder/
processor). Ported from ComfyUI-RealRestorer nodes.py load()."""
import torch

from .components import AutoEncoder
from .dit import Step1XEdit, Step1XParams
from .weight_loader import (detect_transformer_config, load_transformer_weights,
                            load_vae_weights, validate_bundle_path)


def build_components(bundle_dir: str, dtype=torch.bfloat16):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    paths = validate_bundle_path(bundle_dir)
    cfg = detect_transformer_config(paths["transformer_dir"])
    params = Step1XParams(
        in_channels=cfg["in_channels"], out_channels=cfg["out_channels"],
        vec_in_dim=cfg["vec_in_dim"], context_in_dim=cfg["context_in_dim"],
        hidden_size=cfg["hidden_size"], mlp_ratio=cfg["mlp_ratio"],
        num_heads=cfg["num_heads"], depth=cfg["depth"],
        depth_single_blocks=cfg["depth_single_blocks"], axes_dim=cfg["axes_dims_rope"],
        theta=cfg["theta"], qkv_bias=cfg["qkv_bias"], mode="torch",
        version=cfg["version"], guidance_embed=cfg["guidance_embeds"],
        use_mask_token=cfg["use_mask_token"],
    )
    with torch.device("meta"):
        transformer = Step1XEdit(params)
    load_transformer_weights(transformer, paths["transformer_dir"])
    transformer = transformer.to(dtype=dtype)
    transformer.eval(); transformer.requires_grad_(False)

    with torch.device("meta"):
        vae = AutoEncoder()
    load_vae_weights(vae, paths["vae_dir"])
    vae = vae.to(dtype=torch.float32)
    vae.eval(); vae.requires_grad_(False)

    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        paths["text_encoder_dir"], torch_dtype=dtype, attn_implementation="sdpa",
        local_files_only=True)
    text_encoder.eval(); text_encoder.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(
        paths["processor_dir"], min_pixels=256 * 28 * 28, max_pixels=324 * 28 * 28,
        local_files_only=True)
    return transformer, vae, text_encoder, processor, cfg["version"]
