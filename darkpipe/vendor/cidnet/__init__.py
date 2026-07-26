"""Vendored HVI-CIDNet (CVPR2025) low-light enhancement network.

Self-contained copy of Fediory/HVI-CIDNet `net/` (CIDNet + HVI_transform +
transformer_utils + LCA), with the huggingface_hub mixin removed so no external
repo or hub access is needed at runtime. Pure torch + einops.
"""
from .CIDNet import CIDNet

__all__ = ["CIDNet"]
