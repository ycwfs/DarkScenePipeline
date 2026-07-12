"""Inference-only reimplementation of the VideoMamba fork's `mamba_inner_fn_no_out_proj`
on top of PyPI mamba-ssm's PUBLIC API (selective_scan_fn) + causal-conv1d's public
causal_conv1d_fn. The fork's private fused autograd Function calls old (pre-1.1) kernel
signatures that don't exist in modern binaries; this wrapper is numerically identical for
the forward pass (same silu-conv -> x_proj -> selective scan with z-gating), verified by
logits parity against the original fork environment.
"""
import torch.nn.functional as F
from causal_conv1d import causal_conv1d_fn
from einops import rearrange
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


def mamba_inner_fn_no_out_proj(xz, conv1d_weight, conv1d_bias, x_proj_weight,
                               delta_proj_weight, A, B=None, C=None, D=None,
                               delta_bias=None, B_proj_bias=None, C_proj_bias=None,
                               delta_softplus=True):
    assert B is None and C is None and B_proj_bias is None and C_proj_bias is None, \
        "only the variable-B/C path used by VideoMamba is implemented"
    L = xz.shape[-1]
    dt_rank = delta_proj_weight.shape[1]
    d_state = A.shape[-1]
    x, z = xz.chunk(2, dim=1)
    x = causal_conv1d_fn(x, rearrange(conv1d_weight, "d 1 w -> d w"), conv1d_bias,
                         activation="silu")
    x_dbl = F.linear(rearrange(x, "b d l -> (b l) d"), x_proj_weight)
    delta = rearrange(delta_proj_weight @ x_dbl[:, :dt_rank].t(), "d (b l) -> b d l", l=L)
    Bv = rearrange(x_dbl[:, dt_rank:dt_rank + d_state], "(b l) n -> b n l", l=L).contiguous()
    Cv = rearrange(x_dbl[:, dt_rank + d_state:], "(b l) n -> b n l", l=L).contiguous()
    return selective_scan_fn(x, delta, A, Bv, Cv, D, z=z, delta_bias=delta_bias,
                             delta_softplus=delta_softplus)
