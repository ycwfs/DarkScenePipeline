"""Batched sequential-offload RealRestorer restore on a single 24GB GPU.
Port of the validated ComfyUI-RealRestorer/batch_restore.py (45 s/frame measured).

4 phases, each with ONE GPU residency for its component:
  1. VAE-encode all frames  2. Qwen embeddings for all frames (cond+uncond)
  3. chunked CFG denoise with dynamic DiT-block preloading (rest streamed from CPU)
  4. VAE-decode all frames
"""
import math
import sys
import time

import numpy as np
import torch
from PIL import Image

from .pipeline import (_get_qwenvl_embeds, _move_transformer_nonblocks,
                       _offload_to_cpu, _offload_transformer_nonblocks, _pack_latents,
                       _pil_to_tensor, _prepare_img_ids, _process_diff_norm,
                       _resize_image, _tensor_to_pil, _unpack_latents,
                       decode_vae_latents, encode_vae_image)
from .scheduler import RealRestorerFlowMatchScheduler


@torch.inference_mode()
def restore_frames(components, frames_pil, prompt, steps=28, guidance=3.0,
                   model_guidance=3.5, chunk=8, size_level=512, seed=42, device="cuda"):
    """frames_pil: list of PIL RGB (uniform size). Returns list of PIL RGB (original size)."""
    tr, vae, te, proc, version = components
    dev = torch.device(device)
    N = len(frames_pil)
    pils, sizes = [], []
    for im in frames_pil:
        r, orig = _resize_image(im.convert("RGB"), img_size=size_level)
        pils.append(r); sizes.append(orig)
    W, H = pils[0].size
    assert all(p.size == (W, H) for p in pils), "frames must share one size"

    t0 = time.time(); vae.to(dev)
    ref = [_pack_latents(encode_vae_image(vae, _pil_to_tensor(p).unsqueeze(0).to(
        device=dev, dtype=torch.float32)).to(dtype=torch.bfloat16)) for p in pils]
    ref_all = torch.cat(ref, dim=0)
    _offload_to_cpu(vae)
    print(f"[rr] vae-enc {time.time()-t0:.1f}s", flush=True)

    t0 = time.time(); te.to(dev)
    embs, masks = _get_qwenvl_embeds(
        te, proc, prompts=[prompt] * N + [""] * N, ref_images=pils + pils,
        edit_types=[1] * (2 * N), device=dev, dtype=next(te.parameters()).dtype,
        max_token_length=640)
    _offload_to_cpu(te)
    print(f"[rr] qwen {time.time()-t0:.1f}s ({(time.time()-t0)/N:.1f}s/frame)", flush=True)

    gen = torch.Generator(device=dev).manual_seed(seed)
    lat_ch = getattr(vae, "latent_channels", 16)
    noise = torch.randn(N, lat_ch, H // 8, W // 8, generator=gen, device=dev,
                        dtype=torch.bfloat16)
    lat_all = _pack_latents(noise); del noise

    _move_transformer_nonblocks(tr, dev)
    blocks = list(tr.double_blocks) + list(tr.single_blocks)
    free = torch.cuda.mem_get_info(dev.index or 0)[0]
    margin = int((4.0 + 0.20 * 2 * chunk) * 1024 ** 3)
    bbytes = sum(p.numel() * p.element_size() for p in blocks[0].parameters())
    n_fit = min(len(blocks), max(0, free - margin) // bbytes)
    pre = blocks[:n_fit]
    for b in pre:
        b.to(dev)
    print(f"[rr] preloaded {len(pre)}/{len(blocks)} DiT blocks", flush=True)

    ph, pw = math.ceil(H / 16), math.ceil(W / 16)
    t_all = time.time()
    for c0 in range(0, N, chunk):
        c1 = min(N, c0 + chunk); B = c1 - c0
        lat, refc = lat_all[c0:c1], ref_all[c0:c1]
        pe = torch.cat([embs[c0:c1], embs[N + c0:N + c1]], dim=0)
        pm = torch.cat([masks[c0:c1], masks[N + c0:N + c1]], dim=0)
        txt_ids = torch.zeros(2 * B, pe.shape[1], 3, dtype=pe.dtype, device=dev)
        iid = _prepare_img_ids(2 * B, ph, pw, pe.dtype, dev, axis0=0.0)
        rid = _prepare_img_ids(2 * B, ph, pw, pe.dtype, dev,
                               axis0=(0.0 if version == "v1.0" else 1.0))
        cid = torch.cat([iid, rid], dim=1)
        sched = RealRestorerFlowMatchScheduler()
        sched.set_timesteps(num_inference_steps=steps, device=dev,
                            image_seq_len=lat.shape[1])
        ts = sched.timesteps.tolist()
        tc = time.time()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for t in ts[:-1]:
                minput = torch.cat([lat.repeat(2, 1, 1), refc.repeat(2, 1, 1)], dim=1)
                tv = torch.full((2 * B,), float(t), dtype=minput.dtype, device=dev)
                gv = torch.full((2 * B,), model_guidance, dtype=minput.dtype, device=dev)
                pred = tr(img=minput, img_ids=cid, txt_ids=txt_ids, timesteps=tv,
                          llm_embedding=pe, t_vec=tv, mask=pm, guidance=gv,
                          block_device=dev)
                pred = pred[:, : lat.shape[1]]
                cond, uncond = pred.chunk(2, dim=0)
                if float(t) > 0.93:
                    dn = torch.norm(cond - uncond, dim=2, keepdim=True)
                    pred = uncond + guidance * (cond - uncond) / _process_diff_norm(dn, k=0.4)
                else:
                    pred = uncond + guidance * (cond - uncond)
                lat = sched.step(pred, t, lat)
        lat_all[c0:c1] = lat
        print(f"[rr] chunk {c0 // chunk}: {B} frames {time.time()-tc:.1f}s", flush=True)
    for b in pre:
        b.to("cpu")
    _offload_transformer_nonblocks(tr)
    torch.cuda.empty_cache()
    print(f"[rr] denoise total {time.time()-t_all:.1f}s", flush=True)

    vae.to(dev)
    outs = []
    for i in range(N):
        dec = decode_vae_latents(vae, _unpack_latents(lat_all[i:i + 1].float(), H, W))
        dec = dec.clamp(-1, 1).mul(0.5).add(0.5)
        outs.append(_tensor_to_pil(dec[0].float()).resize(sizes[i]))
    _offload_to_cpu(vae)
    return outs
