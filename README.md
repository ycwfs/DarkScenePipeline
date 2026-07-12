# DarkScenePipeline

The **dark complex scene algorithm** as one deployable project: low-illumination enhancement,
video super-resolution, and human action recognition — each function independently
enable/disable-able and freely combinable — with offline file processing and an online
streaming-inference server.

```
                     ┌───────────────────────────┐
 dark video ──────►  │ 1. low-light enhancement  │   off | retinexformer (NTIRE) | realrestorer
 (file/RTSP/webcam)  ├───────────────────────────┤
                     │ 2. super-resolution       │   off | lightsr_x2 (MambaIRv2)
                     ├───────────────────────────┤
                     │ 3. action recognition     │   off | r3d (R(2+1)D-18) | videomamba (VideoMamba-T/32f)
                     └───────────────────────────┘
                       │                    │
              offline: mp4 with the       serve: HTTP MJPEG stream +
              action text BELOW the       SSE recognition events
              frame + events JSON
```

Everything runs in **one uv-managed environment** (Python 3.10, torch 2.7/cu126, prebuilt
mamba CUDA kernels — no local compilation).

## Requirements
- Ubuntu Linux x86_64 with an NVIDIA GPU (driver ≥ 560, i.e. CUDA 12.6 runtime capable).
  ≥ 8 GB VRAM for the standard pipeline; **≥ 24 GB for `--enhance realrestorer`**.
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

## Environment setup
```bash
cd DarkScenePipeline
uv python install 3.10          # uv-managed interpreter (ships C headers triton needs)
UV_MANAGED_PYTHON=1 uv venv --python 3.10 .venv
UV_MANAGED_PYTHON=1 uv sync
# smoke test
.venv/bin/python -c "import torch, mamba_ssm; print(torch.cuda.is_available())"
```
Notes:
- torch 2.7.0 comes from default PyPI wheels (they bundle cu126; no custom index).
- `mamba-ssm` / `causal-conv1d` install from **pinned prebuilt release wheels**
  (`cu12torch2.7cxx11abiTRUE-cp310`, see `[tool.uv.sources]`) — no nvcc needed. If the
  GitHub URLs are slow, prefix them with `https://ghfast.top/`. If they ever disappear,
  fall back to a source build: install `cuda-toolkit` 12.x, then
  `uv pip install --no-build-isolation mamba-ssm causal-conv1d`.
- Use the **uv-managed** Python (as above). A system Python without the `python3.10-dev`
  headers will crash at import time when triton JIT-compiles its launcher.

## Checkpoint preparation
```bash
bash scripts/download_ckpts.sh     # or stage manually per the table
```
| file (in `ckpts/`) | size | function | provenance |
|---|---|---|---|
| `NTIRE.pth` | 6.2 MB | enhance: retinexformer | Retinexformer model zoo (NTIRE 2025 low-light weight) |
| `mambairv2_lightSR_x2.pth` | 3.9 MB | sr: lightsr_x2 | [MambaIR release v1.0](https://github.com/csguoh/MambaIR/releases/tag/v1.0) (use `https://ghfast.top/` prefix for speed) |
| `r2plus1d_arid.pth` | 120 MB | recognize: r3d | in-house: torchvision R(2+1)D-18 finetuned on NTIRE-enhanced ARID v1.5 split_1 (top-1 0.656 TTA) |
| `videomamba_t_arid_32f.pth` | 27 MB | recognize: videomamba | in-house: VideoMamba-Tiny 32-frame finetuned on enhanced ARID (top-1 0.688 TTA — best) |
| `realrestorer/` | ~39 GiB | enhance: realrestorer | `huggingface-cli download RealRestorer/RealRestorer --local-dir ckpts/realrestorer` (or symlink an existing HF snapshot) |

## Usage — offline
```bash
# full default pipeline: retinexformer + lightSR x2 + VideoMamba recognition
.venv/bin/darkpipe --input dark_clip.mp4 --output out.mp4 --events-json events.json

# enhancement only
.venv/bin/darkpipe --input dark.mp4 --sr off --recognize off

# best-quality offline restoration (diffusion; ~45 s/frame, short clips only)
.venv/bin/darkpipe --input dark.mp4 --enhance realrestorer --sr off --recognize off --max-frames 16

# recognition only, R(2+1)D-18
.venv/bin/darkpipe --input dark.mp4 --enhance off --sr off --recognize r3d
```
Output: mp4 whose frames are the enhanced (and 2×-upscaled) video with a **label bar below
the frame** showing `Action NN%` (green when confident); `--events-json` writes every
recognition event (frame index, label, confidence, top-3). All three functions off =
passthrough copy (warned).

## Usage — streaming server
```bash
.venv/bin/darkpipe --mode serve --input rtsp://camera/stream --port 8000
# file and webcam sources work too:  --input 0   |   --input demo.mp4  (file loops)
```
| endpoint | content |
|---|---|
| `GET /stream` | annotated video as browser-viewable MJPEG (`multipart/x-mixed-replace`) |
| `GET /events` | recognition results as Server-Sent Events JSON |
| `GET /health` | `fps_in`, `fps_proc`, `frames_dropped`, `latency_ms_last`, reconnects |
| `GET /config` | the active pipeline configuration |
```bash
curl -s localhost:8000/health | jq .
curl -N localhost:8000/events          # live recognition JSON
firefox http://localhost:8000/stream   # or ffplay/VLC
```
When the source is faster than the GPU, the newest frame wins (bounded latency; drops are
counted in `/health`). RealRestorer is rejected in serve mode (offline-only by design).

## Model selection guide
- **retinexformer** (default): 1.6 M params, ~6 ms/frame — the real-time enhancer.
- **realrestorer**: 12.4 B-param diffusion restorer conditioned on Qwen2.5-VL. Far higher
  visual quality on stills, ~45 s/frame with batched sequential offload on one 24 GB GPU.
  Offline keyframes/short clips only. Caution: on inputs with almost no signal it
  *hallucinates plausible detail* — superb for display, not evidence-preserving.
- **videomamba** (default): 7 M params, 63 ms/clip, top-1 0.688 — the better recognizer.
- **r3d**: 33 M params, 29 ms/clip, top-1 0.656 — the conservative baseline.
- Recognition consumes **post-enhance, pre-SR** frames (both checkpoints were trained on
  enhanced frames; SR was measured recognition-neutral, so it stays out of the decision path).

## Performance (single RTX 4090, 320×240 input)
| configuration | throughput | per-frame latency |
|---|---|---|
| enhance only | ~168 fps | ~6 ms |
| enhance + recognize | ~150 fps | ~10 ms |
| enhance + SR (+recognize) | **~6.5 fps** | ~170–200 ms |
| realrestorer | ~1/45 fps | 45 s |

Spec compliance: **real-time ≤ 0.5 s/frame — met** for every non-realrestorer configuration
(197 ms measured live in serve mode with the full pipeline). **Offline ≥ 25 fps — met without
SR** (150+ fps); with SR the single-GPU rate is ~6.5 fps (the lightSR stage is the bottleneck)
— reaching 25 fps requires ~4-GPU data-parallel sharding (27.6 fps aggregate measured in
development); this release is single-GPU and documents that scaling path.

## Architecture notes
- `darkpipe/vendor/` carries the model code so no external repos are needed at runtime:
  - `retinexformer_arch.py` — verbatim from Retinexformer (self-contained).
  - `mambairv2light_arch.py` — MambaIR arch with its basicsr couplings removed; runs
    against PyPI mamba-ssm (API-identical `selective_scan_fn`).
  - `videomamba/` — VideoMamba-Tiny with the fork's private fused op replaced by
    `bimamba_interface.py`, a thin wrapper over PyPI public kernels (logits-parity verified,
    max|Δ| ≈ 2e-4).
  - `realrestorer/` — the ComfyUI-RealRestorer reimplementation (SDPA-only) + a batched
    4-phase sequential-offload runner (never keeps Qwen-7B and the DiT resident together;
    streams DiT blocks from CPU with partial residency).
- lightSR invariants: fp32 weights + fp16 autocast (`.half()` breaks its norm layers), a
  PSNR self-check at load with fp32 fallback, and a fixed torch seed per forward (the arch
  uses gumbel-softmax routing *at eval time*).
- Every vendored model passed numerical parity gates against the original environments
  (`scripts/check_parity.py`; enhancement 68.5 dB, SR 73.2 dB, recognizer logits argmax-equal).

## Troubleshooting
- `Python.h: No such file or directory` at first import → you used a system Python without
  dev headers; recreate the venv with `UV_MANAGED_PYTHON=1` (see Environment setup).
- mamba wheel URL 404 → check the release pages for the `cu12torch2.7cxx11abiTRUE-cp310`
  asset of a newer version, or use the source-build fallback above.
- `numpy` ABI errors → keep `numpy>=2.0,<2.1` (pinned; do not upgrade past 2.1).
- VideoWriter fails → the opencv-python wheel ships mp4v; `avc1` is tried as fallback.
- RTSP drops/reconnects → watch `reconnects` in `/health`; capture retries with backoff.
- Everything about checkpoints → `scripts/download_ckpts.sh` prints what is missing.

## Tests & verification
```bash
.venv/bin/python -m pytest tests/ -q          # imports, CLI validation matrix, label bar
.venv/bin/python scripts/check_parity.py      # numerical parity gates (needs ckpts + refs)
```
