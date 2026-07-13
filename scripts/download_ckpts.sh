#!/usr/bin/env bash
# Checkpoint preparation for DarkScenePipeline. Run from the project root.
# GitHub downloads use the https://ghfast.top/ mirror prefix (faster in CN; remove if unwanted).
set -e
mkdir -p ckpts && cd ckpts
GH=https://ghfast.top/https://github.com

# 1) MambaIRv2 lightSR x2 (github release v1.0)
[ -f mambairv2_lightSR_x2.pth ] || \
  wget -O mambairv2_lightSR_x2.pth "$GH/csguoh/MambaIR/releases/download/v1.0/mambairv2_lightSR_x2.pth"

# 2) Retinexformer NTIRE weight (originally from the Retinexformer model zoo,
#    mirrored on this repo's release for one-command setup)
REL="$GH/ycwfs/DarkScenePipeline/releases/download/v1.0.0"
[ -f NTIRE.pth ] || wget -O NTIRE.pth "$REL/NTIRE.pth"

# 3) Finetuned recognizers (in-house ARID finetunes, published on this repo's release)
[ -f r2plus1d_arid.pth ] || wget -O r2plus1d_arid.pth "$REL/r2plus1d_arid.pth"
[ -f videomamba_t_arid_32f.pth ] || wget -O videomamba_t_arid_32f.pth "$REL/videomamba_t_arid_32f.pth"

# 4) RealRestorer HF bundle (~39 GiB) — only needed for --enhance realrestorer.
if [ ! -d realrestorer/transformer ]; then
  echo "Downloading RealRestorer bundle (39 GiB)..."
  ../.venv/bin/huggingface-cli download RealRestorer/RealRestorer --local-dir realrestorer
fi
echo "checkpoint dir:"; ls -la .
