#!/usr/bin/env bash
# Checkpoint preparation for DarkScenePipeline. Run from the project root.
# GitHub downloads use the https://ghfast.top/ mirror prefix (faster in CN; remove if unwanted).
set -e
mkdir -p ckpts && cd ckpts
GH=https://ghfast.top/https://github.com

# 1) MambaIRv2 lightSR x2 (github release v1.0)
[ -f mambairv2_lightSR_x2.pth ] || \
  wget -O mambairv2_lightSR_x2.pth "$GH/csguoh/MambaIR/releases/download/v1.0/mambairv2_lightSR_x2.pth"

# 2) Retinexformer NTIRE weight — from the Retinexformer release/model zoo.
#    On this machine: cp /data1/data1/wfs/project/low-light/Retinexformer/pretrain_model/NTIRE.pth .
[ -f NTIRE.pth ] || echo "!! NTIRE.pth missing: copy from a Retinexformer checkout (pretrain_model/NTIRE.pth)"

# 3) Finetuned recognizers (in-house ARID finetunes, published on this repo's release)
REL="$GH/ycwfs/DarkScenePipeline/releases/download/v1.0.0"
[ -f r2plus1d_arid.pth ] || wget -O r2plus1d_arid.pth "$REL/r2plus1d_arid.pth"
[ -f videomamba_t_arid_32f.pth ] || wget -O videomamba_t_arid_32f.pth "$REL/videomamba_t_arid_32f.pth"

# 4) RealRestorer HF bundle (~39 GiB) — only needed for --enhance realrestorer.
if [ ! -d realrestorer/transformer ]; then
  echo "Downloading RealRestorer bundle (39 GiB)..."
  ../.venv/bin/huggingface-cli download RealRestorer/RealRestorer --local-dir realrestorer
fi
echo "checkpoint dir:"; ls -la .
