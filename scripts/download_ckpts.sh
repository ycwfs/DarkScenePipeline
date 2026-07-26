#!/usr/bin/env bash
# Checkpoint preparation for DarkScenePipeline. Run from the project root.
#   bash scripts/download_ckpts.sh                 # everything, x2 neural SR weights
#   bash scripts/download_ckpts.sh --sr-scale 4    # ... and the x4 SR weights as well
# GitHub downloads use the https://ghfast.top/ mirror prefix (faster in CN; remove if unwanted).
set -e
SR_SCALES=2
while [ $# -gt 0 ]; do
  case "$1" in
    --sr-scale) SR_SCALES="$SR_SCALES $2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
mkdir -p ckpts && cd ckpts
GH=https://ghfast.top/https://github.com

# 1) Super-resolution backends (--sr): MambaIRv2 lightSR and CATANet (CVPR2025). Both are
#    optional — --sr defaults to off (see README 'Performance') — and per-scale: a checkpoint
#    is trained for one factor, so --sr-scale 3/4 needs its own file from the same release.
#    (--sr bicubic has no weights at any scale.)
for S in $(echo "$SR_SCALES" | tr ' ' '\n' | sort -u); do
  [ -f "mambairv2_lightSR_x$S.pth" ] || \
    wget -O "mambairv2_lightSR_x$S.pth" "$GH/csguoh/MambaIR/releases/download/v1.0/mambairv2_lightSR_x$S.pth"
  [ -f "catanet_x$S.pth" ] || \
    wget -O "catanet_x$S.pth" "$GH/EquationWalker/CATANet/releases/download/v0.0/x$S.pth"
done

# 2) Retinexformer NTIRE weight (originally from the Retinexformer model zoo,
#    mirrored on this repo's release for one-command setup)
REL="$GH/ycwfs/DarkScenePipeline/releases/download/v1.0.0"
[ -f NTIRE.pth ] || wget -O NTIRE.pth "$REL/NTIRE.pth"

# 2b) HVI-CIDNet generalization weight (enhance: cidnet). From Fediory/HVI-CIDNet
#     (LOLv2_syn/generalization.pth — its strongest cross-domain weight). Not on a
#     GitHub release; stage from a local HVI-CIDNet checkout, else fetch via its README links.
if [ ! -f CIDNet_generalization.pth ]; then
  if [ -f ../../HVI-CIDNet/weights/LOLv2_syn/generalization.pth ]; then
    cp ../../HVI-CIDNet/weights/LOLv2_syn/generalization.pth CIDNet_generalization.pth
  else
    echo "  (missing CIDNet_generalization.pth — copy HVI-CIDNet/weights/LOLv2_syn/generalization.pth here)"
  fi
fi

# 3) Finetuned recognizers (in-house finetunes, published on this repo's release)
[ -f r2plus1d_arid.pth ] || wget -O r2plus1d_arid.pth "$REL/r2plus1d_arid.pth"
[ -f videomamba_t_arid_32f.pth ] || wget -O videomamba_t_arid_32f.pth "$REL/videomamba_t_arid_32f.pth"
# 10-class behavior head (--recognize behavior)
[ -f videomamba_t_behavior_32f.pth ] || \
  wget -O videomamba_t_behavior_32f.pth "$REL/videomamba_t_behavior_32f.pth"

# 3b) X-CLIP zero-shot snapshot (780 MB) — only needed for --recognize xclip.
#     huggingface.co times out from CN; hf-mirror.com serves the same snapshot.
if [ ! -e xclip-base-patch16-zero-shot/pytorch_model.bin ] && \
   [ ! -e xclip-base-patch16-zero-shot/model.safetensors ]; then
  HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com} \
    ../.venv/bin/huggingface-cli download microsoft/xclip-base-patch16-zero-shot \
      --local-dir xclip-base-patch16-zero-shot
fi

# 4) RealRestorer HF bundle (~39 GiB) — only needed for --enhance realrestorer.
if [ ! -d realrestorer/transformer ]; then
  echo "Downloading RealRestorer bundle (39 GiB)..."
  ../.venv/bin/huggingface-cli download RealRestorer/RealRestorer --local-dir realrestorer
fi
echo "checkpoint dir:"; ls -la .
