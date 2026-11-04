#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
DARK_NV="$PWD/.venv/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="$DARK_NV/cudnn/lib:$DARK_NV/cublas/lib:$DARK_NV/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
DARK_PIDS=()
trap 'kill "${DARK_PIDS[@]}" 2>/dev/null || true' INT TERM
for DARK_SHARD in 0 1 2 3; do
  DARK_GPU=$((DARK_SHARD+2))
  CUDA_VISIBLE_DEVICES="$DARK_GPU" .venv/bin/python reports/dark-annotations-20261023/scripts/propose.py \
    --shard "$DARK_SHARD/4" > "reports/dark-annotations-20261023/proposals-$DARK_SHARD.log" 2>&1 &
  DARK_PIDS+=("$!")
done
DARK_FAILED=0
for DARK_PID in "${DARK_PIDS[@]}"; do
  wait "$DARK_PID" || DARK_FAILED=1
done
exit "$DARK_FAILED"
