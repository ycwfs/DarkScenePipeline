#!/usr/bin/env bash
# Install into a private venv. System prerequisites are documented in README.md.
set -euo pipefail
BOX_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(uname -m)" == aarch64 ]]; then
    exec bash "$BOX_ROOT/setup_jetson.sh" "$@"
fi
BOX_BASE_PYTHON="${BOX_BASE_PYTHON:-python3.10}"
BOX_VENV="${BOX_VENV:-${BOX_ROOT}/.venv}"
"$BOX_BASE_PYTHON" -c 'import platform,sys; assert sys.version_info[:2] == (3,10), "需要 Python 3.10"; assert platform.system() == "Linux" and platform.machine() == "x86_64", "预编译 CUDA wheels 需要 Linux x86_64；Jetson/aarch64 需要单独适配"'
for cmd in ffmpeg ffprobe gcc nvidia-smi; do
    command -v "$cmd" >/dev/null || { echo "缺少 $cmd，请先安装 README 中的系统依赖。" >&2; exit 1; }
done
"$BOX_BASE_PYTHON" -m venv "$BOX_VENV"
BOX_PYTHON="$BOX_VENV/bin/python"
"$BOX_PYTHON" -m pip install --upgrade pip
"$BOX_PYTHON" -m pip install -r "$BOX_ROOT/requirements.txt" -c "$BOX_ROOT/constraints.txt"
# Matched wheels avoid a source build; --no-deps avoids rtmlib's CPU ORT dependency.
BOX_MAMBA_WHL="${BOX_MAMBA_WHL:-https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl}"
BOX_CONV1D_WHL="${BOX_CONV1D_WHL:-https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.2/causal_conv1d-1.5.2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl}"
"$BOX_PYTHON" -m pip install --no-deps -c "$BOX_ROOT/constraints.txt" "$BOX_CONV1D_WHL" "$BOX_MAMBA_WHL" rtmlib==0.0.16
"$BOX_PYTHON" "$BOX_ROOT/check_env.py"
echo "环境安装及自检完成。运行：BOX_PYTHON=$BOX_PYTHON bash $BOX_ROOT/run.sh --video_path /data/input.mp4"
