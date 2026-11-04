#!/usr/bin/env bash
# JetPack owns torch/torchvision/TensorRT. Do not overwrite those with x86/PyPI wheels.
set -euo pipefail
BOX_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BOX_BASE_PYTHON="${BOX_BASE_PYTHON:-python3}"
BOX_VENV="${BOX_VENV:-${BOX_ROOT}/.venv}"
"$BOX_BASE_PYTHON" -c 'import platform,sys; assert platform.machine()=="aarch64", "仅供 Jetson ARM64"; assert sys.version_info[:2]==(3,10), "当前适配目标为 JetPack 6 / Python 3.10"'
for cmd in ffmpeg ffprobe gcc nvcc; do
    command -v "$cmd" >/dev/null || { echo "缺少 $cmd；请按 JETSON.md 安装系统依赖" >&2; exit 1; }
done
if [[ ! -x "$BOX_VENV/bin/python" ]]; then
    "$BOX_BASE_PYTHON" -m venv --system-site-packages "$BOX_VENV"
fi
BOX_PYTHON="$BOX_VENV/bin/python"
if [[ -n "${BOX_TORCH_WHL:-}" ]]; then
    "$BOX_PYTHON" -m pip install --no-deps "$BOX_TORCH_WHL"
fi
if [[ -n "${BOX_TORCHVISION_WHL:-}" ]]; then
    "$BOX_PYTHON" -m pip install --no-deps "$BOX_TORCHVISION_WHL"
fi
"$BOX_PYTHON" -c 'import torch,torchvision,tensorrt,cv2; assert torch.cuda.is_available(), "需要与 JetPack 匹配的 NVIDIA ARM64 PyTorch"; assert int(tensorrt.__version__.split(".")[0])>=10, "需要 TensorRT 10+"; print(torch.__version__,tensorrt.__version__,torch.cuda.get_device_name(0))'
"$BOX_PYTHON" -m pip install -r "$BOX_ROOT/requirements-jetson.txt"
"$BOX_PYTHON" -c 'import torch,triton; assert torch.cuda.is_available(); print("CUDA/Triton ready")'
export MAX_JOBS="${MAX_JOBS:-1}"
export TORCH_CUDA_ARCH_LIST=8.7
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export MAMBA_FORCE_BUILD=TRUE
"$BOX_PYTHON" -m pip install --no-build-isolation --no-deps --no-binary=causal-conv1d causal-conv1d==1.5.2
"$BOX_PYTHON" -m pip install --no-build-isolation --no-deps --no-binary=mamba-ssm mamba-ssm==2.2.6.post3
"$BOX_PYTHON" "$BOX_ROOT/check_env.py" --files-only
"$BOX_PYTHON" "$BOX_ROOT/build_engines.py"
"$BOX_PYTHON" "$BOX_ROOT/check_env.py"
echo '环境检查完成。首次模型自检已在本机构建 TensorRT engines；真机容量/帧率仍需 benchmark_box.py 验收。'
