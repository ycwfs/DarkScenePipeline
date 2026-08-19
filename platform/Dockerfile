# 暗光场景行为识别算子运行镜像（处理算子与测试验证算子共用）
#
# 为什么自带基础镜像：规范预置的是 python 3.5-3.9 基础镜像，而本算子需要 python 3.10 ——
# VideoMamba 依赖的 mamba-ssm / causal-conv1d 官方预编译 wheel 只提供 cp310 版本，且需要与
# torch 2.7 / cu12 / cxx11abiTRUE 严格对应。规范允许“如预置镜像不满足条件，需要单独提供算子
# 运行基础docker镜像”，本镜像即为此提供。
#
# 镜像里只放**运行环境和模型权重**，不放算子源码：源码（main.py / suanzi.json / README.md /
# darkpipe/）由 zip 包提交，框架注入容器后以 command 启动。这样源码只有一个来源，不会出现
# 镜像里的代码和 zip 里的代码版本不一致。
#
# 构建上下文由 build_image.sh 生成（它只把需要的权重挑进去），不要直接对仓库根目录 build。
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/conda/bin:$PATH

# libglib2.0-0 是 opencv 的运行期依赖；装的是 headless 版 opencv，所以不需要 libgl1。
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates wget bzip2 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# conda 环境路径必须与 suanzi.json 的 command 一致：
#   /opt/conda/envs/darkpipe/bin/python
#
# 用 Miniforge 而不是 Miniconda：Miniforge 默认只用 conda-forge 频道。Miniconda 默认的
# defaults 频道（repo.anaconda.com）自 2024 年起要求 200 人以上的组织购买商业授权，构建时还
# 会因未接受条款直接失败（TermsOfServiceNotAcceptedError）。本镜像要交付给数据中心平台运行，
# 不能把这种授权义务一起交付出去，所以整个环境走 conda-forge。
#
# Miniforge 安装包在 GitHub Release 上，与下面的 mamba/causal-conv1d wheel 同一套网络依赖：
# GitHub 不通时用 --build-arg GH_PREFIX=https://ghfast.top/ 走镜像。
ARG GH_PREFIX=
ARG MINIFORGE_URL=https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
# 国内构建可换成 https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
ARG CONDA_CHANNEL=conda-forge
RUN wget -q -O /tmp/miniforge.sh "${GH_PREFIX}${MINIFORGE_URL}" \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && /opt/conda/bin/conda create -y -n darkpipe \
        --override-channels -c "${CONDA_CHANNEL}" python=3.10 \
    && /opt/conda/bin/conda clean -afy

ENV PY=/opt/conda/envs/darkpipe/bin/python

# 国内构建可用镜像源加速：
#   docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple ...
ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

# torch 2.7.0 的 PyPI 默认 wheel 自带 cu126，与基础镜像的 CUDA 12.6 对应，无需额外 index。
RUN $PY -m pip install --no-cache-dir --upgrade pip \
    && $PY -m pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0

# mamba-ssm / causal-conv1d 预编译 CUDA kernel（cp310 + torch2.7 + cxx11abiTRUE），装它们就
# 不需要容器里有 nvcc。默认从 GitHub Release 取，同样受上面的 GH_PREFIX 控制。
ARG MAMBA_WHL=https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
ARG CONV1D_WHL=https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.2/causal_conv1d-1.5.2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
RUN $PY -m pip install --no-cache-dir "${GH_PREFIX}${CONV1D_WHL}" "${GH_PREFIX}${MAMBA_WHL}"

# 其余依赖与仓库 pyproject.toml 保持一致。opencv 用 headless 版：容器里没有 X11，算子也从不
# 调用 imshow（已确认 darkpipe/ 中无任何 GUI 调用）。同一个镜像也用来跑常驻的 serve 模式
# （摄像头长开、实时出流，见 platform/op_dark_behavior_serve 与 platform/run_serve.sh），
# 所以装上 fastapi/uvicorn（几 MB，相对镜像体积可忽略）。
RUN $PY -m pip install --no-cache-dir \
        "transformers==4.57.3" \
        "timm==1.0.27" \
        "einops>=0.8" \
        "numpy>=2.0,<2.1" \
        "opencv-python-headless>=4.10" \
        "fastapi>=0.115" \
        "uvicorn>=0.30" \
        "safetensors>=0.4" \
        "pillow>=10" \
        "huggingface_hub>=0.30"

# triton（torch 的依赖，由 mamba_ssm 触发）在第一次用到 GPU 时会现场编译一个 C 辅助模块，
# 编译不了就直接 RuntimeError: Failed to find C compiler。runtime 版 CUDA 基础镜像不带编译器，
# 所以必须补一个 gcc。这个坑构建期发现不了——构建期没有 GPU，triton 根本走不到编译那一步，
# 只有 docker run --gpus all 实跑才会暴露。
# 放在 pip 安装之后而不是和上面的 apt 合并：合并会让 13 GB 的依赖层全部失效重装。
# libc6-dev 不能省：--no-install-recommends 装的 gcc 不带 crti.o/Scrt1.o，链接阶段会失败。
# 链接 -lcuda 用的 libcuda.so 由 nvidia-container-toolkit 在运行时注入，不需要装 CUDA stub。
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# ffmpeg：实时服务算子的多格式出流（HTTP-FLV / HLS / 推流到外部 RTMP-RTSP 服务器）全部由它
# 封装，MJPEG 那条路不经过它。opencv 自带的是 ffmpeg 的**库**，没有命令行程序，而这里要的正是
# 一个能把 MJPEG 转封装成 FLV/HLS 并写到管道或推到远端的进程，所以必须单独装。
#
# 单独一层、放在 gcc 那层之后：上面那条注释的道理同样适用——改动这里不应该让已经装好的
# 13 GB pip 依赖层（乃至 gcc 层）失效重装。代价是镜像多约 250 MB，相对 17 GB 可以接受。
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# triton 默认把编译缓存写到 $HOME/.triton。平台以哪个用户、哪个 HOME 启动容器都不确定，
# HOME 不可写时同样会失败，因此固定到一定可写的 /tmp。
ENV TRITON_CACHE_DIR=/tmp/triton_cache

# 模型权重。build_image.sh 已把需要的权重挑进构建上下文的 ckpts/ 下，路径与两个算子
# ckpt_dir 的默认值 /opt/darkpipe/ckpts 一致，因此默认配置开箱即用、运行期不下载任何东西。
COPY ckpts/ /opt/darkpipe/ckpts/

# 自检。构建期没有 GPU，所以只做两件事：
#   1) 纯 CPU 也能导入的依赖，真的 import 一遍；
#   2) mamba_ssm / causal_conv1d 只查版本、不 import——它们会连带初始化 triton，而 triton 在
#      没有 GPU 的构建环境里直接抛 "0 active drivers"。这两个包能否真正跑起来，由交付前的
#      docker run --gpus all 实跑验证（见 DEVELOPING.md），而不是在这里假装验证过。
RUN $PY -c "import torch, cv2, timm, einops, transformers; \
from importlib.metadata import version; \
print('torch', torch.__version__, 'cv2', cv2.__version__, \
'mamba_ssm', version('mamba_ssm'), 'causal_conv1d', version('causal_conv1d'), 'ok')"

# ffmpeg 不需要 GPU，所以这一条构建期就能真正验证，不是"假装验证"：确认命令行存在，且真的
# 带着 libx264 编码器与 flv/hls 封装器——Ubuntu 的 ffmpeg 是全功能构建，但装错源或
# 换成精简构建时，缺的正是这几样，而那种缺失要到运行期开流才暴露。
# libx264 不只出流要用：所有落盘的 mp4（动作切片、离线整段视频）也由 darkpipe/media.py 交给
# ffmpeg 编码。cv2 这个构建里根本没有可用的 H.264 编码器，缺了 ffmpeg 就只能退回 MPEG-4
# Part 2，文件在浏览器和微信里打不开——这正是之前"切片都打不开"的成因，别把这一条删了。
RUN ffmpeg -hide_banner -version | head -1 \
    && ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libx264 \
    && ffmpeg -hide_banner -muxers 2>/dev/null | grep -qE '^ *E +flv' \
    && ffmpeg -hide_banner -muxers 2>/dev/null | grep -qE '^ *E +hls' \
    && echo 'ffmpeg libx264/flv/hls ok'

WORKDIR /opt/darkpipe
# 不设 ENTRYPOINT：规范要求由 suanzi.json 的 command 提供
#   ["/opt/conda/envs/darkpipe/bin/python", "-u", "main.py"]
