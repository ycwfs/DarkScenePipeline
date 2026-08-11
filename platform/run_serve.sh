#!/usr/bin/env bash
# 启动常驻的在线推理服务（摄像头一直开着的场景），不是"算子"。
#
# 为什么不是算子：platform/op_dark_behavior 那条路径要求容器跑完退出、把结果写到
# outputPath 指定的文件里，框架再去收集——docs/第三方算法开封接入规范1.docx 里的算子机制
# 通篇是这个"文件产出"模型，没有常驻服务这个概念。摄像头永远不断流，容器也就永远不退出，
# 天然不满足这个模型；所以 serve 模式单独用这个脚本以普通常驻容器的方式跑，不参与
# suanzi.json / zip 打包，也不提交给平台。
#
# 复用同一个镜像：darkpipe-operator:0.2.0 已经装好 torch/mamba_ssm/fastapi/uvicorn 等
# 全部依赖、也内置了权重，缺的只是 darkpipe/ 源码——算子路径靠 zip 注入源码，这里没有
# zip，所以用 -v 只读挂载仓库里的 darkpipe/，和 DEVELOPING.md 第 5 步验证 zip 时的挂载
# 方式是同一个道理：镜像里不放源码，源码只有仓库这一份。
#
# 用法：
#   bash platform/run_serve.sh rtsp://<gb28181网关下发的拉流地址>
#   bash platform/run_serve.sh http://cam.local/live.flv --recognize behavior
#   PORT=8080 GPU=1 bash platform/run_serve.sh rtsp://...
#
# 本地无摄像头时用一段视频代替（server.py 的 capture_loop 对本地文件会自动循环播放，
# 可以当摄像头长开的替身来验证服务本身）：
#   bash platform/run_serve.sh /data/some_clip.mp4
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"

if [[ $# -lt 1 ]]; then
    echo "用法: $0 <视频源: rtsp:// | http(s):// (flv/hls) | 本地文件路径> [darkpipe.cli 额外参数...]" >&2
    exit 1
fi
SRC="$1"; shift

IMAGE="${IMAGE:-darkpipe-operator:0.2.0}"
NAME="${NAME:-darkpipe-live}"
PORT="${PORT:-8000}"
GPU="${GPU:-all}"

# 本地文件走 -v 只读挂载进容器；rtsp/http(s) 直接透传地址，不需要挂载。
DOCKER_MOUNTS=(-v "$REPO/darkpipe:/opt/darkpipe/darkpipe:ro")
INPUT="$SRC"
if [[ "$SRC" != *"://"* ]]; then
    if [[ ! -f "$SRC" ]]; then
        echo "error: 本地路径不存在: $SRC" >&2
        exit 1
    fi
    DOCKER_MOUNTS+=(-v "$(cd "$(dirname "$SRC")" && pwd)/$(basename "$SRC"):/data/in:ro")
    INPUT="/data/in"
fi

GPUS_FLAG="all"
[[ "$GPU" != "all" ]] && GPUS_FLAG="device=$GPU"

echo "[serve] image=$IMAGE name=$NAME port=$PORT gpu=$GPU input=$SRC"
docker run -d --restart unless-stopped \
    --gpus "$GPUS_FLAG" \
    --name "$NAME" \
    -p "$PORT:8000" \
    "${DOCKER_MOUNTS[@]}" \
    -e PYTHONPATH=/opt/darkpipe \
    -w /opt/darkpipe \
    "$IMAGE" \
    /opt/conda/envs/darkpipe/bin/python -u -m darkpipe.cli --mode serve \
        --input "$INPUT" \
        --enhance retinexformer --sr bicubic --recognize behavior \
        --ckpt-dir /opt/darkpipe/ckpts --host 0.0.0.0 --port 8000 \
        "$@"

echo "[serve] 已启动，容器名 $NAME（docker logs -f $NAME 看启动日志）"
echo "[serve]   健康检查:  curl http://localhost:$PORT/health"
echo "[serve]   事件流:    curl -N http://localhost:$PORT/events"
echo "[serve]   画面流:    浏览器打开 http://localhost:$PORT/stream"
echo "[serve]   停止:      docker rm -f $NAME"
