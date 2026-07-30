#!/usr/bin/env bash
# 构建算子运行镜像，并可选导出成交付用的 tar.gz。
#
# 用法：
#   bash platform/build_image.sh                       # 只构建
#   bash platform/build_image.sh --save                 # 构建并导出 darkpipe-operator-0.1.0.tar.gz
#   GH_PREFIX=https://ghfast.top/ bash platform/build_image.sh   # GitHub 不通时走镜像
#   PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple bash platform/build_image.sh
#   CONDA_CHANNEL=https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge bash ...
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
IMAGE="${IMAGE:-darkpipe-operator:0.1.0}"
CTX="$HERE/.build_ctx"

# 默认配置（retinexformer + behavior）必需的权重，缺一不可
REQUIRED=(NTIRE.pth videomamba_t_behavior_32f.pth)
# 可选权重：存在就打进镜像，让 enhance=cidnet 等备选配置也开箱即用；不存在只提示
OPTIONAL=(CIDNet_generalization.pth r2plus1d_arid.pth videomamba_t_arid_32f.pth)

rm -rf "$CTX"
mkdir -p "$CTX/ckpts"

for f in "${REQUIRED[@]}"; do
    if [[ ! -f "$REPO/ckpts/$f" ]]; then
        echo "error: 缺少必需权重 $REPO/ckpts/$f" >&2
        echo "       先执行 scripts/download_ckpts.sh 准备权重再构建镜像" >&2
        exit 1
    fi
    cp "$REPO/ckpts/$f" "$CTX/ckpts/"
    echo "[ctx] 必需权重 $f"
done
for f in "${OPTIONAL[@]}"; do
    if [[ -f "$REPO/ckpts/$f" ]]; then
        cp "$REPO/ckpts/$f" "$CTX/ckpts/"
        echo "[ctx] 可选权重 $f"
    else
        echo "[ctx] 跳过未准备的可选权重 $f（对应配置需在运行时挂载 ckpt_dir）"
    fi
done
cp "$HERE/Dockerfile" "$CTX/Dockerfile"
echo "[ctx] 构建上下文 $(du -sh "$CTX" | cut -f1)"

BUILD_ARGS=()
[[ -n "${GH_PREFIX:-}" ]]      && BUILD_ARGS+=(--build-arg "GH_PREFIX=$GH_PREFIX")
[[ -n "${PIP_INDEX_URL:-}" ]]  && BUILD_ARGS+=(--build-arg "PIP_INDEX_URL=$PIP_INDEX_URL")
[[ -n "${CONDA_CHANNEL:-}" ]]  && BUILD_ARGS+=(--build-arg "CONDA_CHANNEL=$CONDA_CHANNEL")

echo "[build] docker build -t $IMAGE"
docker build -t "$IMAGE" "${BUILD_ARGS[@]}" "$CTX"
rm -rf "$CTX"

echo "[build] 完成：$IMAGE"
docker images --format '  {{.Repository}}:{{.Tag}}  {{.Size}}' "${IMAGE%%:*}" | head -5

if [[ "${1:-}" == "--save" ]]; then
    OUT="$HERE/$(echo "$IMAGE" | tr ':/' '-').tar.gz"
    echo "[save] 导出到 $OUT（体积较大，请耐心等待）"
    docker save "$IMAGE" | gzip > "$OUT"
    echo "[save] $(du -h "$OUT" | cut -f1)  $OUT"
fi
