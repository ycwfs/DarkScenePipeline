#!/usr/bin/env bash
# 构建算子运行镜像，并可选导出成交付用的 tar.gz。
#
# 用法：
#   bash platform/build_image.sh                       # 只构建
#   bash platform/build_image.sh --save                 # 构建并导出 darkpipe-operator-0.2.0.tar.gz
#   GH_PREFIX=https://ghfast.top/ bash platform/build_image.sh   # GitHub 不通时走镜像
#   PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple bash platform/build_image.sh
#   CONDA_CHANNEL=https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge bash ...
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
IMAGE="${IMAGE:-darkpipe-operator:0.2.0}"
CTX="$HERE/.build_ctx"

# 平台部署只暴露 off/retinexformer + off/bicubic + off/behavior，缺一不可；
# 其余权重（cidnet / r3d / ARID-videomamba / xclip 等）不再被 suanzi.json 的
# choice 列表提供，因此不再打进镜像。
REQUIRED=(NTIRE.pth videomamba_t_behavior_32f.pth)

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
    # pigz 就是多线程版 gzip，产物仍是标准 .tar.gz，接收方按普通 gzip 解即可。18 GB 的镜像
    # 单线程 gzip 要压十几分钟，这台 56 核的机器上并行压不到两分钟；没有 pigz 时自动退回 gzip。
    if command -v pigz >/dev/null; then
        ZIP=(pigz -p "$(( $(nproc) / 4 + 1 ))")
    else
        ZIP=(gzip)
    fi
    echo "[save] 导出到 $OUT（用 ${ZIP[0]}，体积较大，请耐心等待）"
    # 先写 .part 再改名：导出中途被打断时，留下的是一个显然不完整的 .part，而不是一个看起来
    # 正常、解压到一半才报错的 tar.gz——后者会一路带到交付现场才暴露。
    docker save "$IMAGE" | "${ZIP[@]}" > "$OUT.part"
    mv "$OUT.part" "$OUT"
    echo "[save] $(du -h "$OUT" | cut -f1)  $OUT"
    echo "[save] 接收方导入：gunzip -c $(basename "$OUT") | docker load"
fi
