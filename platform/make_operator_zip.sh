#!/usr/bin/env bash
# 打包算子 zip：suanzi.json / README.md / main.py 直接位于包根，不套任何文件夹。
#
# 用法：
#   bash platform/make_operator_zip.sh                      # 打包全部算子
#   bash platform/make_operator_zip.sh op_dark_behavior     # 只打包指定算子
#
# 包名取自算子目录下的 zip_name 文件（一级类别_二级类别_算子名称）。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
DIST="${DIST:-$HERE/dist}"

command -v zip >/dev/null || { echo "error: 需要 zip 命令（apt-get install zip）" >&2; exit 1; }

ops=("$@")
if [[ ${#ops[@]} -eq 0 ]]; then
    ops=()
    for d in "$HERE"/op_*/; do ops+=("$(basename "$d")"); done
fi

mkdir -p "$DIST"
for op in "${ops[@]}"; do
    src="$HERE/$op"
    [[ -d "$src" ]] || { echo "error: 找不到算子目录 $src" >&2; exit 1; }
    for f in main.py suanzi.json README.md zip_name; do
        [[ -f "$src/$f" ]] || { echo "error: $op 缺少 $f" >&2; exit 1; }
    done

    name="$(tr -d '[:space:]' < "$src/zip_name")"
    zipfile="$DIST/$name.zip"
    stage="$(mktemp -d)"
    trap 'rm -rf "$stage"' EXIT

    # 规范：suanzi.json 与 README.md 不能包含在任何文件夹下
    cp "$src/main.py" "$src/suanzi.json" "$src/README.md" "$stage/"
    # 共用小工具与算法本体。darkpipe 只从仓库这一处复制，zip 里的代码与仓库不可能版本漂移；
    # main.py 会把自身所在目录放在 sys.path 最前面，因此包内的 darkpipe 一定优先被导入。
    cp "$HERE/oputil.py" "$stage/"
    rsync -a --exclude='__pycache__' --exclude='*.pyc' "$REPO/darkpipe/" "$stage/darkpipe/"

    # 提交前自检：manifest 不合规就不出包
    python3 "$HERE/validate_suanzi.py" "$stage/suanzi.json" >/dev/null || {
        echo "error: $op 的 suanzi.json 未通过规范校验，运行以下命令查看详情：" >&2
        echo "       python3 platform/validate_suanzi.py platform/$op/suanzi.json" >&2
        exit 1
    }

    rm -f "$zipfile"
    (cd "$stage" && zip -qr "$zipfile" . -x '*.pyc' '*/__pycache__/*')
    rm -rf "$stage"
    trap - EXIT

    echo "[zip] $zipfile  ($(du -h "$zipfile" | cut -f1))"
    # 用 sed 取前几行而不是 head：head 会提前关闭管道，pipefail 下 unzip 收到 SIGPIPE
    # 会被当成失败，整个脚本在打完第一个包之后就退出了。
    unzip -l "$zipfile" | sed -n '1,8p;$p' | sed 's/^/      /'
done

echo
echo "提交前请确认：包根是 suanzi.json / README.md / main.py，没有多余的外层目录。"
