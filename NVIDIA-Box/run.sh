#!/usr/bin/env bash
set -euo pipefail
BOX_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BOX_PYTHON="${BOX_PYTHON:-${BOX_ROOT}/.venv/bin/python}"
if [[ ! -x "$BOX_PYTHON" ]]; then
    echo "找不到 Python 环境，请先执行 bash $BOX_ROOT/setup_env.sh，或设置 BOX_PYTHON 为已有环境的 Python 路径。" >&2
    exit 1
fi
exec "$BOX_PYTHON" -u "$BOX_ROOT/run.py" "$@"
