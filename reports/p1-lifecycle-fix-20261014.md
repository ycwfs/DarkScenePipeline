# P1-1 / P1-2 修复与验收

已修复交付目录 `platform/dark-demo-v2` 的处理故障传播和持续连接退出问题。独立修复包解包后，在指定镜像 `darkpipe-operator:0.3.0` 下通过全部 **22 项回归测试**。原始算子包与历史审查证据保留，仓库改动尚未提交。

## 交付文件

修复包：[dark-demo-v2_p1fix_20261014.tar.gz](../platform/dark-demo-v2_p1fix_20261014.tar.gz)。归档继续包含 `dark-demo-v2/` 外层目录，文件集合与原包相同，共 44 个文件。

| 项目 | 值 |
| --- | --- |
| 修复包大小 | 190,640,920 字节 |
| MD5 | `e9825754d94f82f33ecd8d26dfa469b2` |
| SHA256 | `f365ce0e49c5ac40667a6663663acf9a84fa6608ec6aa3767d2ff5af2fcda389` |
| 验收镜像 ID | `sha256:d359716620d7eb757cd7c36733e25642469450f1c801e0367fc99248f0a2f212` |
| 原包 SHA256 | `74b30d21ea0367a34df60f2ebb807337baa4cbb219c35fe288f187c68adc9e91` |

归档内仅 `darkpipe/server.py`、`main.py` 和 `README.md` 的内容改变。7 份权重及其余代码、字体、清单逐文件 SHA256 与原包一致。[完整校验记录](p1-lifecycle-fix-20261014/package-integrity.json)。

## P1-1：处理故障传播

- `ServerState` 记录处理存活、模型就绪、首个致命错误与最近成功生成 JPEG 的时间。
- 模型初始化、预热、主增强循环，以及输出、独立识别、其他增强工作线程异常，统一触发停止。线程内 `SystemExit` 也会被记录为失败。正常停止后清理已加载模型；初始化失败时也清理部分加载的模型。
- `/health` 同时检查采集、处理和最近产帧；初始化未完成、尚无首帧、处理失败、正在停止或超过 5 秒未更新 JPEG 时返回 `503/degraded`。处理失败时额外给出 `processing_status: failed` 与 `processing_error`，不再用正常采集掩盖处理死亡。单一路出流暂时重连仍只降级该输出。
- 服务收尾后抛出带最终状态的 `ProcessingFailed`；入口保存 `failed`、错误和事件/片段统计，以非零码退出。清理失败不会覆盖首个处理错误；工作线程超时未退出时不会卸载它仍在使用的模型。

## P1-2：持续连接与信号退出

- 到期、处理失败和 Uvicorn 停止信号共用 `request_stop()`；停止通知发生在等待 HTTP 请求结束之前。
- SSE 用结束标记唤醒阻塞的 `q.get()`，并解除订阅。客户端取消 SSE 时也唤醒后台读取，避免 executor 再等待 15 秒。
- MJPEG、FLV 生成器检查停止状态；停止 FLV 的 FFmpeg 子进程以解除阻塞的 stdout 读取。
- HTTP 优雅退出设置 5 秒上限。客户端不读取数据、使 ASGI 发送阻塞时，超时取消请求并继续会话收尾。编码、线程等待、片段写盘和上传收尾耗时另计。
- Uvicorn 的信号处理保留优雅退出及第二次 Ctrl-C 强制退出，但不在返回入口前重新触发终止信号，确保 SIGTERM/SIGINT 后能够写完整的 `stopped` 会话。

## 验证结果

镜像测试挂载的是修复归档的独立解包副本；资源为 CPU limit=8、memory limit=12Gi，禁用 GPU 与外部网络。初始化/模型推理和视频采集使用确定性的 CPU 替身，实际执行交付包的线程管线、Uvicorn/FastAPI、SSE/MJPEG、FFmpeg FLV、动作切片和入口会话写入。

| 验证项 | 结果 |
| --- | --- |
| 本地 v2 生命周期回归 | 22 passed |
| 修复归档 + 指定镜像生命周期回归 | **22 passed，54.51 秒** |
| 旧代码 CPU 回归 | **286 passed，1 skipped，4 deselected，25.92 秒** |
| 修改的 Python 文件语法、diff 空白检查 | 通过 |
| 镜像测试产出的 6 个动作片段 | 全部 H.264，完整解码无错误 |
| 原包、修复包文件集合及权重校验 | 通过 |

真实 HTTP 连接在子进程退出前始终保持打开，没有先由测试端断开来帮助服务退出。

| 镜像内实际入口场景 | 退出码 / 会话 | 入口运行耗时 |
| --- | --- | ---: |
| SSE + MJPEG + FLV 同时连接，`run_seconds=3` | 0 / stopped | 3.75 秒 |
| 输入打开但未产出首帧，三路连接等待，`run_seconds=3` | 0 / stopped | 3.65 秒 |
| MJPEG 客户端不读数据，`run_seconds=2` | 0 / stopped；5 秒后取消阻塞请求 | 7.67 秒 |
| SIGTERM，保持 SSE 连接 | 0 / stopped | 0.83 秒 |
| SIGINT，保持 SSE 连接 | 0 / stopped | 0.85 秒 |
| 注入增强异常，三路连接保持打开 | 1 / failed，保留原因与最终统计 | 1.04 秒 |
| 注入识别异常，三路连接保持打开 | 1 / failed，保留原因与最终统计 | 0.90 秒 |
| 模型加载失败 | 1 / failed | 0.68 秒 |
| 等待输入帧率时 `run_seconds=0.5` 到期 | 0 / stopped | 1.09 秒 |

耗时从调用实际 `main()` 到其返回/抛出退出统计，包含测试等待发起请求/故障/信号及收尾，排除子进程启动和 Python/Torch 导入；信号/故障行不是“注入后到退出”的独立延迟测量。

原有完整测试在禁用 GPU 后曾出现 4 项环境失败（`No CUDA GPUs are available` / Triton `0 active drivers`），分别是 XCLIP CUDA 前向、LightSR 构建、神经超分构建和 VideoMamba 构建。随后 CPU 回归明确排除这 4 项，不把它们计为通过。1 项 GPU 测试按原条件跳过。本次没有重跑真实 GPU 模型精度或三卡实时性能；原审查的 P1-3/P1-4 及 P2 项仍待处理。

## 证据与复现

- [镜像测试日志](p1-lifecycle-fix-20261014/image-tests.log)、[JUnit](p1-lifecycle-fix-20261014/image-evidence/junit.xml)。每个入口场景的日志、会话、事件日志和视频保存在 `p1-lifecycle-fix-20261014/image-evidence/pytest/`。
- [CPU 回归日志](p1-lifecycle-fix-20261014/cpu-tests.log)、[JUnit](p1-lifecycle-fix-20261014/cpu-regression.xml)。
- [会话与视频解码汇总](p1-lifecycle-fix-20261014/sessions-and-media.json)。
- [回归测试](../tests/test_v2_serve_lifecycle.py)、[入口测试驱动](../tests/v2_serve_probe.py)。本地运行：`.venv/bin/python -m pytest -q tests/test_v2_serve_lifecycle.py`。
- 镜像复验临时目录 `/tmp/darkpipe-p1fix-11s6xmnm` 保留独立解包副本和纯 Python pytest 依赖；它们只挂载到验证容器，不修改提交包或镜像。验证容器均使用 `--rm`，完成后已退出。
