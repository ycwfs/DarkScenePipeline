# 验证脚本复现说明

这些是本次实际使用的验证脚本副本。脚本从原归档提取的代码运行，不依赖仓库根目录的旧 `darkpipe/`。实际数据路径仍为 `/ssd1/wfs/project/lowlight/DarkScenePipeline`，A缓存在仓库外 `../compare/behavior/cache/realact_rf_v10_full/val`。重新运行会消耗GPU，先检查GPU0、1是否空闲。

从项目根目录创建独立验证目录：

```bash
audit_dir=$(mktemp -d /tmp/darkpipe-review.XXXXXX)
cp reports/submission-audit-20261014/scripts/*.py "$audit_dir/"
mkdir -p "$audit_dir/package"
tar -xzf platform/dark-demo-v2.tar.gz -C "$audit_dir/package"
docker load -i platform/darkpipe-operator-0.3.0.tar.gz
```

`live_test.py` 在宿主机用OpenCV读真实素材并提供25fps本地流；启动原清单构造的容器，限制CPU8/内存12Gi，并收集健康快照、SSE和视频。端口为18764和18043，使用host网络。一次只运行一个实时驱动：

```bash
# 默认参数与持续SSE：复现run_seconds无法退出的问题。
.venv/bin/python "$audit_dir/live_test.py" --sse-seconds 0
# 参数对照：片段off、部分线程环境限制；35秒后主动断开SSE。
.venv/bin/python "$audit_dir/live_test.py" --clip-denoise off --thread-caps
# 诊断启动器：额外限制ORT/torch/cv2，保留片段quality。
.venv/bin/python "$audit_dir/live_test.py" --clip-denoise quality --thread-caps --ort-tuned
```

持续SSE会触发已知退出故障，驱动设置了195秒运行上限，超时后停止自己创建的容器。`ort_launcher.py` 只属于诊断对照，不是提交包中的代码。三次精确容器参数数组见各输出目录 `command.json`。

全量ABCP回放在镜像内运行，必须同时挂载原包、验证目录、现场数据和仓库外的A缓存。参考本次命令：

```bash
docker run --rm --gpus device=0 \
  -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=4 \
  -e MKL_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=1 \
  -v "$audit_dir/package/dark-demo-v2:/opt/darkpipe/op:ro" \
  -v "$audit_dir:/audit" \
  -v /ssd1/wfs/project/lowlight/DarkScenePipeline:/ssd1/wfs/project/lowlight/DarkScenePipeline:ro \
  -v /ssd1/wfs/project/lowlight/compare/behavior/cache/realact_rf_v10_full/val:/ssd1/wfs/project/lowlight/compare/behavior/cache/realact_rf_v10_full/val:ro \
  -w /opt/darkpipe/op darkpipe-operator:0.3.0 \
  /opt/conda/envs/darkpipe/bin/python -u /audit/native_suites.py
```

全量回放输出 `native-windows.jsonl` 和 `native-summary.json`，只保留窗口用到的帧并以batch8增强；只跑C可在脚本末尾追加 `--suites C`。该回放验证采样窗口和分类，不计实时帧率。

`offline_test.py` 使用原清单跑4段C视频批处理、暗光单视频及短输入多卡回退，使用宿主GPU1，复制交付到两个验证目录。它与实时性能测试应分别运行。`shard_probe.py` 应在 `offline_test.py` 结束后运行，会把300帧输入送入真正的多卡分片，预期记录缺CLI的退出1。

`media_verify.py` 应在上述离线和实时输出都生成后运行；若使用了持续SSE的新增目录后缀，需相应补充目录枚举。它对已落盘的视频完整解码并检查元数据、源帧数/帧率及交付副本。

`failure_probe.py` 和 `clip_contract_probe.py` 用原包代码注入可控故障，验证错误上报与写盘契约。后者会等待30秒关闭超时，再主动解除阻塞让自己的写线程收尾。它们本次在项目 `.venv` 中运行，不需要GPU。

保留的证据均是原始日志或结构化结果。报告中的实时延迟口径统一限制20–75秒，默认驱动的早期summary包含75秒后的事件，最终比较使用 `evidence/live-comparison.json` 中重新筛选的数值。
