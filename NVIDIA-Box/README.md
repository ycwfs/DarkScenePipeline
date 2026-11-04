# NVIDIA-Box · Orin Nano 4GB 优化候选 · 20261028

目标设备为 ARM64 Orin Nano 4GB Super，CPU 与 GPU 共享 4GB 内存，使用 NVMe 保存结果。
本版已在单张 RTX3090 上完成独立解包测试；**尚未连接 Orin 真机，不能承诺板端内存足够或达到15FPS。**
当前 Jetson 安装分支面向 JetPack 6 / Python3.10 / TensorRT10+；实际 JetPack 版本尚未确认。

完整功能：照度增强、人体检测/跟踪、逐人 RGB＋骨架动作融合、物体/挥手/跑动先验、
人物框/骨架/物体框可视化、实时输入输出流、按人和动作并发保存完整画面片段。
六个主模型及现场权重保留，本轮没有重新训练或采用新轻量学生模型。模型训练来源见
[models.json](models.json)，Box 后端与派生 ONNX 说明见 [box-models.json](box-models.json)。

## 2026-11-02 低置信度候选复核

与平台包共用 `candidate_recheck_enabled=true`、`candidate_recheck_lower=0.85`。复核仅处理
通过既有先验且位于0.85–0.925的候选：同人高分动作续接（锚点1.2秒、相邻窗口0.65秒）和
跑动双窗口复核（速度0.35人体框高/秒、双窗跨度0.5秒）。固定值不作为额外启动参数暴露，
恢复事件会进入可视化、事件日志和按人物动作保存；未恢复候选仍为 `other`。

## 默认参数与优化

| 参数 | 默认值 |
|---|---|
| 模型处理长边 / 输出长边 | 720 / 720，非720p高度 |
| RGB融合权重 | 0.972 |
| 原图照度门 | 35 |
| 动作阈值 / 物体阈值 | 0.925 / 0.581 |
| 低置信度复核 | 开启；下限0.85（要求 0.5 < 下限 < 动作阈值） |
| 短动作显示保持 | 0.8秒；仅影响画面，`*`标记，不改事件或片段 |
| 骨架可视化 (`--pose_overlay`) | 默认开启；关闭会同时隐藏实时画面和保存片段中的骨架，不关闭骨架识别/融合 |
| 挥手先验 | 同一腕高于双肩，累计至少3个真实源帧 |
| 跑动先验 | 0.35人体框高度/秒；缺可靠姿态时保留原结果 |
| 片段前 / 后 | 各2秒 |
| 默认GPU | 单张可见卡0 |
| 动作批次上限 | 2，可用 `--box-action-batch 1` 降低峰值需求 |
| 后端 | ARM64 auto=TensorRT；x86_64 auto=ORT CUDA |
| CUDA Graph | 默认关闭 |

人体/七类物体检测使用 TensorRT FP16，姿态使用 FP32（关闭 TF32）。增强保持原 PyTorch FP16；
TensorRT增强候选因像素偏差未采用。RTMO仅做等价ONNX算子改写，参数没有重新训练。
engine 必须在目标设备构建，禁止复用3090生成的engine。

动作RGB缓存从512帧float32改成128帧uint8，缓存本体约525→33MiB；取批次后再做原归一化。
人体/姿态关联向量化。单卡实时动作独立线程，所有人公平轮转，同人待处理窗口可被较新窗口替换，
计入coalesced；离线保持源帧完整处理。该调度不保证所有短动作都会被模型识别；实时画面对刚确认
动作提供0.8秒显示保持，事件与片段仍遵循原始最终判定。

默认 `--sr off` 直接输出720长边，减少CPU编码负担；模型输入尺寸与原版相同。
若需要原1440长边显示，可加 `--sr bicubic --sr_scale 2`，性能需重新测量。

## 安装

### Jetson Orin ARM64

先阅读 [JETSON.md](JETSON.md)。保留本机JetPack的TensorRT/OpenCV，安装匹配JetPack的NVIDIA
ARM64 PyTorch、torchvision与Triton；不要安装普通x86 CUDA wheels。

```bash
bash setup_jetson.sh
```

脚本从源码编译Mamba/CausalConv1d的sm87扩展，默认MAX_JOBS=1，然后顺序构建engine并做真实模型自检。
此安装路径尚未在用户设备上执行。缺依赖时明确失败，不静默改成CPU或关闭模型。
可用 `BOX_VENV=/path/to/env` 指定已有环境，具体变量与板端测量步骤见JETSON.md。

### Linux x86_64 开发机

已验证环境为Python3.10、torch2.7.0+cu126、RTX3090；需要相容驱动和FFmpeg libx264/FLV/HLS。

```bash
sudo apt-get install python3.10 python3.10-venv python3.10-dev gcc libc6-dev ffmpeg libglib2.0-0
bash setup_env.sh
```

桌面安装器使用requirements.txt/constraints.txt及对应x86 Mamba wheels，默认ORT CUDA。
本轮TensorRT单卡实验显式指定 `--box-backend tensorrt`，需另装TensorRT10依赖。
不要把这套桌面环境直接复制到ARM64。安装器本身不执行sudo。

## 运行与输出

```bash
bash run.sh --video_path /data/input.mp4 --output-dir /data/results
bash run.sh --video_path 'rtsp://camera/live' --gpu_ids 0 \
  --output-dir /data/results --stream_formats mjpeg,flv,hls \
  --rtmp_push_url 'rtmp://media/live/darkpipe'
```

文件/目录输入离线处理一次，保存完整输出、逐人事件及动作片段。实时输入常驻服务，
默认不保存无限长的整段流；仅按事件保存动作片段。可用 `--run_seconds 300` 限定测试时长。
无音频。历史已增强回传切片才使用 `--input_processed true`，避免重复增强及误用原图照度。

实时接口：`/stream`、`/live.flv`、`/hls/index.m3u8`、`/events`、`/health`、`/config`。
默认不上传HDFS、不改hosts。会话状态在输出目录sessions/；失败会明确记录。

人物ID是跟踪ID，不是永久身份。按人物＋动作独立保存全画面，允许重叠；前后各2秒，
长动作按15秒上限重叠分段。保存侧等待1.2秒对齐异步事件，不给实时推流增加此等待。
EOF、断流、跟踪切换或分段边界可能使请求的上下文不足，片段JSON显式标记。
写盘失败仍明确上报，使用Ctrl+C/SIGTERM正常排空。`clip_denoise`默认off。

单卡共享全部模型；双卡是一前端一动作；实时≥3卡前两卡交替前端，其余动作。
Orin设备按单GPU使用，不要填写不存在的卡号。`--box-profile reference`可恢复原缓存/串行调度；
它仍使用所选Box后端，**不代表原版运行时逐字节回滚**。

## 本轮验证与限制

同一300秒25FPS输入，单张RTX3090，实际RTSP→RTMP并外部录制：

| 指标 | 原Box（1440输出） | 候选（720输出） |
|---|---:|---:|
| 实际处理帧数/7500 | 3170 | 3833 |
| 300秒平均实际处理FPS | 10.57 | 12.78 |
| health处理FPS中位数 | 11.37 | 13.93 |
| 采集交付后输出延迟P95 | 231ms | 181ms |
| 动作结果年龄P95 / 最大 | 0.910s / 1.792s | 0.864s / 1.760s |
| 主Python进程RSS峰值 | 4071MiB | 2893MiB |

平均吞吐约提升21%，是整体配置比较，包含输出尺寸变化。候选在最后拥挤场景中位8.02FPS，
仍未全面达15FPS；为保持延迟会丢旧输入帧，并替换285个待处理人物窗口。
29动作片段，28个上下文完整，写入失败/强制释放/写队列丢帧0。延迟不含相机网络缓冲。

107项CPU回归、真实入口和完整六模型前向通过。完整23样本与110秒受控断流恢复证据见
仓库 `reports/box-single-gpu-20261028/` 及 [VERIFICATION.md](VERIFICATION.md)。
这些是有限loopback/本地盘测试，不是无限流或板端保证。

已知数值代价：真实“喝水27”原5个喝水窗口在TensorRT候选变3个，仍保存动作片段；
人体/姿态ROI与物体检测阈值附近的变化造成额外过滤，不能声称识别完全无损。
RTX独立显存与主机RSS不能直接相加或等同Jetson统一内存。板端需测全部进程的内存、温度和持续帧率。

## 自检、测速与打包

```bash
.venv/bin/python check_env.py --files-only
.venv/bin/python build_engines.py
.venv/bin/python check_env.py
.venv/bin/python benchmark_box.py --video /data/test.mp4 --output-dir outputs/bench-file
.venv/bin/python benchmark_box.py --video rtsp://camera/live --duration 300 --output-dir outputs/bench-live
```

自定义环境可设置 `BOX_PYTHON=/path/to/env/bin/python` 运行run.sh。测速每次使用新的输出目录，
保存环境、会话、RSS、系统可用内存/交换和可用时的tegrastats；离线FPS不等于推流验收。

仓库打包命令：

```bash
python3 scripts/package_nvidia_box.py --sync --output NVIDIA-Box_Orin4GB_candidate_20261028.zip
```

runtime-overrides.json明确列出Box差异和平台基底哈希，打包不会静默覆盖优化，也会拒绝未经复核的基底变化。
交付不含3090 engines或实验依赖。平台代码包与原0.3.0镜像均不变；历史Box包保留供回滚。
`behavior_infer.py`/`check_models.py`是历史单模型工具，不代表当前逐人完整入口。
