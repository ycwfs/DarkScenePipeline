# NVIDIA Box 本地完整部署包

版本 `20261022-multiperson-pn2`。直接在 Linux NVIDIA GPU 主机安装 Python 环境运行，无需 Docker、算子注册或平台服务。支持本地视频、目录批处理、在线流输入、多格式出流、动作片段与事件本地保存。

`runtime/` 的执行代码与 `platform/dark-demo-v2` 逐文件一致；`run.py` 只处理盒子路径、环境与启动参数。现役模型仍是 pn2，增强、识别、照度门、骨架绘制和录制收尾规则沿用平台。

多人更新：骨架分支激活时逐人显示动作类别与概率，包括其他。最多6个有效人物中分类概率
最高者的完整向量参与alpha融合（0.7×RGB+0.3×骨架），黄色骨架和星号标出该人物；
底栏是融合后并经过0.75阈值的结果，可以与人物标签不同。窗口内采用位置关联，不提供跨窗口身份ID。
`behavior_infer.py`独立工具也复用同一骨架实现。模型权重和依赖未改变，旧评测成绩不代表本次选择规则的精度。

## 1. 安装环境

已验证架构：Linux x86_64、Python 3.10、RTX 3090、torch 2.7.0+cu126。需要可用 NVIDIA 驱动，建议 560 系列或更新、支持 CUDA 12.6。CUDA 用户态库随 PyTorch 安装，不需要另装完整 CUDA Toolkit。Jetson/aarch64 尚未适配，安装器会明确拒绝该架构，不能直接使用这里的 x86_64 wheels。

Ubuntu 22.04 示例（系统依赖只需安装一次）：

```bash
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev gcc libc6-dev ffmpeg libglib2.0-0
unzip NVIDIA-Box.zip
cd NVIDIA-Box
bash setup_env.sh
```

安装器创建包内 `.venv`，按 `constraints.txt` 固定的实测版本安装 torch、CUDA 扩展、ORT GPU 和服务依赖，最后执行真实 GPU 自检。需要能访问 PyPI/GitHub；也可用 `BOX_MAMBA_WHL`、`BOX_CONV1D_WHL` 指向下载好的对应 wheel。`rtmlib` 必须 `--no-deps` 安装，避免其 CPU onnxruntime 覆盖 GPU 版。不要在这个环境追加 CPU onnxruntime 或另一份 OpenCV。

安装过程不会配置驱动或使用 sudo。可设置 `BOX_BASE_PYTHON=/path/to/python3.10`、`BOX_VENV=/path/to/venv`。使用自定义或已有环境时：

```bash
export BOX_PYTHON=/path/to/venv/bin/python
"$BOX_PYTHON" check_env.py
bash run.sh --help
```

默认 `run.sh` 使用包内 `.venv/bin/python`。直接执行该环境的 `python run.py ...` 也可以；入口自动配置 ORT 所需 cuDNN/cuBLAS/CUDA 库路径与可写 Triton 缓存。`check_env.py --files-only` 可在没有 GPU 依赖的情况下核验所有交付文件哈希。

## 2. 离线视频与目录

```bash
# 完整处理一次，保存整段结果、事件、动作片段，然后退出
bash run.sh --video_path /data/input.mp4 --output-dir /data/darkpipe-results

# 批量顺序处理目录中的视频
bash run.sh --video_path /data/videos --output-dir /data/darkpipe-results
```

`--output-dir` 等价于平台的 `--clip_dir`，不传时为包内 `outputs/`。它是结果目录，不是单个 MP4 文件名。输入文件/目录的判定、批处理规则与平台一致。文件输入不会循环，也不会启动 HTTP 服务或 RTMP 推流。

产物布局：

```text
结果目录/
├── sessions/<启动时间_PID_随机后缀>.json   # 状态、实际产物路径、计数、错误
└── <输入源标识>/<会话时间_PID>/
    ├── <源文件名>-处理后.mp4              # H.264，完整视频，带标签及符合条件的骨架
    ├── <源文件名>_events.json             # 事件文件（实际路径以 session JSON 为准）
    ├── <起点>-<终点>s-<动作>.mp4          # 触发动作的片段
    └── <片段同名>.json                    # 片段元信息/完整标记
```

多个输入视频时，会话下再按视频名分目录。可以显式传 `--session_json /data/session.json`，否则每次自动生成独立文件，避免覆盖旧运行记录。

## 3. 在线流输入和输出

```bash
bash run.sh --video_path 'rtsp://user:password@camera:554/live' \
  --output-dir /data/darkpipe-results --serve_port 8000

# 可选 HLS 与外部 RTMP 推送
bash run.sh --video_path 'rtsp://camera:554/live' \
  --output-dir /data/darkpipe-results --serve_port 8000 \
  --stream_formats mjpeg,flv,hls --rtmp_push_url 'rtmp://media-server/live/darkpipe'
```

支持 FFmpeg/OpenCV 能解码的 RTSP、RTMP、HTTP MJPEG/HLS 等流。有限长 HTTP MP4 会先下载并走离线。实时服务监听 `0.0.0.0:8000`，局域网其他机器使用盒子 IP 访问；端口须可达。

| 接口 | 用途 |
|---|---|
| `http://盒子IP:8000/stream` | MJPEG，默认开启，可直接浏览画面 |
| `http://盒子IP:8000/live.flv` | HTTP-FLV，默认开启；VLC/ffplay 或支持 FLV 的网页播放器 |
| `http://盒子IP:8000/hls/index.m3u8` | 可选 HLS，有分段缓冲延迟 |
| `http://盒子IP:8000/events` | SSE 识别事件 |
| `http://盒子IP:8000/health` | 健康状态、处理/出流/片段统计 |
| `http://盒子IP:8000/config` | 实际运行配置 |

HTTP 输出由客户端拉取。`--rtmp_push_url` 才是主动推往外部媒体服务器；本包不提供 RTSP/RTMP 接收服务器。实时模式默认不录制全天完整视频，只保存触发动作的片段到 `<结果目录>/<源>/<会话>/<动作>/`，并持续写同会话的 `events.jsonl`。

默认运行到 Ctrl+C/SIGTERM；验证时可以加 `--run_seconds 60`。服务会等待片段收尾，再将会话状态写为 stopped；真实处理/写盘错误仍以非零状态退出并记录 failed。网络重连和收尾修复已保留，不承诺无限时长零故障。不要用 `kill -9` 代替正常停止。
处理线程结束推理及释放模型最多等待 30 秒；片段写盘只要仍有进展便继续排空，连续 60 秒无进展则失败，因此质量去噪积压时总收尾可能超过 30 秒。

实时录制队列有上限。quality 去噪在 CPU 上耗时较多，过载时可能丢弃录制帧或放弃尚未开录的片段；计数见 `/health` 和会话 JSON 的 `frames_dropped`、`clips_abandoned`。停止时尚未开录的片段也按过载计数告警，已开录片段继续完整收尾。需降低录制负担时可显式设置 `--clip_denoise fast`；默认仍与平台保持 quality。

## 4. 默认方案与 GPU

```text
原始帧 → 长边缩至720 → Retinexformer增强 → VideoMamba pn2
          原图照度 <75：只用RGB；否则RTMO + ST-GCN，α=0.7融合
       → 最终阈值0.75 → 去噪/饱和度/2倍bicubic → 逐人骨架与动作 + 融合结果底栏
       → 在线出流或离线整段视频，同时保存事件和动作片段
```

骨架只在骨架分支实际参与、最终为具名动作且存在匹配人物时显示；开头还没积累识别窗口时不显示，Other 和照度门挡下时也不显示。片段前后各 2 秒，最大 15 秒，默认忽略 Other，片段去噪 quality。1080p 输入默认输出约 1440×874（含底部标签），720 是处理画面的最长边，不是最终推流高度。输出无音频。

默认 `--gpu_ids 0`，编号相对于 `CUDA_VISIBLE_DEVICES`。多卡示例：

```bash
bash run.sh --video_path 'rtsp://camera/live' --gpu_ids 0,1,2
CUDA_VISIBLE_DEVICES=2 bash run.sh --video_path /data/input.mp4 --gpu_ids 0
```

在线多卡：最后一张负责识别，其余增强；单卡串联运行。离线启用片段时只用第一张，以保留完整连续时间轴。模型和算法默认与平台一致；GPU 默认数量根据本地部署改为一张。FPS/时延需要在实际盒子另行验收，本包不承诺达标。

所有平台参数均可经 `--help` 查看，仍使用下划线写法，如 `--reco_min_conf 0.75`、`--pose_overlay true`。盒子默认不设置平台内部 hosts、不上传 HDFS；需要时可显式配置。可选 `--fuse_mode max_prior` 与 YOLO 权重一并提供，但默认继续使用已验证的 alpha 方案。

## 5. 目录、维护与兼容

- `run.sh` / `run.py`：完整本地流程入口。
- `setup_env.sh` / `requirements.txt` / `check_env.py`：环境安装、自检。
- `runtime/`：平台完整执行代码与字体，同源同步；修改应先在平台修复，再通过仓库 `python3 scripts/package_nvidia_box.py --sync --output NVIDIA-Box_20261021.zip` 同步、校验并打包。
- `weights/`：平台的 8 份权重，另保留旧盒子 crop 文件名作为兼容，共 9 个文件；现役 pn2 的 SHA256 为 `d30b36c4c1ca9ab276442b4d0544b3d705f4ead68c7dcb5a721b696f9ee5f362`。
- `manifest.json`：版本、文件哈希、平台来源映射；`VERIFICATION.md`：本次实际验证与限制。
- `behavior_infer.py`、`lowlight_infer.py`、`models/`、`check_models.py`：保留旧单模型工具供研究，使用说明见 `MODEL_TOOLS.md`。生产连续推理请用 `run.sh`。

ZIP 已含代码、权重、字体与公开样例，环境依赖在安装时获取。无需重新上传或构建平台镜像；本次只更新盒子交付。
