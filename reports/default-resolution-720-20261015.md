# 默认处理长边调整为 720（2026-10-15）

已按用户要求将 v2 的 `proc_max_side` 默认值从 **840** 改为 **720**，CLI、平台清单、
帮助文字与交付文档同步。该参数控制增强前的处理长边，不是最终出流高度。

新包：[dark-demo-v2_720_20261015.tar.gz](../platform/dark-demo-v2_720_20261015.tar.gz)。
保留此前修复包与原始包，不覆盖旧交付物。更新后须重新导入 `suanzi.json`；旧任务若显式
填写 840，应改为 720，并确认启动日志 `[config]` 中的实际值。

| 交付项 | 核验值 |
|---|---|
| 归档大小 | 190,636,483 字节 |
| MD5 | `ad2aa5e8e1a4d819408685f6b8d4f047` |
| SHA256 | `0bc3b0636c59e69f6b2728cbd1fab265b799ab7d4acd009eb4b39727814241c9` |
| 镜像 | `darkpipe-operator:0.3.0` |
| 镜像 ID | `sha256:d359716620d7eb757cd7c36733e25642469450f1c801e0367fc99248f0a2f212` |

归档共 45 个文件，独立解包后逐项核验 SHA256；相比 `dark-demo-v2_fixed_20261014.tar.gz`，
仅 `main.py`、`suanzi.json`、`README.md`、`CLAUDE.md` 改变。算法文件及全部 7 份权重
未变，片段前后各 2 秒、识别阈值 0.65、`clip_denoise=quality` 等默认保留。
镜像 ID 与镜像归档的大小、修改时间、状态变更时间在验证前后相同；未重建镜像。
可继续使用此前上传的同一份 0.3.0 镜像，仅需更新算子包及平台清单。

验证使用独立解包副本，在原镜像内只读挂载、禁用 GPU、限制 CPU 8 / 内存 12 GiB：

- `platform/package_v2.py --check` 通过，入口、清单引用/默认值/选项、源码语法与权重齐备。
- `tests/test_v2_delivery.py` **19 项通过**，覆盖参数契约、2 秒片段覆盖、阈值与打包等回归。
- 实际入口 `--help` 与编译通过，解析默认值为 720。
- 使用包内 `DownscaleStage`、`BicubicStage`、`append_label_bar` 和 `VideoWriter`，
  对合成 1920×1080 帧验证尺寸链：**720×404 → 1440×808 → 1440×874**。
- 生成的 15 帧视频经 ffprobe 确认为 H.264 / yuv420p、1440×874，FFmpeg 全片解码通过。

这是默认值、尺寸与编码兼容性验证，**没有重新运行真实模型性能或识别精度评估**。
降低处理尺寸减少像素量，也可能损失细节或影响姿态可见性；不能据此宣称达到 ≥15 fps、
识别时延 ≤1 秒，或解决此前持续负载下的片段队列满载问题。
此前 840 配置的实测保留在 [性能确认报告](performance-confirmation-20261015.md)，不作为
此次 720 配置的新测量。

证据：[身份与哈希](default-resolution-720-20261015/identity.json)、
[包间差异](default-resolution-720-20261015/package.diff)、
[回归日志](default-resolution-720-20261015/pytest.log)、
[尺寸与编码结果](default-resolution-720-20261015/dimensions.json)。
