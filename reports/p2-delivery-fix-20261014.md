# 片段默认值与 P2-1 / P2-2 / P2-6 修复

本轮在 P1 修复基础上完成用户指定的三项修复，并把 v2 平台与命令行的片段前后默认时间统一为 2 秒。
功能回归通过；额外的全尺寸片段压力测试仍触发已有的 P1-4 写盘退出上限，因此本报告不把默认
`quality` 在高负载停止时的片段完整性判为通过，也不宣称已达到三卡实时性能指标。

## 交付与范围

新包：[返回 platform 的交付归档](../platform/dark-demo-v2_p1p2fix_20261014.tar.gz)。原包及 P1 包保留，未提交 Git、未部署平台。

| 项目 | 值 |
| --- | --- |
| 归档大小 | 190,634,724 字节 |
| MD5 | `bace63a86d7b79e66489a4c0932cf067` |
| SHA256 | `9b4fe380dd38afbf44d1949ace0ce8cf72db8613dbdcb90a47c5179b18065894` |
| 外层目录 / 文件数 | `dark-demo-v2/` / 44 |
| 验收镜像 | `darkpipe-operator:0.3.0` |
| 镜像 ID | `sha256:d359716620d7eb757cd7c36733e25642469450f1c801e0367fc99248f0a2f212` |

相比 P1 包，仅 `main.py`、`darkpipe/clips.py`、`suanzi.json`、`README.md`、`CLAUDE.md` 内容变化。
7 份权重逐文件 SHA256 相同，识别、融合、增强模型代码及此前 P1 生命周期修复保留。
[归档对比](p2-delivery-fix-20261014/package-integrity.json)、
[逐文件哈希](../platform/dark-demo-v2_p1p2fix_20261014.tar.gz.manifest.json)。

## 修复行为

- **片段时间**：`clip_pre_sec=2.0`、`clip_post_sec=2.0`，底层本已有该默认值，本轮修正覆盖它的
  `main.py` 与平台清单；`clip_max_sec=15.0`。单次触发测试在时间 3 秒命中，保存约 1–5.1 秒的
  片段。行为切换、输入结束和停止可以提前收尾，仍不保证固定最短片长或任意识别延迟下的动作覆盖。
- **P2-1**：编码器打开之后的每个新帧也先 `_clean()` 再重采样。预卷、开头缓冲、后续帧均执行
  指定去噪；重复帧和尾部填充复用已清理帧，不重复处理，也不修改实时出流共享的原帧。
- **P2-2**：CLI 默认 `reco_min_conf` 从 0.5 改为平台使用的 0.65。边界回归用相同概率验证：
  0.60 报 other、0.70 报 wave；不改变识别模型权重。增加 v2 清单/CLI 全部默认值、选项、
  参数引用和类型转换的检查，打包时同样强制执行。
- **P2-6**：新增 `platform/package_v2.py`，直接打包 v2 自己的源码、字体和 7 份权重；
  支持含外层目录的 tar.gz、入口在根目录的 zip，排除缓存和运行产物；固定归档元数据，并生成
  SHA256 文件清单。缺权重、参数漂移或 active checkpoint 未入包会拒绝构建。旧脚本明确只维护
  `op_*`，传入 v2 会给出正确新命令。`build_image.sh` 默认标签对齐 0.3.0；文档补全权重来源、
  解包层级及 v2 的 `ckpt_dir=ckpts`，避免示例误用镜像里的旧权重。

## 验证

- 旧 P1 包运行新增针对性回归：**5 失败 / 5 通过**，三个长片段去噪模式、前后留存时间和默认阈值
  各复现目标问题；证据 [before-regressions.log](p2-delivery-fix-20261014/before-regressions.log)。
- 本地 CPU 回归：**305 通过、1 跳过、4 deselected，28.53 秒**。4 项分别为 XCLIP CUDA 前向、
  LightSR、神经超分构建、VideoMamba 构建，明确排除，未计为通过。
- 最终归档解包后，在指定镜像内运行 **19 项交付回归 + 22 项 P1 生命周期回归，41 通过，63.09 秒**；最终结果见
  [镜像日志](p2-delivery-fix-20261014/final-image-regression.log) 与
  [JUnit](p2-delivery-fix-20261014/final-image-regression.xml)。容器 CPU=8、memory=12Gi，禁 GPU，
  网络只用容器内回环。模型/采集为 CPU 替身，HTTP、FFmpeg、切片与会话写入是真实路径。
- 缺权重、错误默认值/choices/flag 引用/现役权重映射均验证为拒绝出包；tar/zip 两次构建的哈希一致。
  完整真实权重归档也重新构建比较，见 [reproducibility.json](p2-delivery-fix-20261014/reproducibility.json)。
- 最终镜像回归产出的 7 段非空视频均为 H.264，完整解码无错误，见 [final-media.json](p2-delivery-fix-20261014/final-media.json)。
- Python 编译、shell 语法、diff 空白检查通过。无新增依赖；未重建镜像，Docker 构建脚本的标签选择
  用替身 docker 命令验证，不能据此声称新的 Dockerfile 构建已经完成。

## 全尺寸去噪成本与尚存限制

使用原审查保存的真实出流截图 `live-quality-stream.jpg`（1680×1020），60 帧、目标 15fps、
实时有界队列、前后各 2 秒；同一 0.3.0 镜像，CPU limit=8 / memory=12Gi。为隔离单帧去噪成本，
探针显式固定 OpenCV 1 线程，未同时运行增强/识别。这不是默认 OpenCV 线程设置或完整 GPU 服务的
吞吐测试，不能外推为三卡端到端帧率。数据见 [初次探针](p2-delivery-fix-20261014/clip-cost.log) 与
[确认探针](p2-delivery-fix-20261014/clip-cost-confirm.log)。

| 条件 | 清理帧数 | 单次清理中位数 | close 返回时已完成片段 | 结果 |
| --- | ---: | ---: | ---: | --- |
| 旧 P1 包 quality | 21/60 | 921 ms | 1 | 后 39 帧漏去噪 |
| 新包 quality | 60/60（探针额外等待后） | 926 ms | 0 | **收尾完整性未通过** |
| 新包 fast | 60/60 | 58 ms | 1 | 完成、无丢帧 |

新包 quality 在开始投帧后约 34.72 秒返回 `close()`，写线程仍活着；探针单独追加等待，至约
62.87 秒完成编码、检查及解码。该额外等待只存在于诊断程序，**生产代码没有这个补偿**。
第一轮未做补偿时只留下尚未完成的暂存视频，ffprobe 失败。这复现了此前 P1-4 的收尾风险，并说明
整段质量去噪增加了遇到该风险的机会。确认探针对此返回退出码 1，未计入通过；fast 返回 0。
两者额外等待后的成品均为 H.264 且完整解码无错误。

本轮保留用户现有 `clip_denoise=quality` 默认，不以降级画质掩盖成本。需要降低写盘压力时可显式
设置 `clip_denoise=fast`；该单独写盘结果不能替代完整服务验收。P1-3/P1-4、P2-3/P2-4/P2-5 仍待处理。

## 复现与生效

从仓库根运行：

```bash
python3 platform/package_v2.py --check
python3 platform/package_v2.py --output platform/dist/dark-demo-v2.tar.gz
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q tests/test_v2_delivery.py
```

镜像验证使用最终归档解包目录 `/tmp/darkpipe-p2fix-20261014-b0cgnwlo/final/dark-demo-v2`，通过
`DARKPIPE_V2_OPERATOR_DIR=/operator` 指向只读挂载的该目录，运行
`tests/test_v2_delivery.py tests/test_v2_serve_lifecycle.py`。pytest 依赖只读挂载到容器，没有修改镜像。

更新平台算子时必须**重新导入本包的 suanzi.json**；已有任务若显式保留 1 秒或 0.5，仍会覆盖默认值。
核对启动 `[config]` 中的 `clip_pre_sec=2 clip_post_sec=2` 与 `reco_min_conf=0.65`。
