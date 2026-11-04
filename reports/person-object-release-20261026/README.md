# 四参数默认与逐人物体先验正式接入 · 2026-10-26

用户接受联合调优参数并确认同时接入、验证和更新交付包。本轮未训练或替换原有五个主模型，新增交付已评估的独立七类物体 YOLO26m ONNX。

| 默认参数 | 值 |
|---|---|
| RGB／骨架权重 | **0.972／0.028** |
| 原图照度门 lum_gate | **35** |
| 动作 reco_min_conf | **0.925** |
| 物体 object_min_conf | **0.581** |
| person_object_prior | **true** |
| 处理长边／片段上下文 | 720；前后各 2 秒，15 秒重叠分段 |

CLI、平台输入清单／容器参数、PipelineConfig、逐人融合绘制保存、Box 和 models.json 均同步。必须重新导入 suanzi.json，并修改已有任务显式传入的旧参数。

## 可编辑流程图与交付

- [原生可编辑 PPT／draw.io、PDF 和预览](../algorithm-flow-20261026/README.md)：第 1 页关键算法及默认参数，第 2 页模型训练状态和来源；无栅格图片，文字编辑后保存／重开验证通过。
- [平台 TAR](../../platform/dark-demo-v2_person_object_20261026.tar.gz)／[平台 ZIP](../../platform/dark-demo-v2_person_object_20261026.zip)
- [NVIDIA Box ZIP](../../NVIDIA-Box_person_object_20261026.zip)
- [视频与动作片段检查页](index.html)、[未经裁剪的外部输出流录像](full-output-stream.mp4)、[五分钟输入](input-multiaction-5min.mp4)

默认别名 dark-demo-v2.tar.gz、dark-demo-v2.zip、NVIDIA-Box.zip 与日期包一致；旧日期包保留。原 darkpipe-operator:0.3.0 镜像 ID 和 6.4 GiB 归档哈希与旧记录相同，无需重新上传镜像。新增 ONNX 随代码包 ckpts/ 交付，不新增镜像 Python 依赖。

## 接入方式

物体先验位于 ActionEngine 的动作阈值之后，结果发布之前；单卡和多卡共用实现。仅喝水／捡物且原图亮度允许时执行：窗口 5 个均匀采样位置，720 增强图上的同人 30% 外扩区域＋可信姿态局部区域。保留原图尺寸，按原坐标生成裁剪后精确映射回增强图，沿用校准阶段几何。
喝水支持瓶／杯，捡物支持瓶／杯／手机／相机／纸／便携物；任一帧兼容物体分数≥0.581 即支持。有有效亮场观测而无兼容物体则报 other；低亮／默认门下原 L 未知／全无可用或绑定不明观测弃权。人体框与别人 IoU>0.4 的观测不参与支持。模型执行错误显式失败，不静默关闭先验。
过滤后 label/confidence/decision_scores 统一用于绘制、事件、保存；动作原概率保留，action_label/action_confidence 记录过滤前结果；object_prior 带原因、采样帧、ROI 和检测证据。否决的 other 独热分数是规则结果，不是新模型后验概率。
实时增强帧历史按 1.2 秒、120 帧、128 MiB 限界；任务仅带 5 帧引用，物体检测结果 LRU 512 个裁剪；重连与人物代次隔离，避免旧证据串人。默认物体模型在动作 GPU 中按需运行，不逐帧逐人无差别检测。

## 验证结果

| 核验 | 实测 |
|---|---|
| 源码定向 CPU 回归 | 67 通过 |
| 最终独立解包＋原镜像回归 | 28 通过；1 项 SciPy 对照测试未在镜像运行（源码已通过，不引入 SciPy 运行依赖） |
| 原镜像单卡真实视频 | 4 段／406 帧／32 事件／5 个片段；覆盖亮场支持、无物体否决、挥手直通、暗场绕过 |
| 独立 Box ZIP 的实际 run.py | 正常退出；2 个输出／片段完整解码 |
| 原镜像三卡实时输入 | 300.56 秒，7514 帧输入；实际处理主段 6993 帧 |
| 处理帧率 | 中位 23.41 fps，最低 17.05 fps |
| 采集到处理输出延迟 | P95 145.9 ms；最大 497.3 ms |
| 动作结果年龄 | P95 0.377 秒；最大 0.527 秒 |
| 输入重连／RTMP 重启／HLS 重启 | 0／0／0 |
| 主段健康状态 | {'ok': 234, 'degraded': 9} |
| 逐人事件／SSE 接收 | 503／503；日志丢弃 0 |
| 保存片段 | 39 段，元数据上下文完整 38 段 |
| 保存异常 | 写入失败／强制释放／未完成／录制丢帧／弃段均 0 |
| 增强帧历史缓存峰值 | 25.80 MiB（上限 128 MiB） |
| 输出与片段严格解码 | 58 个视频通过，另有 Box 输出 |
| 停止与清理 | SIGTERM 正常退出 0，会话 stopped，自建子进程清理完成 |

实时先验原因统计：`{"unaffected_action": 414, "single_frame_support": 49, "no_compatible_object": 30, "below_luma_gate": 10}`。无兼容物体否决：`{"pick_up": 14, "drink": 16}`；这些是实际过滤数量，未经逐窗真值审核，不称误报减少量。

主段 243 次健康采样中 234 次 ok、9 次 degraded；这 9 次均 HTTP200、采集/处理/推流/HLS 存活、画面年龄≤0.11 秒、无重启。原因均为出流瞬时写入阻塞，显示时长四舍五入为0.0秒；现有 health 对 blocked_s()>0 即降级（10-25 已记录）。完整外部接收、视频解码及输出更新间隔提供独立连续性证据，未改健康规则以求全绿；本轮不声称健康状态全程ok。
链路为原速 H.264/RTSP 输入 → 最终 TAR 独立解包＋原镜像真实六模型 → RTMP 主动推流 → 本机 MediaMTX → 外部 FFmpeg 录制。期间在 10/140/275 秒连接 MJPEG／FLV／HLS，各记录 2 秒；没有离线补帧替代漏处理帧，没有给展示画面添加黑白条。完整录像含 75 秒源预热的一部分及收尾，实际输入和素材时间索引可对照。处理输入采用低延迟取最新策略，输出 15 fps，不能称全部 7514 帧都经过模型。
首轮测试接收端在 RTMP 首帧发布前启动，收到 404；正式轮先等待推流可读，再启动录制。诊断在 diagnostic-recorder-startup/，算子正常停止，未通过改算子规避问题。原镜像不自带 pytest，回归仅挂载独立纯 Python 测试工具，不修改镜像。

## 边界与复现

此次是约五分钟本机回环、三张 RTX3090 与本地磁盘，单卡另做四段离线功能验证；不是实体 Box、平台 vGPU、现场摄像头、NFS／HDFS 或无限流长稳验收，也未重新验收单／双卡实时速度。未注入本轮断流故障，重连和并发边界由 CPU 回归及历史真实流测试提供补充证据。
前后 2 秒相对于识别事件，不能保证覆盖未识别到的真实动作；15 秒续段和视频起止会标记上下文不足。合成输入存在硬切、误识别及漏识别均保留。四参数是误报优先校准：新已知喝水误报 49→1/526，旧 other 18→10/1009；历史 test 真动作 39→34/51，唯一新真喝水仍过滤。此为此前固定窗口离线口径，不是本次流式准确率；精确实时绑定与此前新数据近邻绑定不同，不能保证同数事件。新9段用于阈值反馈校准，不是独立泛化集。

- scripts/functional.py：原镜像四段真实模型＋保存功能；functional/summary.json 和各视频 events.json。
- scripts/run_live.py、instrumented_operator.py：最终归档真实推拉流及只读元数据观察；live/protocol.json 保存实际启动命令。
- scripts/archive_cpu.py：独立解包在原镜像回归；pytest-source.log、pytest-image.log。
- scripts/check_box_entry.py：独立 Box 入口与视频解码；box-native/verification.json。
- scripts/analyze.py、make_review.py：最终事件／亮度／先验／缓存／录像校验与检查页。
- scripts/verify_delivery.py --publish：TAR/ZIP 字节载荷一致、66 份平台文件与 85 份 Box 文件（另加 manifest）、12 份原权重不变，发布对应别名。
- verification.json、video-verification.json、image-verification.json、archive-verification.json：核验结果与哈希。
