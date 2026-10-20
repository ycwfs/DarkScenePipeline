# 最新算子包镜像性能确认（2026-10-15）

**镜像内容未变化，无需因本轮算子修复重新构建或上传镜像。最新算子包在本次可用的单卡降级
条件下，处理帧率与识别时延均未达标；默认 quality 还发生了实时写盘队列满载造成的弃段。**

本轮只测量和核对，没有修改交付包、权重、镜像或默认参数。镜像内运行的是真实增强、RGB 和
骨架模型，不是上一轮 CPU 替身回归。

## 镜像与算子身份

| 对象 | 本次确认 |
|---|---|
| 算子 | `platform/dark-demo-v2_fixed_20261014.tar.gz` |
| 算子 SHA256 | `78018a678d7c3a3ba5901eac164cf6546cc74574e21efad630706d7d93c864ce` |
| 镜像 | `darkpipe-operator:0.3.0` |
| 镜像 ID | `sha256:d359716620d7eb757cd7c36733e25642469450f1c801e0367fc99248f0a2f212` |
| 镜像归档 | `platform/darkpipe-operator-0.3.0.tar.gz`，6,796,727,259 字节 |
| 镜像归档 MD5 | `106ff7b1b309345acca0ddab1c502a65` |
| 镜像归档 SHA256 | `a562d53d7bc588aded2d146939a634cf3e0a78afd6dfb8c5f53692c6afec2b77` |

镜像 ID 和全部文件系统层与 10-14 原包审查记录一致；完整扫描归档并校验其中 **26 个 OCI
数据块**，归档内镜像描述及 rootfs 与实际加载的镜像匹配。两次运行后再次核对镜像 ID，未变。
算子归档解包到独立临时目录只读挂载，45 个文件哈希全部匹配随包 manifest；测试结束后归档
哈希仍一致。

因此，若平台已经使用此前上传的这份 0.3.0 镜像，**保留原镜像，仅更新算子包并重新导入
`suanzi.json` 即可**。本地验证没有连接平台镜像仓库，未检查远端是否将同名标签指向其他镜像。
构建脚本默认标签的修改没有重建现有镜像。

证据：[身份对比](performance-confirmation-20261015/identity.json)、
[归档完整性](performance-confirmation-20261015/image-archive-integrity.json)、
[镜像检查](performance-confirmation-20261015/image-inspect.json)。

## 测量条件与边界

- 清单要求三张 GPU，但本机只有 **物理 GPU 1（RTX 3090 24 GiB）空闲**。GPU 0、2、3、4、5
  有其他用户任务，本轮没有使用、停止或调整这些任务。前后占用见
  [测试前](performance-confirmation-20261015/gpu-before.txt) /
  [测试后](performance-confirmation-20261015/gpu-after.txt)。
- 容器只暴露物理 GPU 1，映射为 `cuda:0`。传入清单默认 `gpu_ids=0,1,2`，由算子正常降级为
  `gpus=0`。这不是默认三卡验收；**当前不能确认三卡配置是否达标**。
- 两次均使用清单 CPU limit=8、memory limit=12 GiB。没有额外设置 Torch/ORT/OpenCV 线程
  环境变量，也没有注入调优启动器。宿主机其他任务仍在运行，属于共享环境实测。
- 同一真实 `喝水16.mp4`，1920×1080、75 帧，经本机 HTTP MJPEG 以 25 fps 循环发送。
  属于动作密集的短片循环压力场景，不代表整夜摄像头或所有动作的分布。
- Retinexformer、VideoMamba pn、RTMO-s+ST-GCN 实际加载，骨架两次都启用；默认 alpha 0.7、
  soft、照度门 75、阈值 0.65、处理长边 840、bicubic×2、画面 fast、片段前后各 2 秒。
- 每次 `run_seconds=120`，SSE 连接保持到服务结束，另抓取 MJPEG 图片及 10 秒 FLV。
  默认运行保持 `clip_denoise=quality`；诊断对照只改 `clip_denoise=fast`。
- 统计 20–120 秒内 `fps_proc>0` 的健康快照（约每秒一次）；不是逐帧独立样本。识别时延
  指窗口最后帧在算子内采集到本机 SSE 收到事件，包含处理、排队和发送，不包含完整 1 秒
  窗口的积累时间，也不包含摄像头到算子采集之前的延迟。由 uptime 对齐时钟，受 0.1 秒
  舍入约 ±50 ms 及请求耗时影响。FLV 文件头的 15 fps 是出流节拍，不作为处理速度。

## 实测结果

| 指标 | 默认 quality | fast 诊断对照 |
|---|---:|---:|
| 处理 fps 中位数 | **8.36** | **6.42** |
| 处理 fps p95 / 最大 | 14.80 / 15.70 | 15.43 / 16.04 |
| 达到 15 fps 的健康快照 | 2/94 | 8/96 |
| SSE 识别时延中位数 | 0.95 秒 | 1.36 秒 |
| SSE 识别时延 p95 | **2.29 秒** | **5.24 秒** |
| SSE 识别时延最大值 | 2.61 秒 | 6.02 秒 |
| 识别时延 ≤1 秒的事件 | 14/24 | 5/19 |
| 最后画面延迟快照 p95 | 473.8 ms | 198.7 ms |
| 完成片段数 | 10 | 14 |
| 弃段计数 / 片段录制丢帧数 | 2 / 70 | 0 / 0 |
| 失败片段 / 写入错误 | 1 / 1 | 0 / 0 |
| 最终会话 / 退出码 | failed / 1 | stopped / 0 |
| 总墙钟（含启动及收尾） | 153.4 秒 | 142.2 秒 |

两次都未达到 ≥15 fps、识别时延 ≤1 秒。fast 对照完成了录制，没有弃段、丢帧或写盘错误，
但**没有取得整体识别速度改善**。共享主机、单轮串行对照和不同推理/事件时序，不足以将
两次识别时延差异归因为一个参数；不能把 fast 当作已验证的达标方案。

quality 的实际日志显示：实时有界队列持续满 10 秒，放弃一条片段结束消息；随后持续满
3 秒，放弃一次开录。收尾发现缺失的结束消息，明确记录失败并退出 1；未发生 30 秒后
误报保存成功。弃段和失败片段计数有重叠，不可相加当作不同片段。最终已接收的任务排空，
`writer_alive=false`、`clips_unfinished=0`，但仍有 `clips_failed=1`。

这暴露的是满功能负载下未解决的性能/队列容量问题；此前 CPU 回归及单片段 quality 排空
通过，不能替代持续真实模型负载的验收。默认 quality 在本次单卡条件下不宜按“已验收”提交。
三卡资源空闲后仍需按相同 CPU/内存和默认功能重新实测，不能拿旧双卡数据或本次单卡结果替代。

默认运行的 10 个、fast 对照的 14 个已完成 MP4 全部验证为 H.264，FFmpeg 完整解码无错误，
同名 JSON 的相对 `video` 路径有效；这不包含被弃掉的动作片段。

原始证据：[汇总](performance-confirmation-20261015/comparison.json)、
[默认日志](performance-confirmation-20261015/single-default-r1-persistent-sse/container.log)、
[默认命令](performance-confirmation-20261015/single-default-r1-persistent-sse/command.json)、
[默认会话](performance-confirmation-20261015/single-default-r1-persistent-sse/session.json)、
[fast 日志](performance-confirmation-20261015/single-fast-diagnostic-persistent-sse/container.log)、
[fast 命令](performance-confirmation-20261015/single-fast-diagnostic-persistent-sse/command.json)、
[fast 会话](performance-confirmation-20261015/single-fast-diagnostic-persistent-sse/session.json)。

测试脚本在 [scripts/live_test.py](performance-confirmation-20261015/scripts/live_test.py)，
统计脚本在 [scripts/summarize.py](performance-confirmation-20261015/scripts/summarize.py)。
复跑前必须重新检查 GPU 占用，只使用空闲卡。
