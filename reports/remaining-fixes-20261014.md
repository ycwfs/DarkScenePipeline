# 剩余审查问题修复与交付验证（2026-10-14）

本轮于 10 月 14 日开始，10 月 15 日完成；归档文件名沿用开始日期。

本轮完成原审查中的 P1-4、P2-3、P2-4、P2-5，并补齐 P2-6 的元数据路径和文档问题。
此前 P1-1/P1-2、P2-1/P2-2/P2-6 打包修复全部保留。用户明确排除的 P1-3（速度不达标）
没有进行调优；本报告不能作为 ≥15 fps / 识别时延 ≤1 秒的达标证明。

## 交付对象

- 新包：[dark-demo-v2_fixed_20261014.tar.gz](../platform/dark-demo-v2_fixed_20261014.tar.gz)，
  190,636,069 字节，45 个文件，保留 `dark-demo-v2/` 外层目录。
- MD5：`c1083152a6ae8f5f84656634ba64354c`。
- SHA256：`78018a678d7c3a3ba5901eac164cf6546cc74574e21efad630706d7d93c864ce`。
- [逐文件清单](../platform/dark-demo-v2_fixed_20261014.tar.gz.manifest.json)。两次完整打包 SHA256 相同。
- 指定镜像：`darkpipe-operator:0.3.0`；ID
  `sha256:d359716620d7eb757cd7c36733e25642469450f1c801e0367fc99248f0a2f212`，
  即此前从 `platform/darkpipe-operator-0.3.0.tar.gz` 加载的镜像；未重建。
- 测试对象为新包的独立解包副本，只读挂载到 `/operator`。45 个归档文件与源码、解包副本逐字节一致。
  原包和前两轮修复包全部保留，未替换原 `dark-demo-v2.tar.gz`，未提交 Git 或导入平台。

证据：[完整性检查](remaining-fixes-20261014/integrity.json)、
[归档检查](remaining-fixes-20261014/archive-review.json)。

## 修复结果

| 问题 | 本轮行为 | 主要验证 |
|---|---|---|
| P1-4 慢写提前完成 | `close()` 等待仍在推进的写盘任务排空；连续 60 秒无进展或写盘异常时抛出明确错误。会话统计失败/未完成片段；离线失败不交付、不删除工作目录，批处理立即中止 | 慢写超过等待阈值仍排空；停滞、去噪/编码/收尾/移动失败；真实单文件、批处理和实时入口的 failed/非零退出 |
| P2-3 缺失分片入口 | 新增随包 `darkpipe.shard_worker`，通过 JSON 传递完整 PipelineConfig；禁用片段才分片。逐段校验解码帧数后合并，子进程失败清理，日志不再堵 PIPE | 300 帧实际双子进程、偏移读取、输出尺寸；识别事件帧号/时间平移；分片短缺拒绝合并；启动中途失败终止已有子进程 |
| P2-4 检测器设备/异常 | CUDA provider 显式传 `device_id`；请求 CUDA 时若回退 CPU 就关闭检测先验。创建、预热、计时、运行异常都遵循降级约定 | cuda:2/cuda/cpu 配置；session 创建、provider 回退、预热、计时、运行与额外 warmup 故障注入 |
| P2-5 评估漂移 | 复用生产 `SkeletonAux.fuse`、`reject_probs`、`gate_luma`。默认 soft/门75，全部概率及亮度全精度保存，额外保留未门控向量；扫门拒绝旧舍入 JSONL | 无人 soft/rgb/other、presence 0.5 两侧、拒识 0.65 两侧、亮度 75 两侧，以及先验否决 +1e-6 的 JSON 往返 |
| P2-6 遗留说明 | 片段 sidecar 的 `video` 改为同目录相对文件名；清单、CLI 帮助和文档对齐三卡架构，去掉旧速度达标保证；pn 蒸馏系数修正为 0.2 | 移动后相对路径可解析；打包契约检查；读取现役权重元数据 |

写盘完成以同名 JSON 为标记；失败的暂存文件不会计入 `clips_saved` 或调用上传回调。
新增 `clips_unfinished`、`clips_failed`、`write_errors`、`write_error`、`writer_alive`、
`recovery_stage_dir` 等统计。离线失败会话另有 `recovery_dir`，方便找到保留的工作目录。
恢复文件仍受容器存储生命周期影响，不能视为容器销毁后也可用的持久备份。
实时上传原有的“失败告警、本地文件保留”约定未改；`uploads_pending` 表示尚未完成的上传任务。

分片保持原有的独立识别窗：各段边界需重新积累窗口，不能保证与单卡事件序列逐位相同。
可选 max_prior 的故障降级不会修复现场小物体漏检或 max 选高分带来的模型误报；默认仍为 alpha。

## 验证范围与证据

- 本地 CPU 回归：**355 通过、1 跳过、4 项 CUDA 测试排除**；随后补充的两项分片事件/启动失败回归也通过。
  [日志](remaining-fixes-20261014/cpu-tests.log)、[JUnit](remaining-fixes-20261014/cpu-junit.xml)、
  [补充日志](remaining-fixes-20261014/shard-extra.log)。另有既有 22 项服务生命周期回归在本地通过。
- 最终归档在指定镜像中：**91 项完整回归 + 2 项补充分片回归通过**。覆盖真实 Uvicorn/HTTP、
  FFmpeg、文件交付、子进程和配置传递。容器禁用 GPU、无外网，CPU 限额 8、内存限额 12 GiB；
  神经网络和异常路径用 CPU 替身，CUDA provider 参数用可检查的 ORT 替身验证。
  [镜像日志](remaining-fixes-20261014/image-tests.log)、
  [镜像 JUnit](remaining-fixes-20261014/image-junit.xml)、
  [补充镜像日志](remaining-fixes-20261014/image-extra-tests.log)。
- 打包检查、源码回读、Python 语法检查及 `git diff --check` 通过。
- 7 份权重、Retinexformer、默认识别与骨架算法文件与原包 SHA256 相同；未训练、替换权重或
  改变默认模型决策。现役 pn 权重元数据为 `epoch=12`、`lwf=0.2`，
  [元数据记录](remaining-fixes-20261014/pn-metadata.json)。

本轮不重跑 GPU 四套件精度评估，也未连接真实平台摄像头/HDFS。上一轮原包真实模型结果见
[原始审查报告](submission-audit-20261014.md)：A 177/177、B 5/365 触发、C 18/18、P 2/460 触发。
A/B/C/P 都已进入 pn 训练数据，属于换装回归，不能据此宣称独立场景泛化或全部 9 类动作已验收。
旧 prior_eval JSONL 的舍入不可逆；新版精确扫门必须重新导出，不能用旧数据补上 schema 版本冒充。

## 全尺寸 quality 收尾复验

在同一指定镜像、CPU 8 / 内存 12 GiB 下，使用原审查保存的 1680×1020 输出帧，目标输入
15 fps、共 60 帧，OpenCV 显式单线程，实际执行 quality 去噪与 FFmpeg 编码。

- **通过，退出 0**：`close()` 自行排空后返回；已删除诊断脚本原来的额外 `join(60)` 补偿。
- 总耗时 **58.53 秒**，去噪 **60 次**，中位每帧
  **919.3 ms**，队列峰值 39。
- 返回时 `clips_saved=1`、`writer_alive=false`；未完成/失败/写盘错误/丢帧均为 0。
- 成品 H.264、60 帧、4 秒，FFmpeg 完整解码无错误。另对新包产生的默认片段、300 帧分片合并
  和含识别事件的分片合并输出完成独立解码，共核对 4 个成品，均为 H.264。

对照前轮同场景：旧 `close()` 约 34.72 秒就返回，`saved=0` 且写线程仍活着，依赖诊断脚本
额外等待才得到片段。当前修复消除了提前完成与后续清理造成的丢片风险；**58.5 秒处理约 4 秒
片段不是实时速度达标**，并未以此掩盖用户排除的 P1-3。

证据：[quality 成本/统计](remaining-fixes-20261014/quality/cost.json)、
[运行日志](remaining-fixes-20261014/quality.log)、
[成品解码核对](remaining-fixes-20261014/video-validation.json)。

## 使用说明

提交新包并重新导入其中的 `suanzi.json`，继续使用指定 0.3.0 镜像。
核对启动日志默认值：`clip_pre_sec=2`、`clip_post_sec=2`、`clip_max_sec=15`、
`reco_min_conf=0.65`、`fuse_mode=alpha`、`skel_fusion_alpha=0.7`、`skel_no_person=soft`、
`lum_gate=75`。已有任务显式填写旧值时，不会自动覆盖。

除用户保留的 P1-3 外，原审查编号中的代码/交付问题均已修复；模型泛化和现场性能仍需各自的实际验收。
