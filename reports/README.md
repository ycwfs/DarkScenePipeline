# 验证与实验记录

当前交付与模型结论见 [pn2 微调报告](feedback-finetune-20261019.md)；全局裁决见 [AGENTS.md](../AGENTS.md)。

| 记录 | 内容 |
|---|---|
| [提交前审查](submission-audit-20261014.md) | 原交付包的工程、模型和性能问题 |
| [生命周期修复](p1-lifecycle-fix-20261014.md) | 健康状态、失败传播和正常停止 |
| [片段与打包修复](p2-delivery-fix-20261014.md) | 前后各2秒、逐帧去噪、可复现打包 |
| [其余工程修复](remaining-fixes-20261014.md) | 写盘排空、离线分片、先验设备与评估一致性 |
| [性能实测](performance-confirmation-20261015.md) | 单卡真实负载；未达到性能要求 |
| [720默认尺寸](default-resolution-720-20261015.md) | 参数同步、尺寸与编码验证 |
| [会议材料](meeting-20261015/README.md) | 两部曲/三部曲流程、可编辑PPT和证据图 |
| [盒子同步](nvidia-box-verification-20261018.md) | 独立模型包与平台一致性 |
| [平台反馈修复](platform-feedback-20261019.md) | 片段标签竞争、收尾消息丢失、误报阈值 |
| [细阈值测试](platform-feedback-20261019-fine-threshold.md) | 0.70～0.80扫描和重复拥塞回归 |
| [骨架预览](pose-preview-20261019/README.md) | 0.75阈值、候选人物骨架、10样例验证 |
| [pn2微调与交付](feedback-finetune-20261019.md) | 新Other数据续训、按会话留出测试、交付验证 |

Git 保存复现脚本、报告、紧凑统计、逐窗概率和交付哈希清单。训练缓存、模型检查点、原始反馈、
解包副本、pytest临时目录、运行日志和预览视频留在本地，不纳入版本管理；`.gitignore`明确白名单。
报告中的这些链接是本地证据位置，新检出仓库中可能不存在。历史报告保留当时的参数和结果，
不要把不同阈值、尺寸、数据预处理口径直接混比。

算子模型需从对应版本的交付包恢复到`platform/dark-demo-v2/ckpts/`；盒子模型与示例需从
对应ZIP恢复到`NVIDIA-Box/weights/`和`sample_data/`，再按manifest核对。数据不随仓库发布，
复现实验前应依照各轮协议准备同一来源和拆分，禁止用测试会话重新选模型。
