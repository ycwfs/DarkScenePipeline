# Orin Nano 4GB Box优化候选

实测为单张RTX3090，不是Orin真机验收。完整模型和规则保留，优化缓存/调度/后端及720直接输出。

- [完整分析与逐项数字](analysis-report.md)
- [5分钟实际推流视频、全部动作片段、23样本](index.html)
- [统计边界](stats-appendix.md) · [图表及解释](figure-catalog.md)
- [主流验收JSON](live-final-verification.json) · [故障恢复JSON](fault-verification.json) · [短片验收JSON](demos-verification.json)
- [Box安装与真机测试说明](../../NVIDIA-Box/JETSON.md)
- [交付候选ZIP](../../NVIDIA-Box_Orin4GB_candidate_20261028.zip)

300秒平均实际处理10.57→12.78FPS（+20.9%）；health FPS中位11.37→13.93。未达到所有场景15FPS。29动作片段，28上下文完整，写失败0；故障测试另列。

已知精度代价：喝水27的5个原drink窗变3个，仍有动作片段；不是精度完全无损。4GB容量、ARM64扩展安装和真机FPS待设备端测量，包名保留candidate。历史FP16姿态目录trt-final-demos不是最终结果；以archive-demos和live-final为准。

交付SHA256：`b39805ea3d5a5f43c40681eca3de122ee71064216c2e8ee2280600d8f73b5c2a`；默认NVIDIA-Box.zip为同一文件。[交付与实测归档一致性](release-verification.json)。
