# 逐人 VideoMamba 适配（2026-10-23）

已完成现场数据处理、逐人标注核对、8轮微调、固定测试集配对评估与10段新旧对照演示。
按验证集选择第7轮；阈值0.75、alpha0.7、原图照度门75保持固定。
正式结果来自同一批逐人裁剪输入；旧整帧训练pn2也使用这些逐人输入作基线。

| 指标 | 原 pn2 | 逐人候选 |
|---|---:|---:|
| RGB宏F1 | 0.4480 | 0.8373 |
| RGB other误报 | 65/154 | 6/154 |
| 融合宏F1 | 0.6722 | 0.8200 |
| 融合 other误报 | 3/154 | 1/154 |


本表为204个人物窗口、52段视频的本次微调留出测试。
宏F1只平均有测试真值的wave、chase、drink、pick_up、other五类。
旧pn2已见过大部分历史现场数据，不能将本表称为全新场景泛化；原反馈留出会话单列在完整报告。

本轮收益主要是other误报减少和追逐/跑动识别改善。喝水召回从11/11降到8/11，错误集中在喝水34；
挥手RGB保持11/11，但固定骨架融合仍只有7/11。融合的来源组增益区间包含0，需继续用独立数据验证。

## 查看结果

- [10段逐人结果浏览页](demo-index.html)
- [完整评估报告](analysis/analysis-report.md)、[统计附录](analysis/stats-appendix.md)
- [逐窗新旧预测CSV](analysis/paired-test-windows.csv)、[所有数值JSON](analysis/results.json)
- [图表目录](analysis/figure-catalog.md)，PNG/PDF/SVG在`analysis/figures/`
- [独立候选权重](models/videomamba_t_behavior_32f_person_roi_20261023.pth)、[模型说明](models/model-card.json)
- [人物与时间段标注](annotations-review.html)、[剩余不确定轨迹](annotation-uncertainties.json)

## 对照视频

左右分别为微调前、微调后；每个人的框下显示自己的融合动作，没有全局动作标签条。
同一视频的新旧轨迹、RGB像素哈希和骨架概率已全部对齐核对，所有输出已完整解码并核对帧数。

- [挥手3](<comparisons/挥手3_微调前后.mp4>)
- [挥手4](<comparisons/挥手4_微调前后.mp4>)
- [挥手20](<comparisons/挥手20_微调前后.mp4>)
- [喝水16](<comparisons/喝水16_微调前后.mp4>)
- [多人0060](<comparisons/多人0060_微调前后.mp4>)
- [多人0054](<comparisons/多人0054_微调前后.mp4>)
- [回传留出_181324](<comparisons/回传留出_181324_微调前后.mp4>)
- [留出_捡东西23](<comparisons/留出_捡东西23_微调前后.mp4>)
- [留出_挥手28](<comparisons/留出_挥手28_微调前后.mp4>)
- [留出_喝水26](<comparisons/留出_喝水26_微调前后.mp4>)

## 数据和训练

742个现场/线上源文件，字节去重为535段，49,160帧、约31分钟；不包括公开HMDB/Kinetics/ARID。
完整处理得到4,486个人物候选窗口；310个模糊或过渡窗口排除。
最终训练3789窗、验证183窗、测试204窗。
120段视频没有足够稳定的人物窗口，其中44段原正样本；这些失败没有被记成正确other。

用户确认的三条other轨迹已加入，增加6个训练负窗与4个验证负窗。此前刚开始的第1轮已中止归档，
正式训练从同一pn2重新开始；划分、测试集、阈值和超参数未改变。证据见`label-amendment.json`。
训练协议在`protocol.json`，复现命令和输入约定见[RUNBOOK](RUNBOOK.md)。
`selection.json`保留选定模型时的快照（当时尚未测试）；测试完成状态另存`test-completion.json`，不反向改写选择记录。

## 使用边界

这是逐人方案的独立候选，必须使用跟踪后的逐人ROI输入，不能直接覆盖现有整帧生产权重。
生产算子、NVIDIA Box包、默认模型和镜像均未更新。
RTMO/ST-GCN未训练；亮场没有合格骨架时，当前融合规则仍会强制显示其他。
交谈、打架、握手无本轮正样本，投掷/跌倒只有单源训练样本，不能声称已验证全部十类能力。
