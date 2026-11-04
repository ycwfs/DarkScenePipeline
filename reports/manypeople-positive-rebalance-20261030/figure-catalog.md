# 图表说明

- 01-fixed-threshold-comparison：同一旧／新测试集合、相同完整方案阈值，展示误报数和真动作正确数。无误差条，单训练种子。应同时比较橙色与蓝色柱，不能只看误报。
- 02-training-diagnostics：上一轮与本轮各分支验证集误报率／正类宏召回随轮数变化。各分支单独阈值0.75，虚线标本轮完整方案选轮；用于判断减少other是否缓解正例召回损失，不能将分支曲线当完整方案结果。

数据来源为results/fixed-threshold-comparison及两轮history.json；所有图提供PNG和PDF，未绘制虚构误差条。
