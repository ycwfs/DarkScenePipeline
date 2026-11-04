# 图表说明

- `01-full-scheme-comparison.png/pdf`：固定模型输入与阈值，分别展示旧来源回归、多人正例训练拟合、同源未训练other误报。来源为`results.json`。蓝色为正确动作，橙色为误报；无误差条（单seed）。应看旧能力是否退化、训练正例能否拟合、负例误报是否增加；多人面板不能作留出泛化证据。
- `02-old-validation-dynamics.png/pdf`：39正例上一轮与85正例本轮的RGB/骨架旧验证曲线，来源为两轮`history.json`。单分支阈值.75、无先验，用于查看学习过程和旧能力变化；不能等同最终完整方案指标。只比较旧验证，因为多人正例已参加训练。

数字对照见`analysis-report.md`、`window-results.csv`，统计边界见`stats-appendix.md`。
