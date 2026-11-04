# 图表目录

全部由实际JSON数据用Matplotlib生成，同时提供PNG、PDF、SVG。

| 图 | 数据源 | 用途与需观察的内容 | 限制 |
|---|---|---|---|
| 01-test-comparison | evaluation/test-*.json | RGB与融合mF1变化；来源组平均准确率差值及5000次bootstrap 95%区间 | 区间不是mF1区间，不含训练种子方差 |
| 02-validation-and-coverage | training-history.json、training-data.json | 全部轮次的验证趋势及进入稳定人物窗口的视频数量 | 覆盖率是视频级“至少有一个窗口”，不是逐人检测召回率 |
| 03-class-recall | evaluation/test-*.json | 各真实类召回变化及明确窗口数，识别是否只学会other | 样本少、历史数据已被初始模型见过 |

使用时保留图题、轴名、样本数和上述边界；不将训练内演示图当作独立测试证据。
