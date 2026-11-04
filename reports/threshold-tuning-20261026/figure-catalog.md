# 图目录

1. `figures/01-baseline-vs-tuned.png/pdf`：旧val/test的喝水/捡物/挥手/跑动TP和other误报，以及新反馈的喝水FP/TP、未审跑动/挥手/捡物触发量。精确窗口数，无误差条。目的：同时展示减误报与漏报代价；非物品动作触发减少不能叫纠错；新反馈已经用于调优，不能当作独立验收。
2. `figures/02-threshold-grid.png/pdf`：动作阈值×物体阈值下的新已知喝水误报、旧val正动作保留。格点来自coarse-grid.csv，非插值预测；标注当前和推荐参数。目的：展示调参的约束与折中，不能用全other角落称最优，也不说明千分位阈值跨现场稳定。

每张图都基于同一固定模型/数据窗口；辅助描述区间见stats-appendix.md。
