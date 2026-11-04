# 图目录

1. `figures/01-true-and-false.png/.pdf`：同窗有无先验的喝水/捡物TP及FP，左为所有旧现场（含训练），右为历史test。精确整数，无误差条；源自summary.json。应同时观察FP下降和TP损失，不能只看触发减少。不能用全量训练内结果推断新摄像机泛化。
2. `figures/02-tradeoff-and-coverage.png/.pdf`：按split显示已知other误报纠正、新增真动作漏报、未知标签被过滤量；同时展示已知/未知窗口覆盖。精确整数，无误差条；源自summary.json。未知样本不能被计为误报收益，零窗口视频仍需单独检查。

源组bootstrap区间在stats-appendix.md，窗口计数柱没有伪装成独立重复实验均值。
