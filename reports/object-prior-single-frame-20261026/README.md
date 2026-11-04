# 单帧物体先验对照

用户提出“附近一帧检出即可”。已有缓存回放表明，相比连续3帧规则，真实喝水由5/11恢复至8/11，捡物0/4恢复至4/4；但已知other误捡物0/3回到2/3。比无物体必要条件的原方案仍损失2个正确喝水。

- [完整分析与精确数表](analysis-report.md)
- [统计限制与区间](stats-appendix.md)
- [图表说明](figure-catalog.md)
- [逐窗结果](window-decisions.csv)
- [原样例视频](../object-prior-finetune-20261025/index.html)：旧视频标的是旧规则，本次标签见CSV。

每窗口只有原有5个采样帧，不是全视频每一帧检测；新9段的90个保留、474个抑制没有完整真值，不能当准确率。此为探索性离线对照，生产包和镜像未变。

从项目根目录复现：

```bash
MPLCONFIGDIR=/tmp/darkpipe-single-frame-mpl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python reports/object-prior-single-frame-20261026/replay.py
```

依赖上一轮`object-prior-finetune-20261025/action-evaluation/`缓存及`prior_rules.py`；源文件哈希记录在results.json，完整定义在protocol.json。
