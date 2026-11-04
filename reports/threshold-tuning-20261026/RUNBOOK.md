# 固定算法阈值调优复现

工作目录DarkScenePipeline，Python环境`.venv`。模型、融合权重0.95、门30、人体检测/姿态阈值、采样/裁剪规则固定；仅改变全局动作置信阈值和物体置信阈值。

```bash
# 首次冻结；已存在protocol时拒绝覆盖。
.venv/bin/python reports/threshold-tuning-20261026/scripts/evidence.py prepare

# 分视频独立分片；已有结果须匹配协议哈希才复用。
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 .venv/bin/python -u reports/threshold-tuning-20261026/scripts/evidence.py infer --shard 0/2
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=2 .venv/bin/python -u reports/threshold-tuning-20261026/scripts/evidence.py infer --shard 1/2

# 原始总量保护搜索，保留作诊断。
.venv/bin/python reports/threshold-tuning-20261026/scripts/tune.py

# 逐类保护的显式修订搜索，最终候选来源。
.venv/bin/python reports/threshold-tuning-20261026/scripts/class_guard.py
.venv/bin/python reports/threshold-tuning-20261026/scripts/report.py
```

初轮发现“总正动作不减”掩盖了跑动类退化；补充逐类保护。两个protocol/selection lock都保留，不能把第二轮历史test当作完全未看过的测试。新9段用于反馈阈值校准，未参与任何模型训练。

最终参数看`recommended-settings.json`与`class-guard-selection-lock.json`；最终指标看`final-summary.json`、`final-window-results.csv`。初轮`summary.json`、`window-results.csv`以及`initial-selected-predictions.json`只是诊断过程。

低物体阈值需要重算保留到0.001的输出：旧缓存只存≥0.1，不能拿截断缓存直接扫0.01。老原视频按源帧顺序解码，新旧ROI使用既定规则；新已知候选复用冻结裁剪PNG，新增候选按历史最近事件关联与同RTMO获取区域，不改采样次数。

实际GPU分片仅用于离线证据收集，不以该运行耗时评价实时性能。未改平台配置/清单/Box/镜像，逐人先验未接入生产引擎。
