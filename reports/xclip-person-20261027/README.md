# 逐人 XCLIP 零样本实验

按用户确认，仅增强＋逐人XCLIP，不使用姿态估计、骨架动作分类、挥手/跑动先验或物体先验，不微调。

验证集选择阈值0.41；旧测试214个已标注窗口，宏F1 0.623、other误报22/163。当前完整方案宏F1 0.784、other误报2/163。固定0.75对照下XCLIP误报降至1/163，但宏F1仍0.694且捡物0/4。不建议直接替换来优先压低误报。

[全部视频](index.html) · [详细报告](analysis-report.md) · [逐类数值](analysis-results.json) · [解码核验](verification.json) · [统计口径](stats-appendix.md)。完整01/07/09及12段旧短视频，共15个输出；完整视频主阈值0.4，短视频对照使用验证集选择0.41，标注在画面中。

`scripts/experiment.py --mode old --split val`运行旧val，`scripts/calibrate.py`锁定阈值，再`--split test`报告。`--mode long --number 1`（或7、9）使用无姿态的人体框＋外观跟踪，独立于旧姿态辅助跟踪的固定窗口分类器对照。`scripts/render_short.py --split val/test`生成按既有真值选取的代表性短视频；`scripts/report.py`汇总及完整解码。

使用原darkpipe-operator:0.3.0容器、现有本地公开权重，无下载或环境修改。32帧/1秒、人物框15%外扩、保持比例填充224×224、预训练mean/std；文本仅推理前修正“追逐”为跑动并从other删除“独自跑步”，不做事后提示词搜索。

本次单模型、历史已查看数据，无全新泛化或实时吞吐声明。正式算子/Box归档及镜像保持。
