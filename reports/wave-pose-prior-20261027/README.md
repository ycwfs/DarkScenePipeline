# 挥手骨架先验验证

[视频对照](index.html)、[详细报告](analysis-report.md)、[复现](REPRODUCE.md)。沿用正式7类物体模型，主条件为同一腕高于双肩至少3个实际源帧，关键点置信度至少.5，原照度门35约束。

隔离候选operator/dark-demo-v2已接入真实ActionEngine与参数wave_pose_min_frames（默认3，0关闭）。正式平台/Box/镜像未替换；候选权重目录仅本机链接。长短视频、动作片段、全部wave审核、旧窗口回归和1/3/5/8帧诊断均保留。不要把减少的所有触发都叫作纠正误报。
