# 人工审核多人数据续训 VideoMamba / ST-GCN

**本轮训练、联合选参和测试评分已完成；不推荐替换正式方案。** 验证集选出的RGB0／ST-GCN2＋物体阈值.25，在多人测试上other误报5/271→11/271，真实动作正确10/28→11/28，不符合优先减少误报的目标。保持原模型与原四参数（.972／35／.925／.581），详见`adoption-decision.json`。VideoMamba完成6轮但最终保留原权重，ST-GCN完成20轮并保留实验候选；不是两个新权重都已晋升。

固定ROI测试：旧少人214窗（163 other／51动作）、新多人299窗（271 other／28动作）。完整对照与误报/漏报取舍见`analysis-report.md`。未根据测试结果再次调参；“选定骨架＋原参数”的预列对照只作诊断，不能当成再次选出的正式方案。

最终状态见 `RUN_STATE.md`、`finish-exportfix.log`、`completion.json`：训练、选参、测试与演示校验全部完成。七输入两配置共2518帧、922事件、14完整输出＋8片段严格解码通过，写错/强制释放/unfinished/片段队列丢帧均0。实验完成不代表候选获得生产采用，`adoption-decision.json`明确保持原版。原失败日志保留作追溯。

输入为用户上传的 `../manypeople-action-review-20261029/10-25_动作审核.json`、原始10-25多人录像、历史逐人ROI数据。XCLIP不参与。8766审核服务已按要求关闭。

完成后的入口：

- `analysis-report.md`：全方案和两个动作分支的前后指标、数据边界。
- `selection-lock.json`：仅验证集选定的模型轮次、四参数与模型哈希。
- `candidate-config.json`、`models/`：独立候选权重及运行参数；epoch0表示保留原模型。
- `DEMO_VIDEOS.md`：实际生产共用离线入口的原方案／候选对照视频及片段。
- `stats-appendix.md`、`figure-catalog.md`、`figures/`：统计限制和真实图表。
- `window-results.csv`、`review-group-results.json`：逐窗以及人工审核条目any-trigger对照。
- `replay-parity.json`、`demo-verification.json`：生产融合/先验一致性、完整视频解码与写盘检查。

训练、预测、选参和演示使用原 `darkpipe-operator:0.3.0`，宿主编排容器并用已有Matplotlib绘图；纯NumPy指标在宿主与容器间复核20组一致。训练/预测/选参/演示脚本位于 `scripts/`，命令和GPU分配留在 `pipeline-jobs.json`。固定协议见 `protocol.json`；训练前网格/执行方式补充见 `protocol-amendment-before-training.json`；训练期、尚未查看test前的联合选轮补充见 `joint-selection-protocol.json`。失败尝试保留在 `attempts/`和原日志，不能覆盖后宣称从未失败。

计划7020窗中，实际RGB训练5607窗＝旧5009＋新人工320＋新弱标签278；其余弱标签仅留作未计分记录。骨架训练按姿态质量限制为3788窗。七个演示输入中，旧暗场来自val、微暗来自train（test无这两档正例），仅展示功能；亮场及新多人来自test。训练/验证示例不加入测试成绩。

`candidate/ckpts` 含本机链接，仅供实验运行，不是可上传的平台算子包。原平台/Box正式模型、归档及镜像不自动替换。没有重新进行实时推流、平台NFS或Jetson真机验收。
