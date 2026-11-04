# 低置信度候选复核验收（2026-11-02）

本次验证使用 `reports/platform-recheck-20261029/input-live-900s.mp4` 的前300秒，单卡 RTX 3090，
输入缩放到720长边，关闭增强和超分以隔离逐人识别、复核及片段保存链路。平台配置为：
`candidate_recheck_enabled=true`、`candidate_recheck_lower=0.85`、`reco_min_conf=0.925`、
`lum_gate=35`、`object_min_conf=0.581`，片段前后各2秒。

结果：7500帧/300秒输入，处理16.156 FPS；输出视频7500帧、300秒、720×404；事件2287条（other 2163、挥手55、喝水53、跑动8、捡拾8）；保存32段动作片段。32段全部可被 ffprobe 解码，写入丢帧、强制释放、未完成、失败和上传丢弃均为0。31段满足前后上下文完整；1段在输入动作接近文件尾部时后置上下文只有0.48秒，侧车 JSON 正确标记 `context_complete=false`，属于有限文件边界，不是编码故障。

事件 JSON 中有11条通过低置信度复核恢复，全部来自“已确认动作续接”；本段没有满足跑动双窗口的低分候选，因此 `running_dual=0` 不代表机制未加载。复核字段写入每条事件，保留 `candidate_label`、`candidate_confidence` 和最终 `recheck` 决策。

此前同源平台实时稳定性归档仍见 `reports/platform-recheck-20261029`；本轮新增的是正式接入后的离线长流回放和片段收尾检查。该回放在 RTX3090 上完成，不能替代 Orin Nano 4GB 真机实测。
