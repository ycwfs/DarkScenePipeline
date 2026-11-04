# 图表说明

## throughput.png / throughput.pdf

- 目的：判断当前实现能否达到15 fps。数据：evidence/results.json，6样本×3复测。
- 柱为均值，散点为每轮结果，误差棒为观测min–max；红虚线15 fps。非95%CI。
- 阅读结论：原视频和拥挤输入均失败；轻负载回传结果不可外推至完整增强链或其他人数。
- 检查项：原始/已增强两类输入明确标记；单位为墙钟处理fps，不是输出视频头的fps。

## cost-and-delay.png / cost-and-delay.pdf

- 目的：定位瓶颈以及观察FIFO输入时延是否积累。
- 左：三轮逐帧平均耗时堆叠，虚线66.67ms/帧；右：一次paced回放帧年龄，虚线1s。
- 数据：evidence/throughput-frames.json、evidence/paced-frames.json及results.json。
- 阅读结论：多人动作批次耗时增加；达到期批次时出现时延跃升，串行回放不能及时消化输入。
- 检查项：RTMO是全帧一次；动作耗时是所有到期人物合计。右图在编码器输入处采样，不包含网络或播放器。
- 不能据此宣称多GPU提升倍数、因果人数曲线或长期服务SLA。
