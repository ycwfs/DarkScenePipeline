# 用户确认 13 个未标注窗口后的复核

用户确认导出的 13 个联合规则窗口预测全部正确，已将它们作为 `user_confirmed_20261101` 真值写入本隔离复核，不改变原始审核数据和生产结果。

## 联合补回数量

- **0.85–0.925 档**：原已标注真实 9 个 + 用户确认未标注 10 个 = **19 个真实窗口**；已知误报 **4 个**。
- **三档合计**：原已标注真实 10 个 + 用户确认未标注 13 个 = **23 个真实窗口**；已知误报 **7 个**。

这两个数字不能混用。13 个窗口中有 3 个位于 0.85 以下档位。

## 旧训练数据（已标注窗口）

|数据范围|方案|真实动作|漏报为other|错报其他动作|误报other|已标注窗口|
|---|---|---:|---:|---:|---:|---:|
|old_train_all|baseline|426|53|0|5|3308|
|old_train_all|continuation|436|43|0|8|3308|
|old_train_all|running2|427|52|0|5|3308|
|old_train_all|combined|437|42|0|8|3308|
|old_train_onsite|baseline|426|53|0|5|1209|
|old_train_onsite|continuation|436|43|0|8|1209|
|old_train_onsite|running2|427|52|0|5|1209|
|old_train_onsite|combined|437|42|0|8|1209|
|old_train_online_supplement|baseline|0|0|0|0|2099|
|old_train_online_supplement|continuation|0|0|0|0|2099|
|old_train_online_supplement|running2|0|0|0|0|2099|
|old_train_online_supplement|combined|0|0|0|0|2099|
|new_human_train|baseline|1|38|0|2|320|
|new_human_train|continuation|2|37|0|2|320|
|new_human_train|running2|1|38|0|2|320|
|new_human_train|combined|2|37|0|2|320|
|all_labelled_old|baseline|485|81|1|10|3721|
|all_labelled_old|continuation|498|68|1|13|3721|
|all_labelled_old|running2|487|79|1|10|3721|
|all_labelled_old|combined|499|67|1|13|3721|


窗口会重叠，按同源人物/代次、同一真值且间隔不超过0.65秒合并后，事件级结果如下：

|数据范围|方案|检出动作事件／动作事件数|漏掉动作事件|误报other事件／other事件数|
|---|---|---:|---:|---:|
|old_train_all|baseline|123/138|15|4/766|
|old_train_all|combined|128/138|10|5/766|
|old_train_onsite|baseline|123/138|15|4/154|
|old_train_onsite|combined|128/138|10|5/154|
|old_train_online_supplement|baseline|0/0|0|0/612|
|old_train_online_supplement|combined|0/0|0|0/612|
|new_human_train|baseline|1/16|15|2/119|
|new_human_train|combined|1/16|15|2/119|
|all_labelled_old|baseline|145/168|23|8/841|
|all_labelled_old|combined|150/168|18|9/841|


`old_train_onsite` 是旧现场动作训练片段；`old_train_online_supplement` 是历史 other 补充集。`new_human_train` 是10月25日多人审核集的训练分组，85个正例中39个属于该组。基线指当前正式模型原阈值与先验；组合列是本轮续接＋双窗口跑动回放。漏报指真实动作被判为 other，误报指真实 other 被判为动作；错报动作类别另列。

旧现场训练部分按动作的基线漏报为：喝水31、捡拾15、挥手4、跑动1、抛物1、摔倒1；加入两条规则后为喝水29、捡拾13，其余这几类在该训练子集不再漏报。基线5个误报窗口为捡拾2、喝水2、跑动1；组合回放8个误报窗口为捡拾4、喝水2、跑动1、抛物1。误报增加的3个均由续接候选引入，提示续接可能延长原有假动作。

10月25日多人训练分组共320个已人工标注窗口（39个真动作、281个other）：正式模型基线漏报38个正例、误报2个other窗口；联合回放漏报37个、误报仍为2个。该数据参与过后续候选微调实验，因此只作为训练数据错误盘点，不是独立泛化评估。

逐条错误窗口见 [train-errors.csv](train-errors.csv)，用户确认窗口见 [confirmed-windows.json](confirmed-windows.json)。本报告是离线审计，未更新模型、阈值、算子包或镜像。
