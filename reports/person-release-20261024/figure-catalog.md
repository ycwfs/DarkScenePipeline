# 图表说明

01-main-comparison.png/.pdf：来自results.json，分别比较验证、测试、训练人物窗的已支持类别宏F1与other误报。观察正动作取舍和误报是否同时改善；不要将训练增益外推或将柱形视为多种子均值。

02-validation-grid.png/.pdf：来自parameter-grid.json，仅显示满足误报约束的验证网格。观察可接受参数区域及选中点，说明阈值／RGB权重存在共同取舍；白格不是缺数据，星号不是测试集最优。无误差条，因为每格为一次固定样本评估。
