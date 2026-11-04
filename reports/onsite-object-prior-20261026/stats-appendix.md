# 统计附录

一个固定模型组合、一次确定性评估；无训练seed重复实验。统计单位为人物动作窗口；同源窗口相关，按原source group整体配对重采样2000次，seed26。以下95%区间为百分位bootstrap描述区间（变化=加入先验后−之前），不作为显著性/泛化证明。未做p值检验，因此无多重检验校正声明。

| 划分 | 指标 | 变化（百分点） | 95%区间（百分点） | 有效来源组 |
|---|---|---:|---|---:|
| all | other_false_positive_rate | -1.19 | [-2.17, -0.37] | 158 |
| all | positive_correct_rate | -10.09 | [-17.39, -4.90] | 77 |
| all | drink | -18.50 | [-54.77, -13.01] | 19 |
| all | pick_up | -31.82 | [-63.89, -15.15] | 14 |
| train | other_false_positive_rate | -1.10 | [-2.16, -0.17] | 96 |
| train | positive_correct_rate | -9.83 | [-18.49, -4.09] | 47 |
| train | drink | -17.48 | [-65.63, -12.52] | 12 |
| train | pick_up | -27.78 | [-64.30, -10.00] | 10 |
| val | other_false_positive_rate | -1.23 | [-4.68, 0.00] | 30 |
| val | positive_correct_rate | -13.89 | [-32.26, 0.00] | 14 |
| val | drink | -20.00 | [-66.67, 0.00] | 3 |
| val | pick_up | -75.00 | [-100.00, -50.00] | 2 |
| test | other_false_positive_rate | -1.71 | [-4.40, 0.00] | 32 |
| test | positive_correct_rate | -9.80 | [-23.95, 0.00] | 16 |
| test | drink | -36.36 | [-66.67, 0.00] | 4 |
| test | pick_up | -25.00 | [-50.00, 0.00] | 2 |

零变化的区间收缩到0仅反映本批样本，没有证明总体风险为零。喝水/捡物样本及来源组较少，区间可能很宽；分母不包含未知标签。总体accuracy/macroF1和十类混淆矩阵保存在summary.json，macro只在有真值支持的类上平均。没有支持的fight/talk/handshake等不能声称验证通过。
