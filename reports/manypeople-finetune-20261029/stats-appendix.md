# 统计附录

{
  "seed_count": 1,
  "training_seed": 29,
  "old_test_source_groups": 48,
  "new_test_acquisition_groups": 1,
  "paired_source_bootstrap_replicates": 2000,
  "old_test_selected_minus_baseline_95ci": {
    "other_fpr": [
      0.0,
      0.0
    ],
    "positive_recall": [
      0.0,
      0.10714285714285714
    ]
  },
  "inferential_pvalues": null,
  "reason": "Single training seed, historical sources, only one new held-out acquisition session. No independent-window significance test or multiple-search p-value.",
  "weak_labels_excluded": true
}

逐窗配对但窗高度相关；old按原source group重采样2000次，报告selected−baseline的百分比差95%百分位区间。该区间只描述固定这一次训练模型在历史源组上的抽样差异，不涵盖训练随机性、参数搜索和旧数据被看过的偏差。new只有1采集组，拒绝构造虚假的组间CI。没有执行以窗口为独立样本的McNemar/t检验，无p值或多重比较显著性声明。具体支持数、混淆矩阵见results.json；各方法绝对差见effect-sizes.json。
