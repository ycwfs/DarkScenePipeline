# 统计边界

{
  "seed": 29,
  "training_seeds": 1,
  "old_test_source_groups": 48,
  "bootstrap_replicates": 2000,
  "old_test_paired_source_95ci": {
    "selected": {
      "other_fpr_difference": [
        0.0,
        0.0
      ],
      "positive_recall_difference": [
        0.0,
        0.0
      ]
    },
    "best_trained": {
      "other_fpr_difference": [
        -0.009711117456691414,
        0.04233367105707528
      ],
      "positive_recall_difference": [
        0.0,
        0.18461538461538463
      ]
    },
    "last_epoch": {
      "other_fpr_difference": [
        -0.016760950190131917,
        0.03225806451612903
      ],
      "positive_recall_difference": [
        0.0,
        0.10909090909090909
      ]
    }
  },
  "crowded_independent_positive_holdout": 0,
  "pvalues": null,
  "limitations": "Previously inspected old sources, one training seed; no untouched-test or significant-generalization claim. Crowded positives all trained, other shares source groups."
}

比较单位为人物窗口，原来源组为旧回归重采样单位。单seed不报告训练均值±标准差，重叠窗口不构造伪独立p值；没有多重比较显著性声明。新85个正例为训练拟合，566个other为同源未训练窗口，均不报告独立泛化置信区间。
