# 统计口径

单次推理，不存在多seed训练方差。单位是重叠人物时间窗，按旧原视频来源group抽样2000次给出配对宏F1差的百分位描述区间。未做p值检验或多参数显著性筛选。阈值只以val选择；历史test仅报告，已历史查看。未知GT和无跟踪窗口排除且单列。

{
  "groups": 48,
  "replicates": 2000,
  "seed": 27,
  "macro_f1_difference_xclip_minus_full": -0.1615799755799755,
  "percentile95": [
    -0.3040666023477732,
    0.04351241699284778
  ],
  "method": "Resample original source groups. Descriptive uncertainty, no significance or external-generalization claim."
}
