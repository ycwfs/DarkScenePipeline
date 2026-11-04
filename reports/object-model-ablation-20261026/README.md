# YOLO26n/s 与高清原帧物体先验对照（2026-10-26）

本轮按照用户要求给逐人先验候选加入原图照度门，并真实运行公开 YOLO26n/s/m detect 和现场七类微调 m，比较同一区域的四种输入。正式逐人服务尚未启用物体先验，本轮没有替换生产模型、归档、Box 或镜像。

- **最新补充：[误报减少与新增漏报](false-positive-tradeoff.md)**。用户已确认全9段喝水触发只有图中原片01/T317/149.92秒是真的；当前527候选为1真、526误报。高清输入n/s/现场微调m分别过滤517/522/436个误报，也均过滤该真例。附16组合、逐窗标签、连续候选分组和图。
- **捡物：[捡拾物品过滤与漏报对照](pickup-tradeoff.md)**。新37个触发按用户“基本全是误报”的意见报告过滤量，并明确保留逐窗真值未知；旧val/test已知标签单列真实动作漏报与误报减少。总览页有[旧test权衡图](figures/04-pickup-tradeoff.png)和[高清裁剪仍放行的4个窗口](examples/pickup-native-retained.jpg)。
- [结论与边界](analysis-report.md)
- [检测对照图及逐例窗口决策](index.html)
- [完整数字表](numeric-summary.md)
- [统计方法与区间](stats-appendix.md)
- [图表说明](figure-catalog.md)
- [冻结协议](protocol.json)、[输入核验](input-verification.json)

参数固定：物体置信阈值0.35，动作阈值0.85，原图亮度门30，RGB融合权重沿用0.95。没有重训、没有在新9段录像上选模型或调整阈值。

候选规则位于 `../../platform/dark-demo-v2/darkpipe/person/object_prior.py`：窗口原图亮度低于30或未知则保留原动作，并在检测调用前跳过；采样帧也检查原图亮度。明亮且可靠绑定的同人ROI中，一帧出现相容物体即可支持已有动作；所有观测不可用则弃权，不能按“未检出物体”否决。它不执行手机/展品的单帧否决。

四种输入为：增强前原像素裁剪、相同裁剪加gamma、先缩长边720再裁剪、缩720并经当前Retinexformer增强再裁剪。历史已处理回传不能还原增强前像素，在manifest标为processed；纯原始亮帧结果另见 `object-counts-raw-source.csv`。78个旧holdout ROI的原像素裁剪已逐像素核对。

物体框真值为历史辅助视觉审核框，非独立专家标注。新9段喝水触发现已得到用户明确确认，可统计上述527窗内的误报过滤。用户认为捡物触发“基本全是误报”，但未逐窗确认，报告区分过滤候选量与经逐窗真值确认的误报数；旧val/test已知标签另行统计。未触发窗口仍无完整真值，不能扩展为全视频动作准确率。公开COCO模型没有相机、纸张、小杂物、展品类别，其捡物支持与七类模型不能直接解释为同任务准确率比较。

## 复现

在仓库根使用 `.venv/bin/python reports/object-model-ablation-20261026/scripts/compare.py prepare` 生成输入；`infer --model n`、`s`、`m`、`m_finetuned` 分别运行四模型；`analyze` 汇总；最后运行 `scripts/report.py` 生成图表。prepare/推理需要CUDA，模型和视频按protocol与inputs中的路径及SHA256恢复。本地输入图和完整预测不纳入Git。

用户确认后的误报统计由 `scripts/user_feedback.py` 复用已冻结预测生成，无需重跑GPU。`user-feedback-labels.json`保留原话、范围确认、图片哈希和逐窗身份；不覆写原预测中的truth字段，不把非喝水等同于所有动作均other。

捡物统计由 `scripts/pickup_summary.py` 复用同一预测，输出37×16逐窗表、旧标签验证集/测试集混淆计数和候选过滤图；新窗口仍保留truth=null。

证据单位是78个审核ROI和619个动作窗口（旧55、新564），每动作窗口最多沿用原来的5个采样帧、同一缓存人物框和姿态局部ROI。并非逐帧新推理动作，也不是本轮新增实时速度验收。缓存人物关联的近似以及窗口重叠会限制结论。
