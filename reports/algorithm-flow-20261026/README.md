# 当前算法与默认参数 · 2026-10-26

- [原生可编辑 PPT（2 页）](逐人动作识别_物体先验与默认参数_20261026_可编辑.pptx)：文字、框和连线均可编辑，不是整页图片。
- [可编辑 draw.io（2 页）](逐人动作识别_物体先验与默认参数_20261026.drawio)
- [PDF](逐人动作识别_物体先验与默认参数_20261026_可编辑.pdf)、[整体流程预览](preview-1.png)、[模型来源预览](preview-2.png)
- [详细讲解备注](讲解备注.md)、[可编辑性与配置一致性检查](editable-qa.json)

第一页面向会议展示关键步骤：逐人 ROI／同人骨架、照度门、动作阈值、单帧同人物体支持，以及输出和并发保存。第二页说明六个主模型是否经过现场训练及训练数据来源。

默认 RGB 权重 **0.972**（骨架 **0.028**）、原图照度门 **35**、动作阈值 **0.925**、物体阈值 **0.581**；person_object_prior=true。用户确认采用误报优先校准结果，并授权接入平台；执行状态和验证见 [本轮交付报告](../person-object-release-20261026/README.md)。保留实测漏报代价，不能称跨场景精确最优。

生成器直接读取当前平台 CLI／清单和 models.json，并校验参数相同。canonical 图同时更新至 [platform/模型运行流程_v2.drawio](../../platform/模型运行流程_v2.drawio)。旧日期目录保留对应历史方案。

```bash
.venv/bin/python reports/algorithm-flow-20261026/scripts/build_flow.py
libreoffice --headless --convert-to pdf --outdir reports/algorithm-flow-20261026 reports/algorithm-flow-20261026/*可编辑.pptx
pdftoppm -png -scale-to 1600 reports/algorithm-flow-20261026/*可编辑.pdf reports/algorithm-flow-20261026/preview
```
