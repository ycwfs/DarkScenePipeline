# 逐人动作识别：整体算法流程（2026-10-23）

按当前逐人候选更新的一页关键流程图，替代主流程图中旧的整帧分类及跨人物骨架选择。
图对应独立实时候选；平台算子、Box包和镜像未因此更新。

- [原生可编辑 PPT](逐人动作识别_整体算法流程_20261023_可编辑.pptx)：文字、流程框、判断菱形、箭头均可编辑，没有整页图片；可直接复制这一页到会议PPT。
- [draw.io 编辑源](逐人动作识别_整体算法流程_20261023.drawio)：同步到 [原主流程图位置](../../platform/模型运行流程_v2.drawio)。
- [PNG 预览](逐人动作识别_整体算法流程_20261023.png) / [PDF](逐人动作识别_整体算法流程_20261023.pdf)：从实际PPT渲染导出。
- [讲解备注](讲解备注.md)：模型、参数和简图省略的实现边界；亦写入PPT演讲者备注。
- [编辑与导出检查](evidence/editable-qa.json)。

图中关键链路：视频输入 → 记录原图亮度并增强 → 人体检测与跟踪 → 逐人时序窗口 →
人物ROI的VideoMamba识别 → 照度门 → 仅RGB或同人骨架融合 → 0.75阈值 → 逐人显示与保存。

RGB使用逐人ROI微调的VideoMamba-Tiny；骨架为RTMO-s与ST-GCN。照度门75，融合系数0.7/0.3。
亮场无合格骨架时保留soft其他票，不画成纯RGB回退；不再跨人取最高骨架，也不使用物体先验。
人体检测、姿态关联和缓存实际交错执行，本图表达算法依赖，箭头不代表各线程严格串行。

保存的是完整画面，每个人、每种动作独立触发，允许重叠；前后各2秒，超过15秒分段。
其他不触发保存。跟踪ID不是永久身份；启动缓存不足、断流、漏识或提前停止可能缺少上下文。
单卡与多卡使用相同算法，具体调度与输入输出故障处理省略，详见
[工程优化](../person-optimization-20261023/README.md)和[实时流核验](../person-stream-audit-20261023/README.md)。

2026-10-15的两部曲/三部曲及历史效果图保留在 [原会议目录](../meeting-20261015/README.md)。
旧主图另存于 `sources/previous-overall-20261015.drawio`，避免把当时的阈值与旧模型结论当成新方案。

## 复现与编辑

```bash
.venv/bin/python reports/algorithm-flow-20261023/scripts/build_flow.py
libreoffice -env:UserInstallation=file:///tmp/lowlight-algorithm-flow-render-20261023 \
  --headless --convert-to pdf --outdir reports/algorithm-flow-20261023 \
  reports/algorithm-flow-20261023/逐人动作识别_整体算法流程_20261023_可编辑.pptx
mv reports/algorithm-flow-20261023/逐人动作识别_整体算法流程_20261023_可编辑.pdf \
  reports/algorithm-flow-20261023/逐人动作识别_整体算法流程_20261023.pdf
pdftoppm -png -singlefile -scale-to 2400 \
  reports/algorithm-flow-20261023/逐人动作识别_整体算法流程_20261023.pdf \
  reports/algorithm-flow-20261023/逐人动作识别_整体算法流程_20261023
```

生成脚本复用旧会议目录 `.tools/` 中的python-pptx/lxml；没有新增算子依赖。
PPT使用Noto Sans CJK SC中文字体，可在PowerPoint中统一替换为微软雅黑等字体。
折线为原生连接线段，移动框后可单独调整拐点；draw.io中的每条折线是完整连接对象。
