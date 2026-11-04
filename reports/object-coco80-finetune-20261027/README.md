# COCO80现场语义归并微调

入口[index.html](index.html)，详细[analysis-report.md](analysis-report.md)。保持原80类输出，未更新正式包；通用平均指标接近原模型，但手机等类别下降、现场真动作召回下降、机器人排除未解决。

候选models/yolo26m_coco80_candidate.pt/.onnx；完整视频long-videos/，同输入短片short-regression/，片段在各自baseline-clips与candidate-clips；逐类CSV、标注与复现脚本均在本目录。原/增强成对图和重复条目不视为独立样本。
