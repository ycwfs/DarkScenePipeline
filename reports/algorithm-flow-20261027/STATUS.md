# 版本状态 · 2026-10-28 核对

用户已确认采用挥手3帧与跑动0.35框高/秒先验。本目录“完整先验方案”双页PPT、draw.io、PDF与预览已同步当前平台正式版及Box单卡优化候选的共有算法。六个模型不变，RGB仍为VideoMamba-Tiny，物体为现场七类YOLO26m。旧“物体先验与默认参数”命名文件保留作历史参考，当前演示请使用README中的“完整先验方案”链接。

默认RGB权重0.972、原图门35、动作0.925、物体0.581；新规则参数可设0关闭。规则的缺观测行为详见讲解备注。实际归档、演示及实时验收见 [本轮报告](../full-prior-release-20261027/README.md)。此前仅候选的描述属于历史状态。

| 交付目标 | 当前版本与默认别名 | 执行差异 |
|---|---|---|
| 平台正式版 | `dark-demo-v2_full_priors_20261027.tar.gz/.zip`；`dark-demo-v2.tar.gz/.zip` | 默认3卡、ORT CUDA、720模型输入、默认bicubic×2至1440长边显示 |
| Box优化候选 | `NVIDIA-Box_Orin4GB_candidate_20261028.zip`；`NVIDIA-Box.zip` | 默认单卡、uint8缓存、异步公平轮转、batch2、720直接输出；ARM64 auto使用TensorRT，x86 auto使用ORT CUDA |

Box的TensorRT人体/物体用FP16，姿态用FP32且关闭TF32，并采用等价ArgMax兼容改写；增强保持原PyTorch。模型未压缩或重新训练。共有算法一致不等于每个窗口结果相同：后端数值和调度存在差异。Box已完成单RTX3090功能/流式验证，**尚未在Orin Nano 4GB真机验收**。

当前文件与哈希见[提交说明](../../platform/提交说明_20261028.md)和[本次逐文件核对](../delivery-consistency-20261028/README.md)；Box证据见[单卡优化报告](../box-single-gpu-20261028/analysis-report.md)。原0.3.0镜像不变。
