# 手机、相机、话筒困难样本微调复现

本实验只替换独立物体检测器，人体检测、姿态、VideoMamba、ST-GCN、增强模型和动作规则均不改。生产默认仍为 RGB α=.972、原图照度门35、动作阈值.925、物体阈值.581。新增 microphone 为索引7，原0–6索引保持；本轮检测器对照不把7额外加入捡物支持集合。

## 标注与划分

`scripts/prepare_labels.py` 从既有部署裁剪缓存挖掘候选，模型框仅用于排序，不能直接充当GT。`label-views.json` 保存127个候选的来源、帧号、坐标变换；`extra-views.json` 保存旧线上相机视频的补充帧。助手从原像素图、必要的相邻帧手工视觉复核，框和有效审核区域写在 `scripts/build_dataset.py` / `reviewed-labels.json`。不确定/重复候选明确排除；未审核边缘区域通过有效区域裁剪排除。白框有效区域与绿色类别框见 `review/verified-*.jpg`。

新视频02/04/05/06/08用于训练，03用于验证，01/07/09整段不进入本次训练。旧数据保持原来源组。此角色变化覆盖此前“新九段仅反馈验证”的历史状态，但九段过去均已被查看和调阈值，不能称全新盲测。05/07/09来自同日不同监控位置，部分人物相同；不同原文件不代表独立场景。相机补充帧来自已有训练组，近邻帧没有跨split。

训练413图、308唯一视图；验证41图；测试79图。新训练同时保留原图和部署720增强图，不能把成对图像视为独立样本。摄像机样本稀少，杯没有人工框；这些限制保留在报告中。类别索引：bottle0/cup1/phone2/camera3/paper4/portable_object5/robot_or_exhibit6/microphone7。

## 训练与导出

环境为仓库 `.venv/bin/python`，Ultralytics8.4.163/Torch2.7.0+cu126，单RTX3090。从10-25现场七类m权重继续训练，扩展到八类；原有两个检测头的七个分类行逐张量保留并核验。随机初始化及训练seed27；20轮，640，batch12，AdamW lr1e-4，freeze10，无mosaic/mixup，轻微几何/HSV增强。本机缺可选polars，实验内Trainer使用等价标准库CSV列字典写checkpoint，不修改镜像依赖。

```bash
.venv/bin/python reports/object-hardneg-finetune-20261027/scripts/build_dataset.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python reports/object-hardneg-finetune-20261027/scripts/train.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python reports/object-hardneg-finetune-20261027/scripts/select_export.py
```

脚本拒绝覆盖已有冻结数据/训练目录，复现请改用新的实验目录。训练前两个失败尝试保留在preflight-v0/v1：分别是校验张量CPU/GPU设备不一致、可选polars缺失；第一个零步，第二个1轮但未保存checkpoint。正式训练从同一原始七类权重重新开始，未使用测试指标决定修复或轮次。

选模仅看固定验证集部署one-to-one头mAP50–95，20个epoch逐个评价；选择epoch8，即第9轮，.36516。ONNX静态batch1/640/FP32/opset17/nms=False，输出1×300×6。10图PT/ONNX匹配核验通过，最大坐标/数值误差6.11e-5。

## 对照的三种粒度

1. `evaluate_objects.py`：在有效审核区域裁剪图上评估旧47test和新增32test；两模型使用相同像素。该输入比部署ROI更紧，不能代替实际部署上下文结果。
2. `evaluate_context.py`：保留缓存中完整的720增强部署ROI，只对人工审核区域内中心点预测计分。使用同一类别IoU≥.5，误认容器诊断IoU≥.3。新增训练来源单列为拟合诊断，不计为泛化成绩。`context_visuals.py`输出逐图对照。
3. `paired_long.py`：在原镜像完整解码01/07/09原视频，执行正式增强、人体、跟踪、姿态、RGB/ST-GCN。两个物体检测器用正式`PersonObjectPrior.apply`处理完全相同任务与ROI；逐事件断言输入概率/帧/几何一致。各自输出完整视频和独立片段保存器。不是截取短片重启跟踪，也不是只回放旧RGB缓存；但它是离线双检测器诊断，不是实时吞吐或无限流稳定性实验。

```bash
docker run --rm --network none --gpus device=1 --ipc=host \
  -v /ssd1/wfs/project/lowlight/DarkScenePipeline:/ssd1/wfs/project/lowlight/DarkScenePipeline \
  -v /ssd1/wfs/project/lowlight/DarkScenePipeline/platform/dark-demo-v2:/app:ro \
  --entrypoint /opt/conda/envs/darkpipe/bin/python darkpipe-operator:0.3.0 \
  /ssd1/wfs/project/lowlight/DarkScenePipeline/reports/object-hardneg-finetune-20261027/scripts/paired_long.py --number 1
```

07/09用同命令换number，可分别使用其他卡。无需网络、安装依赖或构建镜像。`paired_long.py`中的microphone显示扩展仅在诊断进程内生效，生产源码和已交付归档未替换。

`cached_windows.py`另对55个旧标注动作候选复用冻结旧上下游输入，仅换检测器，检查旧动作召回回退。其旧动作筛选配置、近似绑定与本轮完整长视频不同，必须作为独立诊断表，不能混成当前全量成绩。

最后运行`review_long.py`生成全部保留/改变的交互触发组审核图；最终统计依赖明确的逐组视觉审核，不把触发减少自动当作误报减少。所有最终视频用FFmpeg完整解码验证，保留失败/剩余错误样例。
