# 80类现场微调复现

原始权重：仓库上一级`yolo26m-detect.pt`，原COCO80类，类别ID和names逐项保持。初始全部768个张量核验与原权重相同。不是从七类/八类现场头扩回80类。

用户确认的项目语义：瓶→39 bottle，杯→41 cup，手机/相机/话筒→67 cell phone，纸张→73 book；机器人/人形展品/人像图片作为背景。原portable_object逐图检查，现有小物件无可靠对应，作为背景，不新增“便携物”。这些映射不表示COCO原生含相机和话筒。

## 数据

`prepare.py`读取上一轮冻结物体审核数据533图（413train/41val/79test）；加入304张既往已清理展品的人体训练图（来源与物体val/test冲突的排除）。保留原来源分组，组/来源/像素跨split检查。现场717train中，用原80类预训练模型conf.35补1288个其他目标弱标签，仅为能力保留，标注GT与teacher预测分开保存；人工目标框覆盖重叠预测，机器人/不明小物件区域删除重叠teacher预测。不能把teacher当作独立人工真值。

本地COCO2017：`/ssd2/coco/coco17`。按seed27选择2400 train图，每类至少30个非crowd框；COCO val的1000图作选模，每类至少5框，剩余4000图锁定后测试，每类至少4框。图像不跨组。原本地副本缺000000414738.jpg，由`/ssd2/DATASETS/coco/train2017`同名完整图补齐。所有图PIL验证通过；首轮数据扫描失败没有训练步，记录在preflight-missing-coco-image。

现场训练重复3次，与COCO回放共4551条目；原图/增强图/重复条目均不能当独立样本。现场测试源01/07/09不用于本次训练，但过去已查看、调参，属于回归，不是新盲测。

## 两组训练与选择

使用仓库`.venv/bin/python`、Ultralytics8.4.163/Torch2.7.0+cu126，各20epochs/seed27/640/batch16，关闭mosaic/mixup，轻微几何/HSV增强。训练不修改镜像依赖，LocalTrainer用标准库CSV写checkpoint metadata（本机缺可选polars）。关闭依赖自动安装。

1. `train.py`：原80类完整继承，freeze10，AdamW lr1e-4，weight_decay.0005。训练val显示通用能力大幅下降，20个checkpoint均未达到原val AP-.01门槛。
2. 看上述validation失败后、在打开test之前，新增`train_retention.py`：仅更新现有class0/39/41/65/67/73最终分类行；其余74行、全部特征/定位参数和BN统计固定；无weight decay，梯度行掩码，EMA每步恢复冻结张量。AdamW lr2e-4。这里不是新增类别或修改网络输出维数。

`validate_select.py`对每个epoch显式用部署one-to-one头nms=False评估1000张COCO val和41张现场目标val。现场val只含审核目标标签，使用有GT的瓶/手机/书本类别AP，不用未标注的其他COCO类别误报计分。约束COCO val AP50–95≥原始-.01，最大化现场val AP，同分优先早epoch；两组共40候选，只用validation选择。

最终为保守组第18轮。`finish_training.py`合并选择后才运行COCO4000 test、现场79图/部署ROI及旧55窗口回归。测试没有反向调参。所有类别表、选模轨迹、失败组结果均保留。

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/prepare.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train.py
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/validate_select.py --watch
CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/train_retention.py
CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/validate_select.py --run retention --watch
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/finish_training.py
```

以上脚本路径按本目录示意，实际从仓库根运行需前缀`reports/object-coco80-finetune-20261027/`，Python为仓库`.venv/bin/python`。冻结目录禁止隐式覆盖，复现实验应复制到新目录。一次训练前检查误访问标量buffer shape[0]已修复，零训练步，原日志保留preflight-scalar-buffer-check。镜像初次脚本名select.py遮蔽标准库select，零视频推理，已改名validate_select.py，日志保留preflight-container-import。

## 部署格式和先验适配

`export.py`导出FP32静态batch1、640、opset17、nms=False ONNX，输出1×300×6；10张COCO/现场图与PT配对数值核验。`preservation-check.json`另核对冻结特征/定位和74类行完全相同。

候选是完整80类。`coco_policy.py`仅把既有先验需要的类别ID适配：喝水39/41，捡物39/41/67/73/65（瓶/杯/合并手机/书本/遥控器），其他类别正常输出但不能自动作为这些动作的支持。原照度门35、RGB alpha.972、动作.925、物体.581、一帧支持规则均不变。相机/话筒归手机后，不会成为喝水容器。

不能直接覆盖正式7类ONNX而不改消费端ID映射；本轮适配器仅用于实验。人体检测器仍是正式的独立模型；把机器人作为物体模型负样本并不会自动阻止上游人体框进入动作识别，因此另测robot_probe。

## 完整视频

`run_videos.py`在原`darkpipe-operator:0.3.0`镜像分别处理01/07/09整段，再严格复用旧85秒短片输入。`paired_long.py`每帧只执行一次正式上游，当前7类/候选80类在完全相同任务上执行物体prior，逐事件断言源帧/几何/概率一致。各自绘制完整视频、各自保存片段。旧7类事件检测ID为方便同时显示两套词表，仅在实验事件中偏移+80，有class_schema字段；这不改变模型类别或动作决策。候选ID保持COCO80。

```bash
docker run --rm --network none --gpus device=2 --ipc=host \
 -v /ssd1/wfs/project/lowlight/DarkScenePipeline:/ssd1/wfs/project/lowlight/DarkScenePipeline \
 -v /ssd1/wfs/project/lowlight/DarkScenePipeline/platform/dark-demo-v2:/app:ro \
 --entrypoint /opt/conda/envs/darkpipe/bin/python darkpipe-operator:0.3.0 \
 /ssd1/wfs/project/lowlight/DarkScenePipeline/reports/object-coco80-finetune-20261027/scripts/paired_long.py --number 1
```

短片使用`--prior-demo`，跟踪/窗口起点不同，必须与完整视频分报。`review_long.py`及`--short`生成所有喝水/捡物保留或改变组的审核图，`review-labels.json`记录逐组视觉结论。`known_positives.py`单独检查149.92秒真喝水与09/54.8秒持瓶靠嘴附近窗口。`verify.py`严格完整解码所有输出/片段，不是只看文件头。

`robot_probe.py`对03号已审核机器人3帧执行同样增强，比较当前人体模型/原80类/候选80类在人体阈值.35下的误检，GT区域只作评分，不作为推理遮罩。有限离线结果不能作为实时吞吐、平台NFS或无限流稳定性结论。
