# 复现与继续标注

本目录是实验候选。正式算子位于 `platform/dark-demo-v2/`，本实验没有改它的物体支路或模型。

## 已冻结资产

`source-inventory.json`保存535来源分组；6条额外空景在`additional-source-inventory.json`中隔离。`review-{actions,others,supplement,extra}.json`保存所有831条审核决定；`tiles-*/manifest.json`保存坐标变换和固定编号。`reviewed-labels.json`含原图坐标、忽略/排除原因及407可用视图；`dataset-manifest.json`含真正参与训练/验证/测试的397图和像素哈希。以manifest列出的文件为准，工作目录中可能存在早期排除样本的遗留图，不可按目录通配符直接拼训练集。

`selection.json`记录20轮独立部署头验证，`prior-lock.json`冻结候选ONNX及规则哈希。模型输出是7类，不可当person-only检测器覆盖现役人体模型。标注展示页可从最终manifest重新渲染，不重选样本或重排编号：

```bash
.venv/bin/python reports/object-prior-finetune-20261025/scripts/rebuild_review.py
```

## 在当前项目重放分析

先运行8项规则行为回归，然后用已有物体和动作输出重建统计、454映射、图表：

```bash
.venv/bin/python -m unittest discover -s reports/object-prior-finetune-20261025/scripts -p test_prior_rules.py -v
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python reports/object-prior-finetune-20261025/scripts/analyze.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python reports/object-prior-finetune-20261025/scripts/make_demos.py
```

该步骤需要原仓库、`reports/manypeople-inference-20261025/events/`、原454片段元数据和源视频；便携候选包只含结果和短演示，没有拷贝数GB原视频。演示右侧为明确标记的冻结采样帧，不是实时逐帧物体检测。

## 检测与训练环境

Python3.10，项目`.venv`，实际训练使用已安装的 Ultralytics 8.4.163；模块位置和版本见`environment.json`，与`train.log`开头一致。历史命令中的`vendor-yolo`路径并不存在，没有改变实际导入模块。GPU使用用户已授权的物理1号卡。与原0.3.0平台镜像无关。

```bash
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH=.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:.venv/lib/python3.10/site-packages/nvidia/cublas/lib:.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib
.venv/bin/python reports/object-prior-finetune-20261025/scripts/evaluate_objects.py
.venv/bin/python reports/object-prior-finetune-20261025/scripts/evaluate_actions.py --kind old
.venv/bin/python reports/object-prior-finetune-20261025/scripts/evaluate_actions.py --kind new
```

动作回放默认跳过已有输出；新实验应复制到新日期目录、保留既有锁和证据，不能覆盖后再声称冻结前已选定。新视频读取的bbox是最近事件框近似，最多偏移0.55秒；该局限随输出记录。

`scripts/train.py`、`select_export.py`是实际运行入口。训练仅运行了一次20轮，已存在`runs/objects`时脚本会拒绝覆盖；重新训练应建独立实验目录并恢复初始化权重。模型、原视频、父仓库依赖没有全部封装进候选包，不能把包内训练脚本当作自包含平台算子。

## 便携单图探针

`scripts/infer_image.py`仅依赖numpy、opencv-python和onnxruntime（GPU可用onnxruntime-gpu），可以从实验包直接运行。对已经预处理的数据集图片不传`--gamma`；对新原始人物ROI可传`--gamma`。默认CPU：

```bash
python scripts/infer_image.py path/to/person_roi.png --output probe
python scripts/infer_image.py path/to/raw_person_roi.png --gamma --device 0 --output probe_gpu
```

其输出`objects.json/objects.jpg`只是候选物体检测，不包含人体跟踪、动作模型、实时推流或片段保存。类别顺序：bottle,cup,phone,camera,paper,portable_object,robot_or_exhibit。

修改标注时记录原视图ID和明确框坐标，不因模型输出自信而自动接受。下一轮若使用新9段训练，必须更换独立验证来源并明确实验版本；不能继续将它们作为未见留出集。
