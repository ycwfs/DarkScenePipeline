# 逐人 VideoMamba 微调复现说明

本轮只覆盖现场采集视频和两批平台负样本；不重新处理 HMDB/Kinetics/ARID。
使用 `DarkScenePipeline/.venv/bin/python`。GPU 命令需在能访问 NVIDIA 驱动的主机环境运行；
先查看显卡占用，再设置 `CUDA_VISIBLE_DEVICES`。

## 输入契约

- 原视频缩放到最长边720，再用现有 Retinexformer 增强；量取亮度必须在增强前。
- 线上回传的视频已经增强：裁掉旧全局标签条，不重复增强，原图亮度未知。
- 每帧 YOLO person 检测，短期轨迹关联；每个轨迹的框外扩15%、等比例填灰114至400×224。
- 最近约1秒内按时间取32帧；同一窗口中的RGB、骨架和亮度时间索引一致，每0.5秒分类一次。
- 人物丢失过久则清空历史，不能拼接其他人的帧。ROI可能仍包含邻近人物，未使用实例分割。
- 固定 RTMO/ST-GCN，固定 alpha0.7、阈值0.75、照度门75；低于阈值显示其他。

## 数据处理与核对

1. `scripts/inventory.py`：列出全部源文件、哈希去重和历史数据来源。
2. `scripts/freeze_splits.py`：按源视频/旧录制家族/平台会话固定划分。已存在的不同划分会拒绝覆盖。
3. `scripts/prepare_people.py --shard 0/2` 与 `--shard 1/2`：可分别放在两张空闲卡执行，完整解码并保存缓存。
4. `scripts/build_review.py --label wave` 等：生成不含模型预测的带人物编号预览；
   `scripts/detail_review.py <video_id> --track N` 用于更密集时间核对。
5. `annotations.json` 保存助手实际视觉核对的逐人/时间标签，不能从文件名批量复制正标签。
   `annotations-review.html` 可点击打开每个源视频的轨迹预览。
6. `scripts/assemble.py --verify-pixels`：验证原视频和每个RGB窗口哈希、帧时间和同轨迹对应、标签、组间隔离。
   生成 `training-data.json`；只有全部535段完成并校验通过才允许训练。

`data/<id>/manifest.json` 是不含真值的预处理快照，`person_label=null` 保持原样。
最终逐人标签来自 `annotations.json`，合并后写入 `training-data.json`；不原地改写缓存。
本次用户补充确认的三个other轨迹见 `label-amendment.json`。只修订标签时，已通过的像素校验
可在缓存未改动的条件下复用；本轮原始校验证据与中止记录均保留在归档目录中。

窗口规则固定在 `protocol.json`：主要动作至少覆盖50%的采样位置且未知部分不超过20%；
竞争动作并列则排除；other至少覆盖90%，且不得包含已标为动作的位置。不确定轨迹整体排除。
负样本默认继承已有other标签，其来源与明确逐人复核的标签分开记录。

## 训练与测试

```bash
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -u reports/person-finetune-20261023/scripts/train.py
```

正式运行还设置了ORT所需的 `LD_LIBRARY_PATH`（cudnn/cublas/cuda_runtime）和
`TRITON_CACHE_DIR`，完整命令与实际运行输出见 `training.log`、`evaluation-demo.log`。

从 pn2 微调8轮：有效batch8，head LR1e-4、backbone LR1e-5，AdamW，weight decay0.05，
label smoothing0.05，随机种子23；所有32帧共享水平翻转。各类抽样质量相等，各类内部每个
源视频抽样质量相等。无蒸馏，避免把旧整帧模型的逐人误报复制给新模型。
混合精度初始缩放512；先通过6次有限梯度和有效权重更新预检。训练中跳过的溢出步数单独记录。

每轮只评估验证集，以融合宏F1优先、RGB宏F1次之、other误报再次之选择检查点；
未微调基线也可胜出。8轮完成并写出 `selection.json` 后，`scripts/evaluate_test.py`
才访问固定测试集。不可通过测试结果重选检查点或阈值。
`analyze.py` 生成逐窗CSV、JSON、报告和真实PNG/PDF/SVG图表。

脚本拒绝覆盖已有训练历史、测试结果或演示。复跑实验应建立独立目录，沿用只读缓存、
协议和划分，避免覆盖本轮证据。`run_training.py` / `finish_run.py` 是本次运行编排记录。

## 演示

`render_demo.py` 使用 `selection.json` 指定的独立候选；`--checkpoint` 可显式选择旧pn2，
`--output` 指向另一个目录。`demo-samples.json` 在查看测试推理结果前固定了10段输入。
`compare_demos.py` 核对新旧轨迹、输入像素哈希、骨架概率一致，合成带标题的对照视频并完整解码验证。

`preprocess=fullfov224x400` 是兼容旧加载器的张量形状标记；新模型同时写有
`input_scope=tracked_person_roi`。它必须接收上述逐人裁剪，不可直接替换整帧生产模型。
本轮未修改生产包、NVIDIA Box包或镜像。
