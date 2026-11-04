# 重现本轮检测器试验

在 DarkScenePipeline 根目录运行。依赖旧实验的只读图像、标注、模型和推理输出，以及原始现场视频；数据和权重不随Git分发。

```bash
export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/reports/dark-finetune-20261024/vendor-yolo"
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$PWD/.venv/lib/python3.10/site-packages/nvidia/cublas/lib:$PWD/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"

# 初次准备：32个冻结train/val来源、每源3帧的双模型草稿。
.venv/bin/python reports/detector-clean-finetune-20261024/scripts/prepare_bright.py
# 根据已完成的逐图核对形成新标签；历史数据不改。
.venv/bin/python reports/detector-clean-finetune-20261024/scripts/build_dataset.py
# 训练目录必须为空；不要在已完成的本目录覆盖重训。
.venv/bin/python reports/detector-clean-finetune-20261024/scripts/train.py
.venv/bin/python reports/detector-clean-finetune-20261024/scripts/select_export.py
.venv/bin/python reports/detector-clean-finetune-20261024/scripts/compare.py
.venv/bin/python reports/detector-clean-finetune-20261024/scripts/publish.py
.venv/bin/python reports/detector-clean-finetune-20261024/scripts/verify.py
```

`prepare_bright.py`可生成草稿，不代表自动标注已确认。固定审核规则在`build_dataset.py`，仅适用于逐图看过的96帧：60帧左侧展品、4帧柱面人像、3帧补标远处坐着的人。展品区域坐标仅用于这些训练标签清理及测试统计，**未加入推理过滤**。

本轮训练固定12轮，单seed24，从原始PT初始化。默认训练器的自动best不用于部署选择；必须运行`select_export.py`，按显式`nms=False`的one-to-one头分域验证。保存的epoch0表示第1轮。选择完成后才运行测试，不能看测试再换epoch。生产模型没有自动替换步骤。

浏览器核验使用已安装在临时目录的Playwright和Chromium：

```bash
PYTHONPATH=/tmp/dark-review-browser PLAYWRIGHT_BROWSERS_PATH=/tmp/dark-review-browsers \
  .venv/bin/python reports/detector-clean-finetune-20261024/scripts/verify.py --browser
```

`dataset/images`是指向旧图或本轮图的只读用途链接，标签是独立新文件；不要原地编辑这些链接指向的图。亮场r0…r7是重复采样路径，不能当独立帧计数。图表取实际JSON结果，相关帧不作独立重复实验。
