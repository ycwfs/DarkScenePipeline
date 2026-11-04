# 四参数实验复现

在 DarkScenePipeline 根目录使用 `.venv/bin/python`，依赖固定模型和历史现场缓存。`protocol.json` 冻结源文件与模型哈希；已有协议不得重新 prepare 覆盖。

1. `scripts/common.py` 首次创建协议；已有协议时跳过。
2. `scripts/luma_audit.py` 完整解码新9段，逐帧计算720原图亮度，核对缓存窗口亮度。
3. `scripts/replay.py --shard 0/2` 和 `--shard 1/2` 在分别指定的空闲GPU执行。151旧来源每档独立生产跟踪器、骨架掩码和ST-GCN；原图只增强一次，同像素RGB缓存优先采用当前gate30版本。
4. 回放全部结束后执行 `scripts/evidence4.py --scope old --shard 0/2`、`--shard 1/2`；新9段用 `--scope new`。固定微调m＋720增强裁剪，推理截断.001，补齐两端α可达候选的并集。五个采样帧、同人绑定、类别兼容与照度门规则不改。
5. `scripts/assemble4.py` 对齐全部51,834个人物窗。主搜索只接受新9段全部原帧都高于门的档位；其他档位只保留旧源诊断。若跟踪改变导致窗不匹配，程序报错，不能静默转移标签。
6. `scripts/tune4.py` 执行粗网格、锁定、局部细网格、最终锁定，之后才输出旧train/test诊断。完整网格在NPZ，α/门最佳条件在CSV。
7. `scripts/verify4.py` 比较累积搜索与暴力逐窗计算，所有候选与正式融合/物体规则对照、模型哈希核验。
8. `.venv/bin/python -m pytest -q tests/test_person_object_prior.py`，然后 `scripts/report4.py` 生成报告、逐源/逐窗结果、图表、分组bootstrap和文件哈希。

GPU进程显式设置 `CUDA_VISIBLE_DEVICES`、`OMP_NUM_THREADS=2`；本地CUDA/cuDNN动态库路径沿用AGENTS.md。每个脚本会限制OpenCV/torch线程。不能把本实验时间当作实时吞吐。

已存在的逐源回放/证据仅在协议SHA相符时跳过。历史源、模型和生产代码不写回。新9段的二元标签只覆盖先前0.85下527喝水触发；不得扩展到新触发、捡物或所有other。
