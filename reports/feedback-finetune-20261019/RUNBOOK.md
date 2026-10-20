# 本轮复现实验

工作目录 `/ssd1/wfs/project/lowlight/DarkScenePipeline`。先查看 AGENTS.md，确认 GPU 可用。训练、回放使用原 `darkpipe-operator:0.3.0` 镜像，未安装或改动镜像依赖。

数据准备：

```bash
.venv/bin/python reports/feedback-finetune-20261019/scripts/prepare_data.py
```

训练（固定 seed=19，4 轮；head LR 2e-5，backbone LR 2e-6，batch=8；新负样本采样占 20%；蒸馏仅约束旧数据，避免跟随老师对新负样本的误报）：

```bash
docker run --rm --gpus device=1 --network none --shm-size=2g --cpus=8   -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=1   -v /ssd1/wfs/project/lowlight:/ssd1/wfs/project/lowlight   -w /ssd1/wfs/project/lowlight/DarkScenePipeline   darkpipe-operator:0.3.0 /opt/conda/envs/darkpipe/bin/python -u   reports/feedback-finetune-20261019/scripts/train.py
```

回放基线与各轮评估使用相同容器挂载，GPU 改为 4，依次执行：

```text
scripts/replay.py --build
scripts/replay.py --checkpoint <epoch-01.pth>
...重复至 epoch-04.pth...
scripts/select_checkpoint.py
scripts/replay.py --build --group holdout
scripts/replay.py --group holdout --checkpoint <选中检查点>
```

自动调度脚本 `evaluate_epochs.py` 等待基线和检查点就绪后评估；`test_selected.py` 仅在选定检查点后执行留出集。训练历史和配置写在 training-history.json、training-config.json。选择规则事先写入 protocol.json。

分析图在宿主虚拟环境生成（镜像未装 matplotlib，保持镜像不变）：

```bash
.venv/bin/python reports/feedback-finetune-20261019/scripts/analyze.py
```

注意：prepare/build 重跑会覆盖本轮缓存及结果。若要复现另一次运行，请使用新实验目录并更改脚本 OUT，保留当前证据。旧训练数据共 3,220 窗，历史文档的 6,440 是 train/val 合计且相同内容，不能当作独立训练窗口数。

## 交付验证与打包（本轮已完成）

候选包位于 `candidates/`，独立解包位于 `unpacked/`。正式包只比测试候选多最终说明文档；
`finalize_delivery.py` 逐文件核对所有非文档内容与已测试解包相同，再同步源码与默认包。
运行环境沿用上面的原镜像/同绝对路径挂载，测试需要以下额外设置：

- CPU pytest 依赖只读挂载 `/tmp/darkpipe-p1fix-11s6xmnm/pytest-deps:/pytest-deps:ro`，`PYTHONPATH=/pytest-deps`。
- `DARKPIPE_V2_OPERATOR_DIR=/ssd1/wfs/project/lowlight/DarkScenePipeline/reports/feedback-finetune-20261019/unpacked/dark-demo-v2`。
- 盒子合约测试另设 `NVIDIA_BOX_ROOT=/ssd1/wfs/project/lowlight/DarkScenePipeline/reports/feedback-finetune-20261019/unpacked/NVIDIA-Box`。
- CPU测试不传GPU，真实推理传 `--gpus device=1` 或 `device=4`；容器内均为`cuda:0`。

镜像内从仓库根依次执行（CPU测试使用上面的环境）：

```bash
/opt/conda/envs/darkpipe/bin/python -m pytest -q \
  tests/test_v2_remaining_fixes.py tests/test_v2_serve_lifecycle.py \
  tests/test_v2_delivery.py tests/test_v2_pose_overlay.py tests/test_v2_platform_feedback.py
/opt/conda/envs/darkpipe/bin/python -m pytest -q \
  reports/feedback-finetune-20261019/scripts/test_box_contract.py
/opt/conda/envs/darkpipe/bin/python -u reports/feedback-finetune-20261019/scripts/verify_delivery.py \
  --operator reports/feedback-finetune-20261019/unpacked/dark-demo-v2 \
  --box reports/feedback-finetune-20261019/unpacked/NVIDIA-Box
```

进入解包盒子目录运行 `python check_env.py`，进入解包算子目录运行：

```bash
python -u main.py \
  --video_path /ssd1/wfs/project/lowlight/DarkScenePipeline/real-actiondata-0909/喝水16.mp4 \
  --clip_dir /ssd1/wfs/project/lowlight/DarkScenePipeline/reports/feedback-finetune-20261019/real-offline/delivered \
  --session_json /ssd1/wfs/project/lowlight/DarkScenePipeline/reports/feedback-finetune-20261019/real-offline/session.json \
  --gpu_ids 0 --extra_hosts ''
```

回到宿主仓库根执行 `.venv/bin/python reports/feedback-finetune-20261019/scripts/verify_offline.py`，
校验完整解码、H.264、帧数、骨架事件与正常收尾状态。容器默认root创建的输出目录若不可写，
只将本轮输出目录的所有者还原为工作区用户，不修改共享训练缓存权限。

本轮选定模型文件为 `checkpoints/epoch-04.pth`，已按新名pn2复制至各包。
正式包命令：

```bash
python3 platform/package_v2.py --source reports/feedback-finetune-20261019/staging/dark-demo-v2 --output platform/dark-demo-v2_pn2_20261019.tar.gz
python3 platform/package_v2.py --source reports/feedback-finetune-20261019/staging/dark-demo-v2 --output platform/dark-demo-v2_pn2_20261019.zip
python3 reports/feedback-finetune-20261019/scripts/package_box.py --source reports/feedback-finetune-20261019/staging/NVIDIA-Box --output NVIDIA-Box_20261019.zip
python3 reports/feedback-finetune-20261019/scripts/finalize_delivery.py
```

`analysis/threshold-sensitivity.*`由`thresholds.py`在选模型之后生成，只解释固定阈值的取舍，没有用于重选模型。
最终报告由`write_report.py`从真实结果组装。再次实验须换目录保留本轮证据，不直接覆盖以上默认路径。
