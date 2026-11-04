# 复现命令

在仓库根目录执行。GPU编号按实际空闲情况调整；这里使用物理GPU1，容器中对应GPU0。预览脚本使用真实模型，且逐事件断言逐人类别、选择与融合概率一致。

```bash
docker run --rm --user 1032:1032 --gpus device=1 --network none --shm-size=2g --cpus=8 \
  -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=1 -e TRITON_CACHE_DIR=/tmp/triton-multiperson \
  -v /ssd1/wfs/project/lowlight:/ssd1/wfs/project/lowlight \
  -w /ssd1/wfs/project/lowlight/DarkScenePipeline darkpipe-operator:0.3.0 \
  /opt/conda/envs/darkpipe/bin/python reports/multiperson-pose-20261022/scripts/preview.py \
  --names 挥手3 挥手4 --lum-gate 0 --out reports/multiperson-pose-20261022/gate-off-demo
```

默认门组改为 `--lum-gate 75 --out reports/multiperson-pose-20261022`，样例为挥手3/4/5/20、追逐16、喝水16、捡东西2、挥手18、other80/1。展品诊断组用追逐15/24/25/26及 `--out reports/multiperson-pose-20261022/default-multi`。不要将关照度门的展示参数误设为生产默认。

```bash
.venv/bin/python reports/multiperson-pose-20261022/scripts/build_gallery.py
python3 platform/package_v2.py --check
python3 platform/package_v2.py --output platform/dark-demo-v2_multiperson_20261022.tar.gz
python3 platform/package_v2.py --output platform/dark-demo-v2_multiperson_20261022.zip
python3 scripts/package_nvidia_box.py --sync --output NVIDIA-Box_20261022.zip
python3 reports/multiperson-pose-20261022/scripts/verify_archives.py
```

最后一条核对全部归档字节、独立解包，并同步默认别名。包内容变化后旧报告哈希不再适用，需要重新运行实际入口验证。

平台完整入口与上面相同的Docker参数，Python后改为：

```bash
reports/multiperson-pose-20261022/unpacked/platform-tar/dark-demo-v2/main.py \
  --video_path /ssd1/wfs/project/lowlight/DarkScenePipeline/real-actiondata-all/挥手3.mp4 \
  --gpu_ids 0 --lum_gate 0 --extra_hosts '' \
  --clip_dir /ssd1/wfs/project/lowlight/DarkScenePipeline/reports/multiperson-pose-20261022/platform-entry \
  --session_json /ssd1/wfs/project/lowlight/DarkScenePipeline/reports/multiperson-pose-20261022/platform-entry/session.json
```

原生Box入口：

```bash
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
TRITON_CACHE_DIR=/tmp/triton-multiperson-box .venv/bin/python \
  reports/multiperson-pose-20261022/unpacked/box/NVIDIA-Box/run.py \
  --video_path /ssd1/wfs/project/lowlight/DarkScenePipeline/real-actiondata-all/挥手3.mp4 \
  --lum_gate 0 --output-dir reports/multiperson-pose-20261022/box-entry \
  --session_json reports/multiperson-pose-20261022/box-entry/session.json
.venv/bin/python reports/multiperson-pose-20261022/scripts/verify_entries.py
```

相关回归：

```bash
.venv/bin/python -m pytest -q tests/test_v2_delivery.py tests/test_v2_platform_feedback.py \
  tests/test_v2_pose_overlay.py tests/test_v2_remaining_fixes.py tests/test_v2_serve_lifecycle.py \
  tests/test_nvidia_box_native.py reports/feedback-finetune-20261019/scripts/test_box_contract.py
```

最终包回归在原镜像内运行同一列表，并设置`DARKPIPE_V2_OPERATOR_DIR`指向解包平台目录、`NVIDIA_BOX_ROOT`指向解包Box目录。原镜像不包含pytest，本轮挂载现有本地pytest纯Python测试依赖到`.tools`，经`PYTHONPATH`加载并设置`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`；无联网安装、无镜像重建。
