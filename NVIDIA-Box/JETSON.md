# Orin Nano 4GB Super 部署候选

目标：用户提供的 ARM64、512 CUDA cores、4GB LPDDR5、7/10/25W Orin Nano。
4GB 由系统和 GPU 共用；256GB NVMe 不增加 GPU 内存。当前没有真机连接或已确认的
JetPack 版本，因此不能把开发机单卡 RTX3090 的结果当作本设备验收。

本版保留增强、人体检测、姿态、逐人 RGB/ST-GCN、物体/挥手/跑动先验、可视化、
HTTP/RTMP 出流、断流恢复和按人物/动作并发保存。默认不做最终 bicubic ×2，输出长边720；
模型输入仍为720。需要旧1440输出时使用 `--sr bicubic --sr_scale 2`。
动作阈值.925、RGB权重.972、原图门35、物体阈值.581、前后2秒保持。

低内存设置：uint8人物缓存128帧、每批最多2人、实时独立动作线程及同人旧任务合并。
内存紧张可用 `--box-action-batch 1`，但多人场景的窗口吞吐和延迟可能恶化，需同机测量。
缓存、队列均有上限，合并数保留统计；没有取消任何人物或先验。CUDA Graph默认关闭。
CPU执行H.264编码；不依赖该型号缺少的独立NVENC编码器。

## 环境准备

适配目标是 JetPack6 / Python3.10 / TensorRT10+，需要在真机确认；旧JetPack不能照抄。
先保存以下输出：

```bash
cat /etc/nv_tegra_release
python3 -c 'import torch,torchvision,tensorrt; print(torch.__version__,torchvision.__version__,tensorrt.__version__); print(torch.cuda.is_available())'
tegrastats --interval 1000
```

通过 JetPack/系统软件安装 CUDA开发工具、TensorRT Python、OpenCV、FFmpeg、gcc、
Python venv和开发头文件。使用与本机JetPack一致的 NVIDIA ARM64 PyTorch/torchvision/Triton，
不要安装普通桌面 `torch==2.7.0` 或x86 CUDA wheels。本包不猜测设备的wheel下载地址。
已有可用环境时可设置 `BOX_VENV=/path/to/existing/env`；有对应离线wheel时可提供
`BOX_TORCH_WHL`、`BOX_TORCHVISION_WHL`。然后：

```bash
bash setup_jetson.sh
```

脚本会保留JetPack依赖，按sm_87从源码编译Mamba扩展，默认MAX_JOBS=1。
缺少匹配的torch/Triton时明确停止，不悄悄切换CPU或取消动作模型。
4GB设备编译和构建引擎可能需要暂时释放桌面/其他应用内存；NVMe交换空间只能辅助
构建，不视作实时性能保证。安装和真机容量尚未在这块设备上验证。

## 本机构建和测试

TensorRT engine绑定GPU型号、算力架构、TensorRT版本及ONNX哈希，必须在Orin本机生成。
本包仅携带便携ONNX，不包含开发机3090生成的engine。首次构建可能耗时数分钟。

```bash
.venv/bin/python build_engines.py
.venv/bin/python check_env.py
.venv/bin/python benchmark_box.py --video /data/test.mp4 --output-dir outputs/bench-file
.venv/bin/python benchmark_box.py --video rtsp://camera/live --duration 300 --output-dir outputs/bench-live
```

每次使用新的输出目录。`benchmark.json`保存系统版本、CPU进程内存、系统可用内存、
交换空间、退出码和正式会话结果；`tegrastats.log`保留板端GPU/内存/功耗/温度读数。
离线帧率不能当作真实推流稳定性。真机需检查持续运行内存余量、动作结果年龄、丢旧输入/
合并窗口、保存失败及片段完整性，并用外部播放器接收实际推流。未完成这些步骤前，
不能承诺4GB设备完整方案15FPS，也不能从34TOPS直接换算实际视频帧率。

`--box-backend cuda`需要另行提供ARM64 ORT CUDA；默认ARM64使用TensorRT。
`--box-profile reference --sr bicubic`可回退原缓存/调度/显示尺寸做同机对照；
模型及所有动作阈值不随性能档位改变。
