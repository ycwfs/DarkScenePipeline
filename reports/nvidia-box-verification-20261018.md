# NVIDIA Box 最新方案核对与运行验证（2026-10-18）

核对前，`NVIDIA-Box/` 与旧 `NVIDIA-Box.zip` 内容一致，四份现役模型和两份回滚权重已经是当前平台版本，
但两处默认尺寸仍为 840，骨架探针预热、设备参数传递和视频输出路径还有差异。现已修复并重打包。

最新交付：`NVIDIA-Box_20261018.zip`，默认 `NVIDIA-Box.zip` 已同步为相同内容。
实测对象为独立 ZIP 解出的模型包，执行在本机空闲 RTX 3090 上；**未连接实体 NVIDIA Box**。
用户尚未提供实体设备连接信息，本次不能判断实体设备上当前安装的版本。

## 版本与代码

- 当前方案：处理长边 720 → Retinexformer → pn VideoMamba；增强前亮度门 75，暗窗仅 RGB，亮窗 α=0.7 融合 RTMO-s + ST-GCN，最终阈值 0.65。
- NTIRE、pn RGB、RTMO-s、ST-GCN 及两份回滚权重，与当前 `platform/dark-demo-v2/ckpts/` SHA256 全部一致；无需重新训练或换模型。
- 模型定义相同；VideoMamba 盒子版仅静默了两行构造日志，包根 `models/__init__.py` 是空文件，均不改变模型计算。
- 同步增强与动作识别 CLI 的默认尺寸 840→720，骨架探针改为初始化 + 两次额外预热 + 三次计时取最小，与平台实现一致。
- 修复 `behavior_infer.py --device cuda:N --enhance` 时增强模型没有跟随指定 GPU；CPU 识别请求在启动时明确拒绝。
- 修复 `lowlight_infer.py --output out/demo.mp4` 被当作目录的错误；原图窗与识别窗长度不同时明确拒绝，避免照度门用错窗口。
- 识别张量传 GPU 后再转置，与当前平台一致。保留增强 CLI、识别 CLI 的独立使用方式；识别执行完整增强链时须加 `--enhance`。
- 包定位不变：这是模型轻量化/独立推理包，不含实时拉流、推流、片段保存、超分，也按既有裁定不携带可选三部曲/YOLO 先验。

代码差异见 [code-changes.diff](nvidia-box-verification-20261018/evidence/code-changes.diff)，
权重核对见 [before-update.json](nvidia-box-verification-20261018/evidence/before-update.json)。

## 实测结果

验证环境：Linux x86_64、Python 3.10.20、torch 2.7.0+cu126、onnxruntime-gpu 1.23.2、rtmlib 0.0.16、RTX 3090。
开始检查时 6 张卡均有人使用，后续 GPU 1 空闲且无计算进程，才设置 `CUDA_VISIBLE_DEVICES=1` 开始本次测试。
无需共享 GPU 0，也未停止或修改其他用户任务。

| 项目 | 结果与口径 |
|---|---|
| ZIP 内容与模型哈希 | 原包 CRC 正常；原目录与原包 22 个文件一致；6 权重与当前平台一致 |
| CPU 合约检查 | 31/31，通过；独立解包后同一组 31 项也通过，不计成 62 个独立测试 |
| 真实 GPU 环境自检 | 8/8，通过；增强、RGB CUDA 扩展、RTMO-s/ST-GCN、暗/亮门控都已执行 |
| 增强单帧 | 像素均值 9.65→77.19；同输入 fp16 输出与平台逐像素一致 |
| 暗光 ARID 样例 | 喝水 0.9346、挥手 0.8970；均走门下纯 RGB |
| 现场 C，720，挥手 | 4/4 窗口命中 |
| 现场 C，720，捡东西 | 5/5 窗口命中 |
| 现场 C，720，追逐 | 3/3 窗口命中 |
| 现场 C，720，喝水 | 6/6 窗口命中 |
| C 合计 | 18/18；全片窗和 1 秒滑窗，每 0.5 秒左右取窗，包含重叠窗 |
| P 负样本抽样 | 3 段整片均报其他，未测全量 460 窗口 |
| 与当前平台对拍 | 23/23 标签一致；置信度最大绝对误差 0.000168681145 |
| 关门融合 CLI | 喝水 0.9056、挥手 0.8770；骨架均启用 |
| 回滚旧裁剪链 CLI | 喝水 0.9077，复现旧基线 |
| 默认最近 1 秒 CLI | 喝水16 → 喝水 0.9428；确认默认处理尺寸 720 |

P 抽取的是 `20261012_094823_drinking_water_0029`、`20261012_083334_picking_up_object_0001`、
`20261012_091922_drinking_water_0010`。这里只做案例回归，不用 3/3 外推误报率。

现场 C 和 pn 使用的 P 含训练内数据；本次结果支持“当前交付代码可运行、指定样例符合预期、与平台默认方案一致”，
不能当作独立泛化评测。增强后人物更可见，但画面仍有暗噪声；未提供配对真值，不计算 PSNR/SSIM。
本包没有服务线程与出流链，本轮未验证实体盒子的环境兼容性或 ≥15 fps / ≤1 秒实时指标。

证据：

- [环境自检日志](nvidia-box-verification-20261018/evidence/check-env.log)
- [GPU 汇总](nvidia-box-verification-20261018/evidence/gpu/summary.json)
- [完整精度逐窗结果](nvidia-box-verification-20261018/evidence/gpu/windows.jsonl)
- [GPU 增强输出](nvidia-box-verification-20261018/evidence/gpu/dark_frame_enhanced_gpu.png)
- [独立解包 CPU 回归日志](nvidia-box-verification-20261018/evidence/extracted-cpu-contract.log)
- [CLI 默认参数实测](nvidia-box-verification-20261018/evidence/gpu/field-cli-default.json)

## 交付与复验

新包：`NVIDIA-Box_20261018.zip`，128,000,313 字节（约 122.1 MiB），24 个文件。

- SHA256：`92ba6c246887ad5bb1496f160d87c96d19cf0392f39f55585c9a8a406fde381f`
- MD5：`d63ac5fc74134b6fc5ba963eed7ea6f9`
- 当前默认 `NVIDIA-Box.zip` 与日期版相同。
- 更新前 ZIP 备份：`reports/nvidia-box-verification-20261018/NVIDIA-Box_before-update.zip`。
- 包内 `manifest.json` 给出逐文件 SHA256；外部 `NVIDIA-Box_20261018.zip.manifest.json` 还包含归档哈希。
- 最终 ZIP 的全部 `.py`、权重及依赖文件与真实 GPU 测试所用解包逐文件比对一致；仅后续补充 README/验证记录/manifest。
- 最终 ZIP 已通过 CRC 和内嵌清单校验。

在实体盒子解包后，先按 README 安装与其架构匹配的依赖，再运行：

```bash
python check_env.py
python lowlight_infer.py --input sample_data/dark_frame.png --output out/ --device cuda:0
python behavior_infer.py --input sample_data/sample_drink.mp4 --enhance --window whole --device cuda:0 --json out/drink.json
python behavior_infer.py --input sample_data/sample_wave.mp4 --enhance --window whole --device cuda:0 --json out/wave.json
```

仓库复验脚本：`reports/nvidia-box-verification-20261018/scripts/test_box_contract.py` 和 `verify_models_gpu.py`。
GPU 命令及物理卡映射保存在 `evidence/gpu-commands.json`、`evidence/gpu-cli-commands.json`。
CPU-only 尝试构造 VideoMamba 时，其 Triton 导入需要可用 CUDA driver，因隐藏所有 GPU 而终止；
这符合识别需 CUDA 的前提，后续在空闲 GPU 的环境自检和真实推理中已成功加载，未用 CPU 替代识别实测。
