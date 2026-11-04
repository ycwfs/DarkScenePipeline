# 当前交付与流程图一致性核对 · 2026-10-28

核对通过：平台10月27日完整先验正式版与Box10月28日单卡优化候选均为当前对应目标的最新交付；各自默认别名正确。共有算法、模型、参数与algorithm-flow-20261027一致。Box执行优化单独说明，不把单RTX3090测试当成Orin Nano 4GB真机验收。

- [当前提交文件、参数与SHA256](../../platform/提交说明_20261028.md)
- [机器可读核验结果](audit.json) / [可重跑核验脚本](scripts/audit_delivery.py) / [集中校验文件](SHA256SUMS)
- [当前可编辑PPT/draw.io/PDF与预览](../algorithm-flow-20261027/README.md)
- [平台已完成的模型/实时验证](../full-prior-release-20261027/README.md) / [Box已完成的单卡验证](../box-single-gpu-20261028/analysis-report.md)

| 检查 | 结果 |
|---|---|
| 平台TAR、ZIP实际内容 | 各69个文件，逐文件哈希/大小等于当前源码及manifest；两种归档载荷相同 |
| 平台默认别名 | dark-demo-v2.tar.gz/.zip分别与full_priors_20261027同格式包SHA256相同 |
| Box实际内容 | 98个载荷＋manifest.json共99个文件；逐文件哈希/大小等于当前源码及内外manifest |
| Box默认别名 | NVIDIA-Box.zip与Orin4GB_candidate_20261028.zip SHA256相同 |
| 权重与训练说明 | 原13份权重两端一致；当前六模型逐个核对models.json；Box等价RTMO派生图额外单独校验 |
| Box运行代码差异 | 仅已记录的5份修改、2份新增；平台基底SHA匹配，其他映射文件相同 |
| 参数/清单/启动器 | 平台全CLI/清单合约通过；Box继承参数、默认配置、当前models.json一致；仅路径/GPU/输出放大等部署默认不同 |
| 实际图文件 | 两页原生可编辑PPT，无栅格嵌入；draw.io连线有效，参数/模型/训练数据文字核对；PDF导出文字检查、两页PNG目视排版检查 |
| CPU入口回归 | 13项通过，见[cpu-checks.log](cpu-checks.log) |
| 基础镜像 | 当前Docker镜像ID及6,796,727,259字节镜像归档SHA256均与历史记录一致，无需重新上传 |

共有默认为RGB0.972、骨架0.028、原图门35、动作阈值0.925、物体阈值0.581；挥手同腕高于双肩至少3个不同源帧，跑动速度至少0.35个人体框高度/秒。均使用逐人ROI、同人骨架和现场七类物体模型；三类先验先过滤，再统一绘制/发布事件/按人物与动作保存前后各2秒片段。照度门、缺观测行为和模型来源详见图中备注。

平台默认3卡/ORT CUDA，720处理后bicubic×2显示放大。Box默认单卡，uint8缓存、异步公平轮转、同人待处理窗口合并、batch2、720直接输出；ARM64 auto使用TensorRT（人体/物体FP16、姿态FP32且关闭TF32），x86 auto仍ORT CUDA。模型输入均为720，增强均保持PyTorch。规则与模型一致不等于逐窗概率或事件相同：已知Box喝水27原5个drink窗变3个，仍保存片段；不能称精度完全无损。

此次修正仓库首页、报告索引、提交说明和流程图中尚未反映10月28日Box候选的入口，补充部署差异并重新导出当前PPT/draw.io/PDF/预览。没有修改或重打已验证的运行归档；其SHA256与此前实测版本一致，所以本次未重复GPU推理。原目录内旧日期包和旧命名流程图保留作历史参考，只有提交入口列出的包和“完整先验方案”图是当前使用版本。包内10月27日的旧Box交付介绍是当时的版本记录，现行入口以10月28日提交说明为准。

复核命令（仓库根目录）：

```bash
# 将本机只读检查得到的镜像ID传给核验脚本。
docker image inspect darkpipe-operator:0.3.0 --format '{{.Id}}'
.venv/bin/python reports/delivery-consistency-20261028/scripts/audit_delivery.py \
  --image-id sha256:d359716620d7eb757cd7c36733e25642469450f1c801e0367fc99248f0a2f212
sha256sum -c reports/delivery-consistency-20261028/SHA256SUMS
```

本次是交付一致性检查，未连接正在运行的平台任务；平台显式填写的旧参数仍需按当前suanzi.json核对。Box没有可用真机连接，4GB共享内存容量、JetPack安装及设备实际帧率仍未验收。
