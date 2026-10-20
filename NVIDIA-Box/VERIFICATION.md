# 2026-10-19 pn2 交付核对与运行验证

默认方案：处理长边 720 → Retinexformer → VideoMamba pn2；原图照度 <75 时只用 RGB，
否则 RTMO-s + ST-GCN 按 α=0.7 融合；拒识阈值 0.75。CLI 加 `--enhance` 才执行照度增强。

新权重 SHA256：`d30b36c4c1ca9ab276442b4d0544b3d705f4ead68c7dcb5a721b696f9ee5f362`。与平台算子包内 pn2 完全相同；
增强、姿态及骨架识别权重未改。旧 pn/full/crop 权重保留回滚，逐文件哈希见 `manifest.json`。

| 检查 | 结果 |
|---|---|
| 原 0.3.0 镜像内独立 ZIP 解包自检 | 8/8 通过，真实 CUDA 前向、姿态及照度门可运行 |
| 盒子 CPU 合约回归 | 31 项通过，预处理、融合、0.75 拒识与平台一致 |
| 独立包真实 CUDA 对拍 | 10/10 标签一致，最大置信度差 0.00024205 |
| 平台包回归 | 108 项通过，含录制收尾、生命周期、参数与骨架绘制 |
| 平台完整离线入口 | 1080p 喝水样例 75 帧输入、75 帧整段输出；H.264 1440×874；4 条喝水事件、1 段片段；写盘失败/强制释放/未收尾均 0 |

对拍验证的是两个包的一致性，**不是 10 个样本全识别正确**。挥手22整段窗在阈值 0.75 下
两端均报 Other（置信度约 0.7396），对应既有 C17/18 的漏检；其他覆盖的现场正样例和 ARID 样例符合动作。

| 样例（整段均匀 32 帧） | 平台标签 | 平台置信度 | 盒子置信度 |
|---|---|---:|---:|
| 喝水16.mp4 | Drinking water | 0.933761 | 0.933873 |
| 挥手22.mp4 | Other | 0.739638 | 0.739608 |
| 捡东西2.mp4 | Picking up object | 0.852618 | 0.852665 |
| 追逐16.mp4 | Chasing | 0.909347 | 0.909256 |
| 追逐19.mp4 | Chasing | 0.924635 | 0.924805 |
| sample_drink.mp4 | Drinking water | 0.937937 | 0.937988 |
| sample_wave.mp4 | Waving | 0.906496 | 0.906738 |
| other1.mp4 | Other | 0.923057 | 0.923096 |
| 20261015_181324_drinking_water_0005.mp4 | Other | 0.737176 | 0.737131 |
| 20261019_092534_waving_0006.mp4 | Other | 0.637652 | 0.637748 |

最近两条是已处理负样本，先裁掉 66px 标签条、不再增强，原始照度不可恢复，故跳过照度门；
其他样例使用增强前原帧判门。两个包使用同一输入，避免把增强批次的 fp16 舍入误差混入模型比较。
盒子包不含骨架绘制，平台绘图在真实样例中已确认；Other 或照度门关闭骨架时不画。

微调回放（阈值 0.75）：新训练负样本 3→0/175，未参与本轮训练/选择的留出误报 2→1/68，
留出纯 RGB 误报 10→5/68。A175/177、B误报5/365、C17/18、P误报0/460 保持。
训练23段、留出9段，按完整会话拆分；ABCP含历史训练内样本，P/Q为处理后视频回放。
留出只有2个同摄像机会话，仍有误报；不能证明跨场景泛化或全部九类正样本均未退化。

环境为开发机 Linux x86_64 / RTX 3090 / Python 3.10.20 / torch 2.7.0+cu126 /
onnxruntime-gpu 1.23.2。本轮未连接实体 NVIDIA Box，也未做实时 FPS/时延验收。
原镜像无需重建；完整证据、复现脚本和图见仓库 `reports/feedback-finetune-20261019/`。

盒子上复验：

```bash
python check_env.py
python lowlight_infer.py --input sample_data/dark_frame.png --output out/ --device cuda:0
python behavior_infer.py --input sample_data/sample_drink.mp4 --enhance --window whole --device cuda:0 --json out/drink.json
python behavior_infer.py --input sample_data/sample_wave.mp4 --enhance --window whole --device cuda:0 --json out/wave.json
```

长视频默认 `--window last` 只验证最后1秒；本包不含平台视频流、切片录制、超分或可选三部曲。
历史 pn 核对证据保留在仓库 `reports/nvidia-box-verification-20261018/`，不适用于本轮 pn2 数值。
