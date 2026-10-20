# NVIDIA 盒子 · 独立模型推理包（轻量化交付基线）

本包把暗光管线的**三个单一模型**（低照度增强、动作识别、以及 v2 起默认开启的动作识别
骨架分支）独立出来，供轻量化（剪枝 / 蒸馏 / 量化 / 换更小骨干等）工作使用：`models/` 是
模型结构定义，`weights/` 是权重，`*_infer.py` 是与线上默认 alpha 链路计算语义一致的加载与推理脚本，
`sample_data/` 是示例数据（取自公开数据集 **ARID11** 暗光动作数据，`compare/arid` 同源）。

不包含任何管线外围功能（视频流、切片、超分、去噪、标签条、事件文件等），依赖只有
requirements.txt 里那几项。

**2026-10-19 当前版本 v4（pn2）**：默认模型已同步为本轮负样本续训的 pn2，
处理长边 720、α=0.7、原图照度门 75、最终阈值 **0.75**。四份现役模型与平台包一致，
旧 pn/full/crop 三份 RGB 权重保留回滚。与平台相同输入上的真实 CUDA 对拍和环境自检结果
见 `VERIFICATION.md`，逐文件 SHA256 见 `manifest.json`。

23 段新负样本用于训练，按完整会话留出 9 段测试：留出融合误报 2→1/68，纯 RGB 10→5/68；
已有回归 A175/177、B 误报5/365、C17/18、P 误报0/460 保持。ABCP 含历史训练内样本，
留出仅两个同摄像机会话，不能据此宣称跨场景零误报或全部九类动作无退化。
本包不含平台的骨架绘制和录制服务；未在实体盒子上验证架构、FPS 或时延。
完整证据见仓库 `reports/feedback-finetune-20261019.md`。

## 一、目录结构

```
NVIDIA-Box/
├── README.md
├── requirements.txt            # 已验证的依赖版本组合（Python 3.10 / CUDA 12.6；骨架分支安装说明在文件尾部）
├── check_env.py                # 环境自检：依赖、CUDA 扩展、权重、模型前向 + 骨架分支端到端
├── weights/
│   ├── NTIRE.pth                                        # 低照度增强 Retinexformer（6.2 MB）
│   ├── videomamba_t_behavior_32f_realactall_full_pn2.pth # 动作识别 VideoMamba-T，v4 默认（26.8 MB）
│   ├── videomamba_t_behavior_32f_realactall_full_pn.pth  # v3 回滚（26.8 MB）
│   ├── stgcn_behavior_32f_realactall_full.pth           # 骨架分支 ST-GCN 头（1.1 MB，0.29M 参数）
│   ├── rtmo-s_end2end.onnx                              # 骨架分支 RTMO-s 姿态（39.6 MB）
│   ├── videomamba_t_behavior_32f_realactall_full.pth    # 上一代识别权重（v2，10-05..14 现役），回滚用（26.8 MB）
│   └── videomamba_t_behavior_32f_realact10_all.pth      # 9 月交付版识别权重（裁剪链），回滚用（26.7 MB）
├── models/                     # 模型结构定义（与仓库 darkpipe/vendor 逐文件一致）
│   ├── retinexformer_arch.py
│   └── videomamba/             # 仅静默了上游构造时的两行 print，无其他改动
├── lowlight_infer.py           # 低照度增强 · 加载 + 推理
├── behavior_infer.py           # 动作识别 · 加载 + 推理（可加 --enhance 串成线上链路）
├── skeleton_aux.py             # 骨架分支（RTMO-s 姿态 + ST-GCN + 加权融合；与平台算子逐行同源）
├── video_io.py                 # 读图/读视频/均匀采样/下采样/编码 小工具
└── sample_data/
    ├── dark_frame.png          # ARID 暗帧（取自 Drink_18_13.mp4 第 3 秒，426x240）
    ├── sample_drink.mp4        # ARID Drink_18_13（喝水，6.4 s，426x240 30fps）
    └── sample_wave.mp4         # ARID Wave_16_15（挥手，6.3 s，426x240 30fps）
```

## 二、三个模型

| 项     | 低照度增强                                           | 动作识别（RGB 分支）                                                        | 骨架分支（v2，默认开启）                                          |
| ------ | ---------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| 结构   | RetinexFormer（`n_feat=40, num_blocks=[1,2,2]`）   | VideoMamba-Tiny（`embed 192, depth 24`, 16x16 patch）                     | RTMO-s 姿态（onnx）+ ST-GCN（10 类头）                            |
| 权重   | `weights/NTIRE.pth`（键 `params`）               | `weights/videomamba_t_behavior_32f_realactall_full_pn2.pth`（键 `model`，自带 preprocess 元数据） | `rtmo-s_end2end.onnx` + `stgcn_behavior_32f_realactall_full.pth` |
| 参数量 | **1.61 M**                                     | **6.99 M**（整帧版；旧裁剪版 6.96 M）                                   | ST-GCN **0.29 M**；姿态模型见 onnx 39.6 MB                   |
| 输入   | BGR uint8 任意分辨率（内部 reflect pad 到 4 的倍数） | 32 帧；输入管线由权重元数据自动选择：整帧 400x224（默认）或裁剪链 224x224   | 与 RGB 分支**完全相同**的 32 帧（增强后、`--proc-max-side` 分辨率） |
| 输出   | 同尺寸增强后的 BGR uint8                             | 10 类概率：9 类行为 +`other` 兜底类                                       | 10 类概率 + `presence`（有人门控信号，0~1）                       |
| 精度   | 线上 fp16（`--precision fp32` 可切）               | autocast fp16                                                               | 姿态 onnxruntime CUDA EP；ST-GCN fp32                            |
| 设备   | cuda（cpu 也能跑，自动回退 fp32）                    | **必须 cuda**（mamba_ssm 无 CPU 实现）                                | cuda；缺依赖/权重或姿态时延超标时**自动停用**，退纯 RGB            |

**权重来源**

- `videomamba_t_behavior_32f_realactall_full_pn2.pth`（v4 默认，2026-10-19）：从 pn 继续训练，
  旧数据 3,220 个唯一窗口 + 新 Other 23 段/175 窗，4 轮，head/backbone LR 2e-5/2e-6，
  旧数据蒸馏系数 0.2，新负样本采样占 20%。先按 ABCP/Qtrain 选择 epoch-04，再测试
  两个封存会话的 9 段/68 窗。SHA256 为
  `d30b36c4c1ca9ab276442b4d0544b3d705f4ead68c7dcb5a721b696f9ee5f362`。
  处理后 P/Q 视频裁标签且不再增强，原图照度未知；不能和旧报告不同预处理数字直接混比。

- `NTIRE.pth`：平台在线版原版权重（NTIRE 低照度增强挑战赛训练），未用现场数据微调。
- `videomamba_t_behavior_32f_realactall_full_pn.pth`（v3，现保留回滚，2026-10-14 换装）：在
  realactall_full 基础上，用平台回传的 96 段误报切片（2026-10-12）回放成负样本窗加训的
  误报压制版，md5 `513cafeb601f8e0eb80ed22433838eca`。历史四套件回放（1020 窗）：平台域
  误报 α=0.7 时 10→1/460、α=0.8 时 57→10/460，验证集 top1/mF1 逐位持平、正样本 18/18
  不掉。与平台算子包（dark-demo-v2）内识别权重同一文件。A/B/C 和 pn 的 P 均含训练内样本，
  且历史工具有舍入、无人策略差异；这些数值用于说明模型沿革，不能作为独立泛化成绩或 720 新包验收。
- `videomamba_t_behavior_32f_realactall_full.pth`（v2 默认，现回滚保留）：与平台算子 v2 包
  （dark-demo-v2）内识别权重同一文件，md5 `04707beb8a8ba7a26de77f0855b5cb90`。训练数据 =
  ARID 暗光底座 + 现场实采 real-actiondata-all（2026-09-29/30 合并的 390 段，与旧 142 段
  零重复），并把输入从裁剪链改为**整帧 400x224**（fullfov 谱系：裁剪链只把画面中央
  约 49%x88% 喂给模型，四周是盲区）重训。α=0.7 骨架融合下，现场三套件实测与基线逐项
  持平（误报 1/365、正样本 18/18、旧 val top1 0.9831）。`--weights` 切到它即回滚到
  10-05..14 的现役行为。
- `stgcn_behavior_32f_realactall_full.pth`：骨架分支 ST-GCN 头，与上面 RGB 权重同批数据
  联合训练（10 类），md5 `6b010fe65805e413ed7da265042e120e`。
- `rtmo-s_end2end.onnx`：RTMO-s 全身姿态模型（COCO-17，rtmlib/onnxruntime 加载）。
  其关键点得分兼任"画面里有没有人"的 presence 门控信号。
- `videomamba_t_behavior_32f_realact10_all.pth`（回滚保留）：2026-09 交付版识别权重，
  裁剪链谱系，md5 `eed5c06a9dbc674194e79ee4984065f1`。`--weights` 切到它即完全复现 9 月版
  行为（输入管线随权重元数据**自动**切回裁剪链，无需其他改动）。
- 行为类别表（输出概率向量的维度顺序，**轻量化替换模型时必须保持**）：
  `wave 挥手 / throw 投掷物品 / chase 追逐 / fall 跌倒 / fight 打斗 / talk 交谈 / drink 喝水 / pick_up 捡拾物品 / shake_hands 握手 / other 其他`。

## 三、快速开始

以下数值注释是 10-18 的 pn 历史基线，pn2 实测见 `VERIFICATION.md`，不要求逐位复现旧分数。

```bash
pip install -r requirements.txt
pip install --no-deps rtmlib==0.0.16 tqdm         # 骨架分支依赖，必须单独装（原因见 requirements.txt 尾部）
python check_env.py                                # 环境自检（含骨架分支端到端），全部 [OK] 再往下

python lowlight_infer.py --input sample_data/dark_frame.png --output out/
#   -> out/dark_frame_enhanced.png   平均亮度 9.7 -> 77.2（10-18 复验，fp16, RTX 3090）

python behavior_infer.py --input sample_data/sample_drink.mp4 --enhance --window whole
#   判定(最终): 喝水 (drink)  置信度 0.9346     <- 10-18 复验（fp16 下不同批次有 ±0.01 级抖动）
#   照度门: 提亮前原图亮度 9.5 < 门限 75，仅用RGB结果（两部曲第一步，骨架分支未运行）
#   纯RGB对比: 喝水 (drink)  置信度 0.9346
#   （ARID 样例本身极暗，照度门默认会挡下骨架分支——这正是线上两部曲的暗场语义）
python behavior_infer.py --input sample_data/sample_drink.mp4 --enhance --window whole --lum-gate 0
#   判定(最终): 喝水 (drink)  置信度 0.9056     <- 关门后的加权融合基线
#   骨架分支: α=0.7  presence=0.781 (有人)
python behavior_infer.py --input sample_data/sample_wave.mp4 --enhance --window whole
#   判定(最终): 挥手 (wave)  置信度 0.8970      <- 10-18 复验（门下纯 RGB）
python behavior_infer.py --input sample_data/sample_wave.mp4 --enhance --window whole --lum-gate 0
#   判定(最终): 挥手 (wave)  置信度 0.8770      <- 关门后的加权融合基线
#   骨架分支: α=0.7  presence=0.830 (有人)

# 回滚到 9 月版行为（纯 RGB、裁剪链），应完全复现旧基线 0.9077：
python behavior_infer.py --input sample_data/sample_drink.mp4 --enhance --window whole \
    --skel-alpha 1.0 --weights weights/videomamba_t_behavior_32f_realact10_all.pth
#   判定(最终): 喝水 (drink)  置信度 0.9077
```

常用参数：`--device cuda|cpu`、`--precision fp16|fp32`（增强）、`--json 输出.json`（识别，
写 RGB/融合两份 10 类概率 + presence + 照度门（lum/gate_blocked）+ 输入管线说明）、
`--proc-max-side 0`（关闭下采样，用全分辨率进链路）、`--skel-alpha 1.0`（关骨架分支做
纯 RGB 对比）、`--skel-no-person soft|other|rgb`（无人窗策略，默认 soft）、`--lum-gate 75`
（照度门门限，0=关门退回纯加权融合）。

## 四、线上链路的预处理与判定（轻量化评估必须保持一致）

**识别模型吃到的从来不是原始帧**，线上链路（服务算子 v2 默认）是：

```
原始帧 -> 长边下采样到 720（INTER_AREA，从不上采样）
       -> Retinexformer 增强（fp16）
       -> 最近 1.0 秒的帧均匀采样成 32 帧
       -> 逐帧预处理（由权重的 preprocess 元数据自动选择其一，--weights 换权重即自动切换）：
            整帧管线（当前默认 realactall_full_pn2）：整帧缩放到 400x224（INTER_AREA，不裁剪）
            裁剪链（9 月旧权重 realact10_all）：短边缩放 256 -> 中心裁剪 224
       -> RGB/255，ImageNet mean/std 归一化
       -> VideoMamba 前向 -> softmax = p_rgb
       -> [照度门（两部曲第一步，--lum-gate 默认 75）] 提亮前原图（下采样后、增强前）
            的 32 帧窗平均 BT.601 亮度 < 75：只用 p_rgb 走判定规则，骨架分支整体不运行
            （暗场姿态不可靠，全量现场视频实测提亮前亮度 ≥75 后姿态可用率才到 1.0 平台期；
            提亮后画面亮度被增强模型拉平，量它等于瞎，所以门必须在增强之前量）
       -> [骨架分支，默认开启，未被门挡下时] 同样 32 帧 -> RTMO-s（每帧取前 2 人 COCO-17）-> ST-GCN
            = p_skel 与 presence（窗内最自信人的平均关键点得分）
            presence >= 0.5（有人）：p = 0.7*p_rgb + 0.3*p_skel
            presence <  0.5（无人）：soft 策略——空骨架仍以 0.3 权重投 other 票
       -> 判定规则：融合后最高分的具名行为 < 0.75（平台默认 reco_min_conf）时报为 other
```

两个脚本默认复现的就是这条链路：

- **整帧管线的动机**：裁剪链只把画面中央约 49%x88% 喂给模型，四周是盲区；v2 默认权重的
  整帧谱系就是为取消这个盲区重训的。预处理分支由权重元数据决定而不是开关，是因为权重和
  它的输入管线是同一个事实，分开可配就可能配错（裁剪链喂整帧模型会静默丢掉四周画面）。
- **骨架分支的动机**：两个分支的失败面互补。RGB 在无人/无关场景可能给出高分，姿态模型
  在空景也可能产生高置信度“幽灵人”，因此 presence 不能单独作为可靠的人体存在判据。
  骨架分支对非体态信息也有限制（暗光下分喝水/挥手不如 RGB）。α=0.7 加权融合是三套现场
  实测（行为窗 / 误报片段 / 现场正样本）选出的工作点。soft 策略下，无人时非 Other 概率最多为 α=0.7，低于当前 0.75 阈值，
  因此无人融合窗不会触发具名动作；原图照度门关闭骨架时，仍按纯 RGB 和 0.75 判断。
- `behavior_infer.py --enhance`＝完整链路。`--window last|whole` 切换采样语义：
  `whole`＝评测算子语义（整段均匀抽 32 帧），适用于 ARID 这类**已剪好的短动作片段**，
  示例数据即按此评测；`last`（默认）＝线上流式语义（最近 `--window-sec` 秒，默认 1.0），
  适用于监控流/长视频。两种语义对同一段视频给出的分数可能差别明显，**对比基线与轻量化
  模型时务必用同一种**。
- `lowlight_infer.py --proc-max-side 720`＝当前 v2 默认处理尺寸；`0` 表示全分辨率增强。
  对比基线与轻量化模型时保持同一取值。历史 840 配置须显式传 `--proc-max-side 840`。
- 骨架分支 **fail-soft**：缺 rtmlib/onnxruntime-gpu/权重文件、或姿态实测 >25 ms/帧
  （＝onnxruntime 装成了 CPU 版）时，自动停用骨架退纯 RGB，打印一行中文原因，识别本身
  照常运行——它只会退化到 v1 行为，不会把链路跑挂。

## 五、已做的一致性验证

v4（pn2，2026-10-19）：本轮独立交付包验证见 `VERIFICATION.md`。以下为历史版本证据。

v1（realact10_all 裁剪链、纯 RGB，2026-09-15 对拍）：用仓库部署实现（`darkpipe` 的
DownscaleStage / RetinexformerStage / BehaviorRecognizer）与本包脚本在同一输入上对拍——
增强输出单帧**逐位一致**（整批处理约 2.4% 像素 ±1 fp16 舍入差）；识别概率最大差 1.4e-4，
top1 标签一致。2026-10-05 复测：`--skel-alpha 1.0 --weights ..._realact10_all.pth` 仍逐位
复现旧基线（喝水 0.9077）。

v2（realactall_full 整帧 + 骨架分支，2026-10-05）：`skeleton_aux.py` 与平台算子
`darkpipe/stages/skeleton_aux.py` diff 核对仅 4 处预期差异（出处说明 / BEHAVIORS 本地定义 /
日志前缀 / 依赖指引），网络、邻接矩阵、归一化、多人合并、门控阈值、时延探针逐字节一致；
预处理整帧分支公式逐行取自平台 `recognize.preprocess_frame`。示例基线见第三节，融合数学
自洽（喝水窗：0.7x0.9302 + 0.3x0.838 ≈ 0.9025）。

v3（realactall_full_pn 换装 + 照度门，2026-10-14）：默认权重换成平台误报加训版 pn（四套件
1020 窗：平台域误报 α=0.7 时 10→1/460、α=0.8 时 57→10/460，A 逐位持平、正样本 18/18 不掉）；
`behavior_infer.py` 移植了平台算子的照度门——`frame_luma`/`gate_luma` 与平台
`recognize.py` 逐字同源，默认 75，量提亮前原图，暗窗只跑 RGB（骨架连姿态都不跑）、亮窗
加权融合，与平台 `main.py --lum_gate` 同语义同默认；`skeleton_aux.py` 的姿态时延探针与
平台采用三次计时取最小（防进程刚加载完的时钟/内存池爬坡把健康卡误判成 CPU）；
10-18 核对补齐计时前的两次额外预热，使初次初始化 + 两次预热 + 三次计时与平台一致。
平台侧 `skeleton_aux.py` 另有 max_prior（三部曲）代码路径，盒子**有意不携带**（三部曲
四套件落败，出厂默认 alpha+照度门 75 即会议两部曲）——alpha 路径与探针协议保持逐字节
同源。第三节示例基线即 v3 栈实测（回滚链 0.9077 逐位复现）。

轻量化后的模型可用同样方式对拍：把 `weights/` 换成新权重（或改 `--weights`），先跑
`sample_data/` 三个文件确认链路通，再在业务数据上对比概率分布与 top1 一致率。替换骨架
分支的模型时保持 `skeleton_aux.py` 的门控（TAU=0.5）与融合（α）语义，`check_env.py` 会
端到端验证。

## 六、环境要求

- **Python 3.10**：mamba-ssm / causal-conv1d 官方只发 cp310 预编译 wheel。
- **NVIDIA GPU + CUDA 12.x**：torch 2.7.0+cu126 实测可用。显存：三个模型本身 <2 GiB；
  增强在 1080p 下自动把 chunk 调到约 2 帧/批，峰值约 3 GiB。骨架分支另需约 0.5 GiB。
- **骨架分支依赖**：`onnxruntime-gpu==1.23.2`（CUDA EP；CPU 版慢 8-15 倍，会被 25 ms/帧
  时延探针拒掉）+ `rtmlib==0.0.16`（**必须 `--no-deps` 单独装**——它声明的依赖是 CPU 版
  onnxruntime，与 GPU 版共用同一目录，混装会把 CUDA EP 静默顶掉，见 requirements.txt 尾部）。
  缺任一项时骨架自动停用（启动日志说明原因），识别照常。
- mamba-ssm / causal-conv1d 的 wheel 与 torch×CUDA×python×cxx11abi 组合强绑定，换版本时
  四者必须一起核对；装不上时需在盒子现场编译（要 nvcc）。requirements.txt 是已验证组合。
- 无 GPU 时增强模型可 `--device cpu`（fp32，慢）；识别模型不行，直接报错退出。
