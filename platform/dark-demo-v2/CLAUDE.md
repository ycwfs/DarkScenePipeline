# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 这是什么

这是一个**交付用的算子包**，不是完整仓库：暗光场景行为识别算子按
`事件_行为识别_暗光场景行为识别实时服务.zip` 打包后的样子。目录是平铺的，这是有意的——
`darkpipe/` 与 `oputil.py` 就摆在 `main.py` 同级，因为 zip 解开后就是这样运行的
（`main.py` 在 import 阶段把自己所在目录和上一级目录插进 `sys.path`）。

`README.md` 是这个算子的交付文档（中文，约 1000 行），也是参数语义、实测帧率/时延数字、
以及各默认值取值理由的**权威来源**。改任何默认值之前先读它。

## 「只是个子集」带来的后果

有若干代码路径引用了**没有**打进这个包的模块。它们能从 `darkpipe/config.py` 与
`darkpipe/stages/__init__.py` 的枚举里走到，但一 import 就会失败：

- `darkpipe/cli.py` —— **缺失**，因此 `darkpipe/shard.py`（多卡离线分片，靠
  `python -m darkpipe.cli` 起子进程）在这里跑不起来。实际影响不大：`main.py` 在
  `clip_dir` 非空时一律降级为单卡，而 `clip_dir` 本身有非空默认值，所以分片这条路
  正常情况下根本走不到。
- `enhance_cidnet.py`、`enhance_realrestorer.py`、`sr_lightsr.py`、`sr_catanet.py`、
  `recognize_xclip.py` —— **缺失**。可用的组合只有 `enhance={off,retinexformer}`、
  `sr={off,bicubic}`、`recognize=behavior`。

`ckpts/` 里正好只有这些可用路径需要的两份权重：`NTIRE.pth`（Retinexformer）与
`videomamba_t_behavior_32f.pth`（行为识别头）。注意 `PipelineConfig.recognize` 的默认值
仍是 `videomamba`，而那份权重不在这里——`main.py` 总是显式传 `recognize=behavior`。

本目录下没有测试套件、`requirements.txt`、`Dockerfile`，也没有 linter 配置，它们都在
上游仓库里。不要在这里凭空造出来。

## 怎么跑

这个包里没有任何东西能在无 CUDA 的机器上跑：`mamba_ssm` 与 `causal_conv1d`
（VideoMamba 的 selective-scan kernel）只有 CUDA 实现，而 `config.validate` 会直接拒掉
`recognize=behavior` 配 `--device cpu`。运行环境是镜像 `darkpipe-operator:0.2.0` 里的
`darkpipe` conda 环境；H.264 编码与 FLV/HLS/RTMP 出流都是调镜像里的 `ffmpeg` 二进制。

两种模式共用同一个入口，由 `--video_path` 决定走哪条：

```bash
# 离线——有限长文件（本地路径、hdfs://、或 WebHDFS http://…:9870/…）：整段处理一遍后退出
python -u main.py --video_path /data/demo.mp4 --session_json /out/session_json.json

# 实时——流地址：常驻在 --serve_port 上，直到被杀或 --run_seconds 到期
python -u main.py --video_path rtsp://10.0.0.5:554/live --run_seconds 60 \
    --gpu_ids 0,1 --proc_max_side 840 --session_json /out/session_json.json
```

`--run_seconds` 的意义就在于把常驻服务变成有限时长，从而可测试、可被编排调度。两种模式
完整的 `docker run` 命令见 README 第四节。

服务起来之后的检查手段：

```bash
curl -s http://localhost:8000/health      # fps_proc、时延、事件数、片段与各路出流存活状态
curl -N http://localhost:8000/events      # SSE 识别事件
tail -f <clip_dir>/*/events.jsonl         # 同样的事件，落在盘上，不依赖端口是否映射出来
```

## `suanzi.json` 与 `main.py` 的耦合

`suanzi.json` 是平台侧的算子清单。它的 `inputs[].name` 必须与 `main.py` 的 argparse 参数
一一对应，而 `implementation.container.args` 又把每个参数显式列了一遍——**加一个命令行参数
就要在三处同时加**（argparse、`inputs[]`、`args[]`），否则平台永远不会把它传下来。

两条清单规则直接影响了代码写法：

- 「带 `default` 字段」⇒ 平台视该参数为必填且不能为空。所以 `hdfs_output_dir` 与
  `rtmp_push_url` 是**故意不给** `default` 的——这是让它们可以留空的唯一办法。在 argparse
  里它们是 `default=""`。
- 只改 `main.py` 里的默认值，在生产上不生效：平台是按已注册的清单拼命令行的，所以
  `suanzi.json` 必须重新导入。启动日志里的 `[config]` 打印的正是实际生效值，就是为了
  诊断这一类问题。
- 真正决定分到几张卡的是 `metadata.gpu.count`（2），不是 `gpu_ids`。

## 架构

### 模式判定（`main.py:run`）

先用 `oputil.resolve_input()` 把输入取到手并判定它**是什么**，判据只看磁盘上的事实、不看
地址字符串：文件 → 离线（`run_offline_mode`）；目录 → 离线批处理（`run_offline_batch`）；
都不是但看得出是流地址 → 实时（`run_serve_mode`）；**都不是也不像流地址 → 当场
`sys.exit`**。因为 HDFS/WebHDFS 输入会先下载成本地文件，所以平台下发的 HDFS 地址会自动落到
离线模式。离线两种形态下 `rtmp_push_url` 与 `run_seconds` 都被忽略；文件永远不循环播放。

最后那条分支是补上的一个真缺陷：以前的否分支是**无条件降级为实时流**，于是一个取不到的本地
路径变成了无限重连、`session_json` 却已写成 `running` 的「健康」服务，日志里只有一句反复出现
的 `cannot open input source: '/mnt/nfs/args/1'`——而这一句对「NFS 卷没挂进容器」「平台物化的
是文本值」「目录里东西不对」「真是流地址」四种原因完全一样。`resolve_input` 的报错因此要逐段
报出路径在哪一层断掉、并列出最后一层存在的目录里有什么，沿用 `_fetch_http` 立的规矩：把试过
的每种形式和各自结果一起报出来。

平台上「选 NFS 单个文件」这条路会物化成一个不存在的路径（值槽 `/mnt/nfs/args/<参数下标>`），
落在上面那个 `sys.exit` 上——这就是它「被禁用」的方式。从字符串分不出「NFS 文件」和「容器内
本地文件」，能分的只有磁盘上的事实，所以容器内真实存在的文件仍照常走单文件离线。

批处理（`run_offline_batch`）是**依次**跑，每个文件调一次现有的 `run_offline_one`，跑完立刻
`deliver()`。两个决定各有依据：不并行，因为同卡并行提交只会互相切分时间片（同类实测见下面
「线程模型」里那个 73 ms → 250 ms）；不为省权重重载去拆 `run_offline`，因为
`recognize.py` warmup docstring 记着的首次推理 4611 ms 是 cuDNN autotune + lazy CUDA module
load + fp16 autocast 初始化，**三样都是进程级**、不随第二次 `load()` 重付，权重本身只有
34.5 MB。顺带一提离线**从不调 warmup**（全代码库只有 `server.py:525` 调），那 4611 ms 落在
第一个文件的第一个识别窗口里，整批只付一次。

失败策略在这里有一处**对 fail-fast 约定的有意例外**：单个文件打不开/解码失败只跳过并继续、
末尾非零退出；环境级错误（交付失败、CUDA OOM、盘满，见 `_is_environment_error`）才当场中断。
理由与「离线上传失败是致命的」背后那句「产出取不回来等于白花 GPU 时间」是同一个——第 3 个文件
坏了就丢掉后面 7 个已经能算出来的结果，正是要避免的事。

产出布局上，批处理给每个源文件一层子目录 `<会话>/<stem>/`，**但只有 ≥2 个文件时才加**。分层
是因为离线片段是 `<起>-<止>s-<中文动作>.mp4` 且平铺在会话目录下，两个视频都出 `2-3s-喝水.mp4`
会直接互相覆盖；只有一个文件时没有可撞的对象，加了只会得到把名字说两遍的
`demo_dark/20260901_.../demo_dark/`。省掉之后，「目录里就一个视频」与「直接给一个文件」产出
完全同一个形状，`session_json` 顶层字段也逐字段相同——这条是**兼容性约束**，改动批处理时要
保住（`/tmp` 之外没有测试套件，验证靠把 `run_offline_one` 换成假的跑编排）。

会话目录是**两层**的（见 `oputil.run_dir_name`）：`<源标识>/<时间戳>_<进程号>`，两种模式
同一个形状——`10.18.31.51_ch1/20260101_090000_1`、`demo/20260101_090000_1`。分层同时给到
按源分组（一个 `clip_dir` 下往往挂着多路相机、或跑过很多个文件）和每个源内部的时间序
（`ls` 出来就是时间序）；早先实时把标识放后缀、离线放前缀，是在这两件事里各挑一件。
时间戳带进程号，避免同一秒起的两个运行互相覆盖。取不到标识时只剩后半段。

标识由 `oputil.source_tag` 生成，取自**原始** `video_path`，不是下载下来的那份副本——副本
一律叫 `input.mp4`。它对取流地址不能只取末段：大华的 `…/cam/realmonitor?channel=1` 末段
是写死的接口名，同型号每台设备每个通道都叫 `realmonitor`，等于没有标识。所以末段命中
`_GENERIC_STREAM_NAMES`（或是纯数字，如海康 ISAPI `/Streaming/Channels/101` 里的 `101`）
时会补上主机名，通道号则从 query（大华）或路径（海康 `/ch1/`）里捞出来当后缀。
`source_tag` 保证结果里**没有 `/`**，两层结构只由 `run_dir_name` 引入——`main.py:290` 拿
它当过切片文件名的词干（`{stem}-处理后.mp4`），带斜杠会直接写坏路径。

同一个会话名会同时用在 `clip_dir`、HDFS 目录树和 `session_json` 上，三处必须对得上，所以
`session_json` 里的 `session` 字段现在也带斜杠。

`session_json` 是本算子唯一的框架级输出。实时模式写两次（启动时 `running`、收尾时
`stopped`），这样容器被直接杀掉时框架收走的也不是空文件；失败时由
`write_failed_session` 标成 `failed`，避免真实原因被「cannot save artifact」盖掉。它是
**就地补写**而不是整份覆盖：实时模式启动时那份带着会话名与目录，批处理每跑完一个文件也会刷新
一次（带 `items`），换成一个只有 error 的骨架等于把「哪些文件已经跑成了、产出在哪」一起丢掉，
而失败之后最想知道的恰恰是这些。

失败策略两种模式相反，且是有意为之：实时模式把 HDFS 上传失败当告警（本地片段还在，常驻
服务不该因为一次网络抖动整体死掉），离线模式把它当致命错误（跑完就退，产出取不回来等于
这次 GPU 时间白花）。

### 管线组装

`darkpipe/stages/__init__.py:build_stages(cfg)` 是配置枚举映射到类的唯一地点，import 都是
惰性的，返回 `(frame_stages, recognizer)`。顺序是有讲究的：

```
downscale → enhance → 【识别取帧点】 → denoise → saturation → SR → 标签条
```

识别器吃的是**增强后、超分前**的帧——行为识别头就是在这种帧上训练的，而且识别内部本来就
会缩到 224。只影响「人看到的画面」的 stage 在类上声明 `post_recognition = True`
（见 `stages/base.py` 的 `FrameStage`）；切分**只按这个标志位，绝不按 stage 名字**——按名字
前缀切分的那一版曾经静默漏掉了一个新加的 stage。`pipeline.py` 与 `server.py` 都用
`getattr(s, "post_recognition", False)`，这样早于该标志位的 stage 仍然会跑。

`proc_max_side` 的缩放刻意放在最前面，好让识别、片段和所有出流看到的是同一个尺寸，谁都
不需要知道曾经发生过缩放。

### 线程模型

三套彼此不同的并发布局。每一套都在自己的模块 docstring 里有长篇说明，连带支撑它的实测
数字——**重构之前先读那些**；有好几种看起来更简洁的写法都试过，而且实测更差。

- **`pipeline.py:run_offline`** —— 解码线程 → 处理（GPU）→ 编码线程，中间是有界的
  单生产者/单消费者队列，因此帧序不需要任何额外记账。`finally` 里会先把 `in_q` 抽干再
  join 解码线程：否则 `process_chunk` 抛异常会表现成卡死（实测：一次 CUDA OOM 之后进程
  在 0% GPU 上挂了 17 分钟）。
- **`server.py:process_loop`，单卡** —— 按争用的资源切成两个线程：GPU（增强 + 识别）与
  CPU（超分 → 标签条 → JPEG → 片段）。吞吐从两者之**和**变成两者之**大**。这笔交易是
  **牺牲时延换帧率**，只有在「≥15 fps 且 ≤1 秒」这条硬指标下才划算。
- **`server.py:process_loop`，多卡**（`gpu_ids` 填了多张）—— 一个 dealer 线程把最新帧
  严格轮询地发给每张卡各自的增强 worker；worker 0 额外独占**全部**识别工作并负责按序输出。
  正因为是严格轮询，`next_out % n_gpu` 就够用了——不需要重排缓冲，也不需要序号。每张卡上
  只有一个提交 CUDA 任务的线程，因为同卡两个提交线程会互相切分时间片（把识别单独拆成一个
  线程与增强共卡时，推理从 73 ms 涨到 250 ms）。

实时是**按帧**轮询，离线分片是**按帧段**切分。两种机制在对方的模式里都不成立——实时流没有
「未来的帧」可切段。`config.validate` 会强制这一点。

任何 worker 线程死掉都会把整条管线一起带下去（`worker()` 会 set 一个共享的 `done` 事件），
而不是留下一个「`/health` 报 ok、画面冻在最后一张 JPEG 上」的半失败状态。

### 实时出流（`streams.py`、`server.py`）

`CaptureThread` 写进一个只保留最新帧的单槽缓冲（`LatestSlot`）——新帧覆盖旧帧，这**就是**
丢帧策略，也是时延的上界。所有视频格式都从管线为 `/stream` 已经编好的那一份 JPEG 喂进去，
所以多开一种格式只多一次转封装，不是多一次画面编码。FLV/HLS/RTMP 都是 `ffmpeg` 子进程，
会自动重启（1 秒起、翻倍到 30 秒封顶）；死掉一个只影响那一种格式。

因为处理循环与推流之间隔着一个覆盖式的槽，`[process]` 那行遥测**完全说明不了**推流是否
正常——所以才单独有一行 `[push]`，而且是由**处理线程**打印的（推流线程卡死在写 ffmpeg 里
的时候它自己什么都印不出来，而那恰恰是最该看见的时刻）。

### 行为片段留存（`clips.py`）

一个 `ClipRecorder` 同时服务两种模式；有三条不那么显然的性质：

1. **前置环形缓冲**，按时间而不是帧数定界。事件描述的窗口在它产生时已经是过去了，所以
   从触发点开始录只能录到动作的尾巴。
2. **一个片段只装一个动作。** 正在录的片段只被**它自己那个标签**的后续事件延长；换成别的
   标签就当场收尾另起一个，但要 `switch_after` 次连续确认才算，避免边界抖动把一次完整动作
   剁成一堆碎片。「任何标签都延长片段」是早期那个 bug，结果一个 30 秒文件里装着喝水、
   挥手、跌倒三个动作。
3. **写盘绝不阻塞 GPU 线程。** 编码、去噪、以及 `on_saved` 上传钩子全在写盘线程上。实时
   模式下队列是有界的——满了就丢帧，丢得很响且有计数，为的是守住 ≤1 秒这条指标。离线模式
   设 `maxsize=0`（无上限）：那边没有时延指标，只要求结果完整，所以写盘慢只会拉长总耗时、
   绝不丢一帧（`clips_abandoned`/`frames_dropped` 恒为 0）。片段先写临时目录，录完整了才
   整体搬进 `clip_dir`，所以挂 NFS 的 `clip_dir` 里永远不会出现半截 mp4。

`naming="key"`（实时，`<行为>/<时间>_<行为>_<序号>.mp4`）与 `naming="range"`（离线，平铺的
`<起>-<止>s-<中文动作>.mp4`）—— 实时是一个流跑很久，按时间排更好找；离线是同一个目录里
很多个文件，按源视频里的时间命名更有意义。

### 编码走 ffmpeg 子进程，绝不用 `cv2.VideoWriter`

`media.py:VideoWriter` 把裸帧 pipe 给 `ffmpeg`。镜像里那个 OpenCV 构建**没有 H.264
编码器**——`avc1`/`H264`/`X264` 全都打不开，输出就静默变成 MPEG-4 Part 2，VLC 播得了而
Chrome/微信/钉钉/Windows 都拒绝。这就是当初「开发机上好好的、交付出去都打不开」那个切片
bug。x264 固定 4 线程，免得把要守 ≥15 fps 的识别线程饿死。交付出去的片段打不开时，
`ffprobe -show_entries stream=codec_name` 应当是 `h264`；是 `mpeg4` 就说明 ffmpeg 没了，
日志里会有对应的 `[media] WARNING`。

相关的一点：yuv420p 描述不了奇数尺寸，所以「必须是偶数」这件事在三个地方各管一段
（`downscale.py` 缩放取偶、`render.py` 补齐标签条高度、`streams.py` 用 `-vf` 裁掉）。

### 中文只用于显示，英文才是身份

`constants.py` 把 `BEHAVIOR_ZH`（标签条画什么）与
`BEHAVIOR_DISPLAY`/`BEHAVIORS`（在需要穿过文件系统或下游解析器的地方标识一个行为）刻意
分开。目录名和文件名要过 `clips.label_key()`，它会剥掉非 ASCII 字符，把每个中文名都变成
`unknown`。**所以中文要保持只用于显示。** 另外 `cv2.putText` 根本画不了 CJK——它会输出
`????` 而不报错——因此标签条走 PIL 加 `assets/NotoSansCJK-subset.ttf`，一个约 120 个字形、
15 KB 的子集，选它就是为了字体能随 zip 发布、不必重建镜像。

### HDFS 只走 WebHDFS（`oputil.py`）

镜像里没有 hadoop 客户端，也没有 JVM。所有 HDFS I/O 都是 WebHDFS 走 HTTP。`hdfs://` URI
可以接受，其中的 RPC 端口（8020）会换算到 HTTP 端口（9870）；缺 `/webhdfs/v1/` 前缀会自动
补上；**先按用户写的原样试一次**，失败再按 WebHDFS 形式试，两个试过的 URL 与各自状态码都
会出现在报错里。

WebHDFS 是**两跳**的：NameNode 回一个 307，其 `Location` 里的 DataNode **按主机名寻址**，
而容器默认解析不了。为此有两层彼此独立、任何一层单独成立都够用的兜底：
`apply_extra_hosts` 在启动时把 `extra_hosts` 追加进 `/etc/hosts`；若该文件不可写，
`_webhdfs_call` 会把跳转 URL 的 host 换成 NameNode 的地址重试一次。

## 约定

- **实测数字记在注释里。** 全代码库的 docstring 和行内注释里都记着基准数据，以及试过又被
  否掉的替代方案。改动与性能相关的决定时，把数字一并更新、并写清是什么取代了它；看到某种
  更简洁的结构很想改过去时，先查查是不是已经有注释写明它实测更差。
- 按受众双语，且是一致的：面向使用者的 `print()` 用中文并带 `[tag]` 前缀（`[mode]`、
  `[config]`、`[serve]`、`[clip]`、`[push]`、`[progress]`、`[done]`、`[warn]`、
  `[metrics]`）；解释设计意图的代码注释与 docstring 用英文。跟着所在文件的既有习惯写。
- 配置错误走 `config._die()` → `sys.exit("error: …")`，即在启动时就快速失败，不要等 GPU
  时间花掉之后再说。不致命的问题攒在 `cfg.warnings` 里，由调用方以 `[warn]` 打印。
- 诊断信息永不影响运行：`probe_video`、`_print_gpu_inventory` 这类都吞掉异常改为打印。
- `parse_bool` 存在的原因是 argparse 的 `type=bool` 会把 `"false"` 当真值；平台以字符串
  下发的布尔参数都要用它。
