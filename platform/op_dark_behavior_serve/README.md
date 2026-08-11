# 暗光场景行为识别实时服务算子

| 项 | 值 |
| --- | --- |
| 算子名称 | 暗光场景行为识别实时服务 |
| 算子类别 | 一级类别：事件；二级类别：行为识别 |
| 算子类型 | 处理算子（常驻服务形态） |
| 提交包名 | `事件_行为识别_暗光场景行为识别实时服务.zip` |
| 主入口 | `main.py` |
| 运行镜像 | `darkpipe-operator:0.2.0`（与处理算子、测试验证算子共用同一镜像） |
| 架构 | amd64，需 1 张 NVIDIA GPU |
| 配套算子 | 暗光场景行为识别（批处理）、暗光场景行为识别测试验证 |

## 一、算子功能

持续拉取一路实时视频流，逐帧做低照度增强与行为识别，一边对外提供实时结果，一边把有价值的
片段留存下来。与批处理算子（`事件_行为识别_暗光场景行为识别.zip`）用的是**同一套 darkpipe
代码路径、同一个镜像、同一份权重**，区别只在运行形态：

| | 批处理算子 | 本算子（实时服务） |
| --- | --- | --- |
| 运行形态 | 跑完退出 | 常驻，直到容器被停止或 `run_seconds` 到期 |
| 输入 | 一段有界视频/流片段（须配 `max_frames`） | 一路永不结束的实时流 |
| 结果 | 一个完整 mp4 + 事件 JSON + 汇总 JSON | 三条 HTTP 流 + 一个个行为片段 mp4 |
| 适用 | 录像回放、离线复核 | 值班大屏演示、实时留证 |

### 对外接口

服务监听 `serve_port`（默认 8000），需要平台把该端口映射出来才能从容器外访问：

| 接口 | 类型 | 用途 |
| --- | --- | --- |
| `GET /stream` | MJPEG（`multipart/x-mixed-replace`） | 实时演示画面，时延最低 |
| `GET /live.flv` | HTTP-FLV | **与输入流同一种格式**，既有播放器/平台可直接接入 |
| `GET /hls/index.m3u8` | HLS | 浏览器原生支持，代价是分片时延 |
| `GET /events` | SSE（`text/event-stream`） | 每识别出一次行为推一条 JSON |
| `GET /health` | JSON | 健康检查、实时帧率/时延、事件数、片段统计、各路出流存活状态 |
| `GET /config` | JSON | 当前生效配置 |

画面内容三种格式完全一致（增强 + 超分 + 底部识别标签条），差别只在封装与时延，用哪几种由
`stream_formats` 决定（默认 `mjpeg,flv`）：

| 格式 | 时延 | 怎么打开 | 适用 |
| --- | --- | --- | --- |
| `mjpeg` | 最低（一帧） | 浏览器 `<img src=...>`、`ffplay`、`cv2.VideoCapture` | 值班大屏、程序再消费 |
| `flv` | 低（约 1 秒内） | `ffplay http://IP:8000/live.flv`、flv.js、多数国产平台播放器 | **对接既有流媒体/监控平台** |
| `hls` | 高（约 3-6 秒） | 浏览器原生 `<video>`、Safari/移动端 | 非实时观看、无插件网页 |

HLS 的时延是格式本身决定的（要先攒满分片才能下发，这里已经压到 1 秒一片、保留 6 片），
**不适合当实时演示用**，列在这里是因为某些浏览器场景只认它。

`/live.flv` 每个观看端各起一个编码进程，因此有 `max_flv_clients`（默认 4）并发上限，超出返回
503。**需要更多观看端时不要调大这个值，而应该填 `rtmp_push_url`**：把流推给已有的流媒体服务器，
由它去分发给任意多个客户端，本算子只负责推一路出去，成本恒定。

三种格式都由镜像内的 ffmpeg 从同一份 JPEG 画面转封装而来——那份 JPEG 本来就要为 `/stream` 编一次，
所以多开一种格式只多一次转封装，不会多一次画面编码。`stream_formats` 里只留 `mjpeg` 时完全不启动
ffmpeg。若镜像里没有 ffmpeg（0.2.0 之前的旧镜像），选了 `flv`/`hls` 会在**启动时**直接报错退出并
说明原因，而不是等到有人打开地址才发现打不开。

`/events` 每条事件形如：

```
event: recognition
data: {"frame_index": 480, "timestamp": 32.06, "label": "Falling", "confidence": 0.83,
       "topk": [["Falling", 0.83], ["Other", 0.09]], "model": "behavior", "window": 32}
```

### 行为片段留存

事件流里每出现一次**不属于 `clip_skip_labels`（默认 `other`）**的行为，算子就把那段画面切成
一个独立的 mp4 存下来，目录结构为：

```
<clip_dir 或 hdfs_output_dir>/<会话>/<行为>/<时间>_<行为>_<序号>.mp4
                                          <时间>_<行为>_<序号>.json
```

例如 `/opt/darkpipe/clips/20260101_090000_1/falling/20260101_091233_falling_0007.mp4`。
同名 `.json` 记录标签、置信度、片段内各标签计数、帧数、帧率、时长与收尾原因。

三件事值得单独说明，因为它们决定了片段好不好用：

1. **片段包含触发之前的画面。** 识别事件是在动作发生**之后**才产生的，从触发点开始录只能录到
   动作的尾巴，因此有 `clip_pre_sec`（默认 2 秒）的前置缓冲。
2. **连续事件合并成一个片段，不是每个事件一个文件。** 识别器每隔半个窗口就产生一次事件，一次
   五秒的跌倒会连续命中十几次；算子只在最后一次命中之后静默满 `clip_post_sec` 才收尾，因此得到
   的是一个完整片段而不是十几个互相重叠的碎片。`clip_max_sec` 是兜底：画面里一直有人活动时，
   片段不会无限增长成一个巨大文件。
3. **片段按管线实测帧率写入，不是按源流帧率。** 实时模式会丢帧以保时延，用源流帧率写会让片段
   看起来忽快忽慢；用实测帧率写出来的片段是正常速度。

**写盘与上传都在独立线程上做**，识别主循环只往一个有界队列里丢帧。这是为了守住「实时时延
≤ 1 秒」这条硬指标——磁盘或 HDFS 抖一下不能把识别拖慢。队列满时会丢帧并打印累计丢帧数（片段
仍会保存，只是略有跳帧），不会阻塞识别。

### 事件日志 `events.jsonl`

除了片段，**每一条识别事件都会按行追加**写到：

```
<clip_dir>/<会话>/events.jsonl
```

每行一个 JSON，字段与 `/events` 推送的完全一致，另加一个 `wall_time`（本地时钟，便于和监控
录像对时）：

```json
{"frame_index": 886, "timestamp": 43.04, "label": "Drinking water", "confidence": 0.9045,
 "topk": [["Drinking water", 0.9045], ["Other", 0.0159]], "model": "behavior",
 "window": 32, "wall_time": "2026-01-01 09:12:33"}
```

**为什么要有它**：片段旁边的 `.json` 只记录了被切成片段的行为，`other` 和被 `clip_skip_labels`
过滤掉的事件都不在里面；而 `/events` 是实时推送，**平台一旦没把 `serve_port` 暴露出来，事件流
就完全取不到**（规范里算子无法声明端口映射）。这个文件和片段一起落在挂载出来的目录上，不依赖
端口是否开放。实测：把 `drinking water` 也加进跳过名单后，**片段 0 个、事件仍完整记录 31 条**。

服务停止时，若填了 `hdfs_output_dir`，日志会整体上传一份到 HDFS（它是全程追加的，不像片段那样
录完一个传一个，所以只能收尾时传；容器被强杀时以 `clip_dir` 上那份为准）。写入同样走独立线程，
不阻塞识别；`/health` 的 `event_log` 字段给出已写条数与丢弃条数。

**片段先写本地临时目录，录完整了才整体搬到 `clip_dir`。** `clip_dir` 通常是挂进容器的 NFS，
如果逐帧直接写过去，网络时延就落在写盘线程上，一次卡顿超过队列容量就开始丢帧；而且浏览目录的
人会看到一个正在长大的半截 mp4。搬运是一次顺序拷贝，不在关键路径上，因此 `clip_dir` 指向 NFS
是安全的，且目录里只会出现**完整可播**的片段。

## 二、输入参数

| 参数名 | 中文名 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `video_path` | 实时流地址 | String | 无（必填） | 国标/rtsp/http(s)(flv、hls)；也可填容器内本地视频文件（会循环播放，便于没有摄像头时演示） |
| `enhance` | 低照度增强算法 | String | `retinexformer` | `off` / `retinexformer` |
| `sr` | 超分辨率算法 | String | `bicubic` | `off` / `bicubic`；只影响画面与片段，不影响识别 |
| `sr_scale` | 超分放大倍数 | Int | `2` | `2` / `3` / `4`，`sr=off` 时忽略 |
| `recognize` | 行为识别模型 | String | `behavior` | 仅 `behavior`，不提供 `off`（无事件即无片段） |
| `reco_span_sec` | 识别窗口时长(秒) | Float | `1.0` | 每次判定依据最近多少秒的画面 |
| `label_bar` | 叠加标签条 | Bool | `true` | 演示画面与片段底部是否叠加识别结果 |
| `gpu_ids` | GPU卡号 | String | `0` | 实时流不做多卡分片，只取第一张 |
| `ckpt_dir` | 权重目录 | String | `/opt/darkpipe/ckpts` | 镜像内已预置，一般无需修改 |
| `serve_port` | 服务端口 | Int | `8000` | 所有接口都在这个端口上 |
| `stream_formats` | 实时流输出格式 | String | `mjpeg,flv` | `mjpeg` / `mjpeg,flv` / `mjpeg,flv,hls` / `mjpeg,hls` |
| `rtmp_push_url` | 推流地址(可留空) | String | 无（**可留空**） | 推到外部流媒体服务器，支持 `rtmp://ip:1935/应用/流名` 与 `rtsp://ip:8554/流名`（RTSP 固定走 TCP）；留空不推 |
| `stream_bitrate` | 出流码率上限 | String | `4M` | 所有 H.264 出流的码率上限；**不要留空**，见下 |
| `max_flv_clients` | FLV并发观看上限 | Int | `4` | 超出返回 503；需要更多观看端请改用推流 |
| `jpeg_quality` | 演示流画质 | Int | `85` | 1-100，调低省带宽 |
| `max_stream_fps` | 演示流最大帧率 | Float | `15.0` | 只限制推流，不限制识别与片段 |
| `clip_dir` | 片段保存目录 | String | `/opt/darkpipe/clips` | 容器内路径，建议挂 NFS 后本地浏览 |
| `clip_pre_sec` | 片段前置时长(秒) | Float | `2.0` | 向前多保留的秒数 |
| `clip_post_sec` | 片段后置时长(秒) | Float | `2.0` | 最后一次命中后再录多久才收尾 |
| `clip_max_sec` | 片段最长时长(秒) | Float | `30.0` | 单个片段的时长上限 |
| `clip_skip_labels` | 不保存的行为 | String | `other` | 逗号分隔；填空表示全都保存 |
| `hdfs_output_dir` | 片段HDFS目录(可留空) | String | 无（**可留空**） | 留空则不推 HDFS，只保留本地一份 |
| `run_seconds` | 运行时长上限(秒) | Float | `0` | `0` = 一直运行到容器被停止 |

### 必填参数

按规范「有 `default` 字段即代表该参数必填、不能为空值」——本算子里没有 `default` 字段的只有
三个参数，其余 19 个不填也会取默认值：

| 参数名 | 是否必填 | 说明 |
| --- | --- | --- |
| `video_path` | **必填** | 每路摄像头的流地址都不同，没有通用默认值 |
| `hdfs_output_dir` | **可留空** | 留空即表示不推 HDFS，只保留 `clip_dir` 里那一份 |
| `rtmp_push_url` | **可留空** | 留空即表示不往外部流媒体服务器推流 |

后两个之所以不能给 `default`，正是因为规范规定「有 default 即必填不能为空」——要让一个参数
可以留空，就只能不写 `default`。这个区别在算子代码里也是显式的：`video_path` 是
`required=True`，另外两个是 `default=""`。

## 三、输出结果

本算子唯一的框架级输出是 `session_json`（会话信息），因为**真正的产出是三条实时流和片段文件，
不是某一个结果文件**。用户填写的参数一律在 `inputs` 里，`outputs` 只有这一个由框架下发路径的
文件。

| 参数名 | 中文名 | 说明 |
| --- | --- | --- |
| `session_json` | 会话信息 | 会话号、状态、流地址、各接口地址、片段本地/HDFS 目录、事件日志路径、生效配置；停止后补写事件总数、片段数与上传结果 |

`session_json` **在服务刚启动时就先写一份**（`status: running`），停止时再补写完整统计
（`status: stopped`）。这样即使容器被直接停掉，框架收走的也不是一个空文件。同一份内容还会写
一份到 `clip_dir/<会话>/session.json`，挂 NFS 浏览时不用回头去找框架收走的那份。

```json
{
  "session": "20260101_090000_1",
  "status": "stopped",
  "source": "http://10.0.0.5:21305/live?app=liveonly&stream=copy_1",
  "endpoints": {"stream_mjpeg": "http://<容器地址>:8000/stream", "events_sse": "..."},
  "clip_dir": "/opt/darkpipe/clips/20260101_090000_1",
  "hdfs_clip_dir": "hdfs://user@ip:port/a/b/20260101_090000_1",
  "clips": {"session": "...", "clips_saved": 7, "clips_abandoned": 0, "frames_dropped": 0},
  "events_total": 812, "hdfs_uploaded": 14, "hdfs_failed": 0
}
```

## 四、部署与验证

平台调度时由框架下发命令行；本地验证可以直接跑，`run_seconds` 让这个常驻服务变成一次有限时长
的任务：

```bash
docker run --rm --gpus all -p 8000:8000 \
    -v /tmp/pkg:/opt/darkpipe/op:ro \
    -v /path/to/clip.mp4:/data/in.mp4:ro \
    -v /mnt/nfs/darkclips:/opt/darkpipe/clips \
    -v /tmp/out:/out \
    -w /opt/darkpipe/op darkpipe-operator:0.2.0 \
    /opt/conda/envs/darkpipe/bin/python -u main.py \
      --video_path /data/in.mp4 --enhance retinexformer --sr bicubic --sr_scale 2 \
      --recognize behavior --reco_span_sec 1.0 --label_bar true --gpu_ids 0 \
      --ckpt_dir /opt/darkpipe/ckpts --serve_port 8000 \
      --stream_formats mjpeg,flv --rtmp_push_url "" --max_flv_clients 4 \
      --jpeg_quality 85 --max_stream_fps 15 \
      --clip_dir /opt/darkpipe/clips --clip_pre_sec 2 \
      --clip_post_sec 2 --clip_max_sec 30 --clip_skip_labels other \
      --hdfs_output_dir "" --run_seconds 60 \
      --session_json /out/session_json.json
```

跑起来之后：

```bash
curl -s http://localhost:8000/health           # 帧率、时延、事件数、片段数、各路出流状态
curl -N http://localhost:8000/events           # 事件流（Ctrl-C 退出）
ffplay http://localhost:8000/stream            # MJPEG，或浏览器直接打开
ffplay http://localhost:8000/live.flv          # HTTP-FLV，与输入流同格式
ffplay http://localhost:8000/hls/index.m3u8    # HLS（需 stream_formats 里带 hls）
ls -R /mnt/nfs/darkclips/                      # 已保存的行为片段
tail -f /mnt/nfs/darkclips/*/events.jsonl      # 事件日志（不依赖端口是否开放）
```

推流给已有流媒体服务器（观看端数量就不再受本算子限制）：

```bash
      --rtmp_push_url rtmp://媒体服务器:1935/live/darkscene     # RTMP
      --rtmp_push_url rtsp://媒体服务器:8554/darkscene          # RTSP（自动走 TCP）
```

两种都已对着真实流媒体服务器（mediamtx）实测：服务端登记到推流会话，再把流拉回来确认是
H.264 640x528。**RTSP 固定使用 TCP 传输**——ffmpeg 默认走 UDP，需要 RTP/RTCP 一对端口穿过
中间的 NAT 与防火墙，在数据中心里常见的结果是握手成功、画面却过不去；TCP 全部走已经建立的
那一条连接。

**码率上限不是可选项。** 增强后的暗光画面噪声多，而 H.264 的比特几乎都花在噪声上：不设上限
时实测 **1080p 约 44 Mbit/s、4K 约 179 Mbit/s**。链路吃不下时不会优雅降级，而是画面出现**绿色
残缺块**（帧被截断，缺失的行在 yuv420p 里就是绿色），最后服务器直接断开连接。`stream_bitrate`
默认 `4M`，按下游带宽调整即可（4K 演示建议 8M 以上）。

**出流进程会自动重启。** 推流/HLS 的 ffmpeg 进程若因网络抖动或服务器断开而退出，会按 1s 起、
翻倍至 30s 封顶的退避自动重连（`/health` 的 `push_restarts`/`hls_restarts` 给出次数），不需要
重启整个算子。`/live.flv` 的每客户端进程不重启——它的 stdout 就是那次 HTTP 响应，进程没了这次
响应也就结束了，客户端重连即可。

推流失败的排查顺序：
1. 看 `/health` 的 `push_error`，里面是 ffmpeg 的原话；
2. **确认那个地址是"推流点"而不是"拉流点"**——摄像头/网关给的 RTSP 地址通常是让你去*拉*的，
   不能往里*推*，这是最常见的一种错；
3. 确认端口与路径符合服务器配置，需要鉴权时把用户名密码带进 URL；
4. 服务器若强制要求 UDP（少见），当前版本不支持，需要加一个开关，告诉我即可。

## 五、注意事项

1. **本算子不会自行结束**（除非填了 `run_seconds`）。这是常驻服务的固有形态：摄像头不断流，
   容器就不退出。平台若要求算子必须终止，请用 `run_seconds` 把它调度成一次有限时长的任务，
   或改用批处理算子。
2. **需要平台把 `serve_port` 映射出来**，否则所有接口只能在容器内访问。
3. **`flv` / `hls` 依赖镜像内的 ffmpeg**（`darkpipe-operator:0.2.0` 起随镜像提供）。用旧镜像
   又选了这两种格式时，服务会在**启动时**就报错退出并说明原因，不会等到有人打开地址才发现；
   只选 `mjpeg` 则完全不依赖 ffmpeg。`/health` 里有 `hls_alive`/`push_alive` 字段，出流进程
   万一中途挂掉能直接看出来。
4. **`clip_dir` 要挂载到容器外**（NFS 或宿主机目录），否则容器一销毁片段就没了——容器内的
   路径本身不具备持久性，这也是同时提供 `hdfs_output_dir` 的原因。
5. **`hdfs_output_dir` 留空即不推 HDFS**；填了 `hdfs://` 地址则依赖容器内的 HDFS 客户端，
   上传失败只打印告警不中断服务（本地片段始终保留）；填非 `hdfs://` 的普通目录则直接复制，
   不依赖 HDFS 客户端。
6. **断流会自动重连**，退避从 0.5 秒指数增长到 8 秒封顶，重连次数记录在 `/health` 的
   `reconnects` 字段；`capture_alive` 为 false 时 `/health` 返回 503，可直接作为存活探针。
7. **`video_path` 填本地视频文件时会循环播放**，这是为了在没有摄像头的环境下也能演示与验证，
   不是缺陷。
8. 任何未捕获异常都会打印完整调用栈并以非零退出码结束，框架据此判定任务失败。
