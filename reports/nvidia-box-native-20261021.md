# NVIDIA Box 完整原生部署交付（2026-10-21）

已将盒子从单模型工具扩展为与平台同源的完整本地流程。无需容器或算子注册：安装环境后，`run.sh --video_path ...` 自动选择文件离线、目录批处理或在线流服务；支持MJPEG/FLV/HLS、外部RTMP推送、本地动作片段、事件与会话JSON。旧单模型脚本保留供研究，生产入口改为run.sh。

交付：[NVIDIA-Box.zip](../NVIDIA-Box.zip) 与 [日期包](../NVIDIA-Box_20261021.zip)，逐文件校验见 [manifest](../NVIDIA-Box/manifest.json)，使用说明见 [README](../NVIDIA-Box/README.md)。平台同步两处正常停止修复，另生成 `dark-demo-v2_nativefix_20261021.tar.gz/.zip` 并同步默认包；10-19日期包保留。镜像与模型均未修改。

## 实现与范围

`NVIDIA-Box/runtime/` 从平台v2的main.py、oputil.py、darkpipe代码与字体逐字节复制；完整8份平台权重放在weights/，另保留旧盒子crop名称，共9文件。打包器 `scripts/package_nvidia_box.py --sync` 同步后校验来源哈希，拒绝漂移与额外运行文件，按白名单确定性打包，包含版本清单。

run.py调用平台解析器，仅覆写本地默认路径、自动会话JSON、GPU0及空extra_hosts；所有算法参数原样传递。以exec启动独立进程，保留信号/退出码，清除开发环境PYTHONPATH，配置ORT动态库路径。输出默认包内outputs/，`--output-dir`与`--clip_dir`等价。相对输入/显式输出按调用者工作目录解析，模型默认定位包内weights/。

setup_env.sh使用独立Python3.10 venv和匹配torch2.7/cu12/cxx11abiTRUE的Mamba扩展wheel，rtmlib单独--no-deps安装，避免CPU ORT覆盖GPU ORT。实跑从空环境安装成功，固定解析版本在constraints.txt；不安装驱动、不修改系统Python。只验证Linux x86_64，未适配Jetson/aarch64。

默认pn2、reco_min_conf0.75、720、照度门75、alpha0.7、pose_overlay=true、pre/post2秒、quality录制均不变。盒子默认单卡；实时多卡最后一张识别，其余增强；离线本地片段启用时只用首卡保证连续时间轴。

## 实测与修复

开发机原生新环境，无Docker；主要用物理GPU1，批处理用当时空闲GPU4。版本快照：[environment-versions.json](nvidia-box-native-20261021/environment-versions.json)。部署安装初次自检发现开发期间README变化导致manifest失效，重打包后独立解包自检全部通过，说明哈希校验能实际检测漂移。

- [125项最终回归](nvidia-box-native-20261021/verified-regression.xml)全部通过。此前一项仅给启动1秒的故障注入测试在并发CPU负载下取消lifespan启动，未走到预期clip-close故障；单独复核通过，测试改为3秒以覆盖正确故障位置。
- 单视频离线75帧→75帧H.264 1440×874，4条喝水事件、1段动作片段，候选人物骨架正常。目录中文/带空格文件名2/2成功，合计150帧、8事件、2片段。详见[完整原生证据](nvidia-box-native-20261021/evidence/native.json)、[批处理证据](nvidia-box-native-20261021/evidence/batch.json)。目录只是同一视频的两份硬链接，用于验证编排，不是独立准确率测试。
- 在线使用真实1080p视频的本地HTTP MJPEG循环源。MJPEG可读；FLV/HLS各90帧、6秒可完整解码；主动RTMP推送接收正常。约65秒SIGTERM，SSE连接仍在，最终exit0/stopped，6段动作片段可解码、磁盘22事件，SSE21条（订阅稍晚），健康HTTP200有55次。
- 最终录制forced/failed/unfinished/write_errors均0；frames_dropped254、clips_abandoned1。默认quality过载仍可丢帧/放弃未开录片段，不能保证无损留存；可按现场CPU能力显式设clip_denoise=fast，本次未更改默认。

真实压力测试先后发现两处正常停止边界，均先复现、再修复、再用相同完整流程回放：

1. [修复前证据](nvidia-box-native-20261021/evidence/native-run-2-failure.json)：尚未提交start的片段在停止时被当作写盘失败，虽然6段已接收片段全部收尾。修复为实时过载弃段计数和告警，与运行期间3秒开录超时一致；离线仍不准丢片段。测试覆盖正常写盘与真实OSError，后者仍失败。
2. [第二次修复前证据](nvidia-box-native-20261021/evidence/cleanenv-run-failure.json)：原外层处理线程join只有5秒，正常模型清理未完成就标记shutdown失败，而片段write_errors已为0。外层扩至30秒（内层worker仍5秒），给正在结束的推理、worker退出、模型/CUDA释放留出整体时间；超时仍报错。新增慢清理5.2秒成功与卡住失败测试。

代码/权重/输入输出行为没有另写一套。共享修复同时落在平台源代码和盒子runtime，并更新两种交付包，原0.3.0镜像不需要重建。

## 复现

```bash
# 生成盒子包：仓库内执行，模型和样例已恢复到各目录
python3 scripts/package_nvidia_box.py --sync --output NVIDIA-Box_20261021.zip

# 独立解包后按包内README安装；或使用已有相容环境
CUDA_VISIBLE_DEVICES=1 /path/to/python NVIDIA-Box/check_env.py
/path/to/python reports/nvidia-box-native-20261021/scripts/verify_native.py \
  --box /path/to/unpacked/NVIDIA-Box --python /path/to/python \
  --source real-actiondata-0909/喝水16.mp4 --gpu 1 --output /path/to/new-results

# 真实批处理：目录中放两个视频（允许中文和空格），会话文件自动创建
CUDA_VISIBLE_DEVICES=4 /path/to/python /path/to/unpacked/NVIDIA-Box/run.py \
  --video_path /path/to/videos --output-dir /path/to/new-batch-results

# 回归
/path/to/python -m pytest -q tests/test_nvidia_box_native.py tests/test_v2_remaining_fixes.py \
  tests/test_v2_serve_lifecycle.py tests/test_v2_delivery.py tests/test_v2_pose_overlay.py \
  tests/test_v2_platform_feedback.py tests/test_v2_clip_sequence.py
```

版本归档未含安装好的6.9GB环境；安装时获取依赖。代码、权重、字体、公开样例均已包含。
最终归档验证与SHA256见同目录 `release.json`。临时解包目录、日志和视频保留本地、不进Git。

本轮不验证实时FPS/时延达标、长期零中断或新场景准确率；没有连接实体盒子及实际RTSP摄像头。
pn2训练/留出结论仍见10-19报告，本次没有重新训练、调整阈值或改动决策策略。

最终ZIP独立解包后10项入口测试、66份文件哈希、自检真实CUDA前向全部通过。
最终安装器使用锁定constraints和前次下载的本地wheel复核成功；日期包与默认包逐字节相同，
SHA256 `38099bf27911a0a1c20558b182e0ebf9324f909c2f0b701648f88e4f46b350d0`，249,069,938字节。
