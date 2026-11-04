# 逐人实时流与动作片段核验（2026-10-23）

已接通独立逐人候选的 RTSP 输入、RTSP/RTMP 输出和“人物ID＋动作”并发完整画面录制。
三卡真实模型回放、输入/输出故障恢复及正常停止已实测。推荐关闭片段额外去噪；Retinexformer增强照常运行。
生产平台入口、默认权重、算子包、Box包和镜像均未替换，本目录是有限时长的验证候选。

- [详细结果与保存规则](analysis-report.md)
- [各轮结果与接收端时延](summary.json)
- [完整视频解码校验](evidence/video-verification.json)
- [4/8路并发录制结果](evidence/capacity-results.json)：动作事件人为注入，只验证录制容量，不代表8人识别精度或完整链性能。
- [最终代码与范围](evidence/final-manifest.json)

## 运行候选

从仓库根目录运行，使用已安装依赖的`.venv`。指定空闲物理GPU；脚本内部使用可见编号0/1/2。
目标流媒体服务器需预先启动并允许发布。

```bash
export CUDA_VISIBLE_DEVICES=2,3,4
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$PWD/.venv/lib/python3.10/site-packages/nvidia/cublas/lib:$PWD/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
.venv/bin/python reports/person-stream-audit-20261023/scripts/run_stream.py \
  --input rtsp://127.0.0.1:8554/input \
  --push rtsp://127.0.0.1:8555/enhanced \
  --output /tmp/person-live-check --cards 3 --seconds 240 --clip-denoise off
```

可重复传`--push`，分别发布RTSP和RTMP。SIGTERM触发正常收尾。
`health.json`记录输入新鲜度、推流重启及片段计数；`events.jsonl`保存逐人结果；`clips/`保存视频与同名JSON。
网络异常时直播显示`INPUT LOST - RECONNECTING`，缺失画面不会伪造进动作片段。
`network/video.mp4`仅用于处理帧诊断，按原fps写入，丢帧后其播放时长会压缩；**动作片段单独按时间戳重采样，保持实际时长**。

复现整套本机故障实验用`scripts/stream_experiment.py`，需要`.tools/mediamtx`；官方版本及SHA256记录于`evidence/relay-tool.json`。
`scripts/clip_capacity.py`为独立录制容量测试，`tests/test_person_stream_clips.py`核对并发、前后文、分段、断流与异步边界。

当前运行器保留逐帧诊断和计时明细，适合有限时长核验。无限常驻服务仍需将诊断记录改为有界滚动记录、接入平台健康接口和NFS/HDFS交付后再验收。
