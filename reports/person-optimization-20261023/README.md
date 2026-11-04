# 逐人方案优化候选

后续已增加真实网络输入/输出及逐人并发录制验证，见[实时流与片段核验](../person-stream-audit-20261023/README.md)。本目录原性能实验保持原口径；runtime的输入时间戳、输出和事件回调接口在后续扩展，测量时源码快照另有保存。

已实现GPU归一化、人物帧缓存、小批次逐批发布、单卡/多卡调度与限长实时输入；保持现有模型和融合参数。
本轮暂不启动轻量模型训练。单卡尚不能覆盖全部场景；三卡在本轮短实时回放中表现较好，平台与原始拥挤视频仍需验收。

- [详细结果与优化结论](analysis-report.md)
- [全帧吞吐精确汇总](summary.csv)
- [实时回放：帧率、时延、丢帧及逐人覆盖](live-summary.json)
- [204窗精度回归](accuracy-regression.json)
- [统计口径](stats-appendix.md)、[图表说明](figure-catalog.md)

在仓库根目录运行。设置实际空闲物理GPU；脚本使用可见编号0、1、2。

```bash
export CUDA_VISIBLE_DEVICES=2,3,4
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$PWD/.venv/lib/python3.10/site-packages/nvidia/cublas/lib:$PWD/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
.venv/bin/python reports/person-optimization-20261023/scripts/run_demo.py \
  --video real-actiondata-all/挥手3.mp4 --cards 3 --process-workers \
  --output /tmp/person-optimized-demo
```

单卡用`--cards 1`；双卡可先用`--cards 2`（线程后端），三卡加`--process-workers`。
`--mode latest-replay --seconds 30`按源帧率供帧并使用最新帧策略，记录丢帧和合并窗口。
默认all-frames保留全部帧并排空所有人物任务，适合检查结果；两种模式不可混用性能数字。
回传片段用`--processed`裁去旧标签条并避免再次增强；普通原始视频不要加此参数。

每次输出video.mp4、events.json、frames.json、result.json，已有result.json的目录拒绝覆盖。
最新帧模式的原始video.mp4只包含已处理帧、仍按源fps编码，播放时长会压缩；真实时序以frames.json源时间戳为准。
若目录中另有video_source_time.mp4，它通过重复显示上一个已处理帧恢复源时间轴，仅供观看，重复帧不计入处理fps。

复测入口：benchmark.py、profile_actions.py、accuracy_regression.py、live_replay.py；核验：verify_runs.py、verify_live.py；统计：summarize.py、make_report.py。
正式性能日志和完整视频保留在runs/、process-runs/、live/、live-extra/、soak/，不纳入Git。
测量中途增加实时功能，原始测量代码快照和各次protocol.json记录于evidence/，避免将旧结果误认为修改后代码的新测量。
生产算子、模型和镜像未更新；当前候选依赖仓库原有person_core和dark-demo-v2模块。
