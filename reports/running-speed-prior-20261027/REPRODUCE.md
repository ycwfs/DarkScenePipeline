# 复现与范围

工作目录 DarkScenePipeline；使用原镜像 darkpipe-operator:0.3.0。所有模型与正式四参数不变，候选基于上轮wave-pose-prior实验。已有结果禁止覆盖；复现实验先复制至新目录，并调整共同常量。

1. `scripts/prepare.py` 冻结协议、速度公式、XCLIP提示词及模型哈希。该时点最初策略为缺姿态严格拒绝；后续在test前记录了修订，见protocol-amendment.json。
2. `.venv/bin/python scripts/old_evidence.py`（从本实验目录运行须补全实际脚本路径）重建旧train/val同人同框姿态。原速度公式以当前采样点在处理画幅内的像素位移/平均人体框高/秒数计算；不以裁剪后相对坐标计算，以免抵消人体平移。
3. 容器运行 `scripts/frontend_long.py`，完整顺序重跑01前端，对原10个跑动触发逐个核对人物框和32采样源帧。保存long-running-evidence.json，07/09原无跑动候选，因此固定证据比较不需要新增速度。
4. 原图审核review/context-*.jpg及labels.json，儿童与成人关联采用用户反馈和原始时序，不以速度反推真值。
5. `scripts/choose_threshold.py` 只用旧val及授权的长视频反馈选参数，严格/跳过缺姿态两种同时记录；候选采用跳过。选0.35后锁定，才运行 `scripts/old_evidence.py --test`。train不参与选值，test不反向调值。
6. `scripts/build_candidate.py` 从上轮挥手候选复制隔离源码，新增Sample.frame_shape、ActionEngine过滤、runtime选项、配置/入口/清单参数。保留正式模型符号链接。`scripts/test_running.py` 15项CPU回归；参数合约及全部Python解析通过。
7. 真实候选完整入口：`scripts/run_candidate.py --number 1`，以及7、9；`scripts/shorts.py`生成5段无损新摘录，再用 `scripts/shorts.py --run`执行这5段和5段旧短片。所有有限离线完整帧，不丢旧输入，不作为实时速度测量。
8. `scripts/report.py`核对独立前向人物/帧/概率差、保存错误统计、所有视频完整FFmpeg解码及帧数，生成报告和浏览页。独立模型前向不保证逐位一致；主要阈值效果来自相同概率的严格回放。

典型运行命令（从仓库根）：

```bash
docker run --rm --network none --gpus device=0 --ipc=host \
  -v "$PWD:$PWD" \
  --entrypoint /opt/conda/envs/darkpipe/bin/python darkpipe-operator:0.3.0 \
  "$PWD/reports/running-speed-prior-20261027/scripts/run_candidate.py" --number 1
```

首次候选启动因实验脚本命名select.py覆盖Python标准库select而失败，未进入推理；已改名choose_threshold.py，原失败日志保留。宿主沙箱直接CUDA不可见，实际模型推理全部在批准的原容器内执行。曾出现一次自动审批服务503，之后同一审批路径重试成功，没有绕过审批。

不更新正式tar/zip、Box、镜像，不提交git；候选路径的ckpts为本机链接，不是独立交付包。完整试验结束时保留production-preserved.json及镜像ID核验。
