# 挥手骨架先验复现

工作目录为DarkScenePipeline，实验目录为reports/wave-pose-prior-20261027。正式平台/Box文件、模型和原0.3.0镜像不改；用户明确选择沿用正式7类物体检测器，不使用80类或8类候选。

## 固定条件

alpha=.972、原图亮度门35、动作.925、物体.581、增强长边720、soft回退、骨架不平滑、前后卷各2秒。主规则在protocol.json冻结：当前1秒窗口，同一只腕严格高于双肩，三关节置信度均≥.5，至少3个不同源帧。32帧RGB采样中的重复帧只计一次。低亮/亮度未知/纯RGB跳过，明亮但缺姿态拒绝wave。判断只用当前及过去窗口，无额外前向或未来帧等待。

implementation/wave_prior.py为唯一规则实现。累计3为主方案，1/5/8累计与连续3/5/8仅作同证据诊断；连续条件不跨超过.12秒的观测间隔。旧val用于判断是否有调严依据，历史test和新长片不得隐式重新选参。

## 旧窗口回归

evaluate_old.py使用上轮四参数实验的inputs.npz、predictions.json及原来源清单；只对原wave候选新增几何判断。对于受门35影响的旧源，取gate35独立跟踪回放及此前低门pose缓存；其余使用原帧缓存。每个源帧通过同一track_id、完全一致的检测框匹配姿态，原图尺寸/裁旧栏/偶数缩放与生产一致，再还原归一化关键点。旧概率、原标签和来源split不变。

535源/4234窗口包含无wave源；239个原wave窗口、7648个含重复帧的采样输入核验。原波形的全部GT覆盖与未知标签分开。此为历史缓存回归，不是重新执行全部模型或新独立测试。

旧演示选择依据既有真实挥手标注及原结果：val2/test2/train2，再加入用户此前指定的挥手3。初版清单因按文件名排序附加了多个冗余暗场例，后精简为上述7段；没有依据新先验成绩筛选，完整旧窗口回归不受演示选择影响。

## 原镜像真实长短片

run_videos.py在GPU0/1/2处理新01/07/09完整视频，随后复用上轮原三段短片。run_old_videos.py在GPU3/4/5处理7段旧真实挥手短片。paired_video.py每帧只运行一次正式增强/人体/跟踪/姿态/RGB/ST-GCN/7类物体模型，原结果复制后增加wave先验，再分别绘制、记录事件和保存片段。逐事件断言上游概率、人物、源帧、物体证据相同。

wave_shorts.py补充4段12秒FFV1无损原片摘取：01号18秒儿童摆臂、01号261秒被否决候选、07号90秒站立候选、09号133秒抬手候选。它们是看到长片触发后的针对性诊断，同时包含保留和过滤例，不是新的独立测试，也不参与选帧数。短片重启跟踪及采样相位，和完整视频分开报告。

review_videos.py生成所有wave保留/过滤组的原帧与真实关节证据，review-labels.json保存助手辅助复核，待确认动作不算FP或TP。未触发时段未穷尽审核，不能据此估计全视频召回率。视频/保存校验由verify.py完整FFmpeg解码完成，不只检查文件头。

review_context.py额外生成每组首触发前1.5秒至末触发后1秒的原图时序，覆盖多窗口组。审核不根据是否满足几何规则反推真值，低位指示/儿童摆臂等不确定动作单列。短片与长片同一现实动作在标注中关联，不重复计算独立纠错。

```bash
.venv/bin/python reports/wave-pose-prior-20261027/scripts/prepare.py
.venv/bin/python -m pytest -q tests/test_wave_pose_prior.py
.venv/bin/python reports/wave-pose-prior-20261027/scripts/evaluate_old.py
.venv/bin/python reports/wave-pose-prior-20261027/scripts/run_videos.py
.venv/bin/python reports/wave-pose-prior-20261027/scripts/run_old_videos.py
.venv/bin/python reports/wave-pose-prior-20261027/scripts/wave_shorts.py
.venv/bin/python reports/wave-pose-prior-20261027/scripts/review_videos.py
.venv/bin/python reports/wave-pose-prior-20261027/scripts/review_context.py
# 首次复现须先执行下述候选构建与两段实际入口核验，并完成人工审核标注。
.venv/bin/python reports/wave-pose-prior-20261027/scripts/verify.py
.venv/bin/python reports/wave-pose-prior-20261027/scripts/report.py
```

本目录已有结果，prepare与部分运行脚本显式拒绝覆盖。复现实验应先复制到新的实验目录，不要删除既有证据后重跑。需要本地原视频、上轮缓存、正式ckpts、原镜像和CUDA；训练/下载不属于本轮。

## 隔离候选真实接入

build_candidate.py复制正式源码到operator/dark-demo-v2，模型目录仅作本机符号链接；把相同wave_prior.py接入ActionEngine，在物体先验之后、事件发布之前执行；新增PipelineConfig、CLI及suanzi参数wave_pose_min_frames=3（0关闭）。参数合约校验通过。这个目录可在本机原镜像里运行，不是可直接上传的完整交付归档。

probe_candidate.py调用该候选实际service.run_offline处理挥手3，并与配对实验核对22个事件的人物、采样、最终类别和先验证据；这段原L<35，验证实际入口和暗门跳过。加`--wave-short new-07-standing-veto`则核验12秒亮场入口，86个事件的相同人物/帧/标签/先验均通过，否决2次wave且不保存挥手片段。

亮场独立运行的RGB概率最大差0.004866，融合最大差0.004729、骨架差0，最初2e-4数值复现检查失败，保留probe-bright-candidate.log。用`--verify-existing`复核保存输出，单独记录全部类别/证据一致和概率差，不提高误差阈值后称通过。主配对实验始终共享同次上游概率，已逐项严格相等核验。正式源树、Box、归档和镜像身份由check_production.py确认。

```bash
.venv/bin/python reports/wave-pose-prior-20261027/scripts/build_candidate.py
docker run --rm --network none --gpus device=3 --ipc=host \
  -v "$PWD:$PWD" \
  -v "$PWD/reports/wave-pose-prior-20261027/operator/dark-demo-v2:/app:ro" \
  --entrypoint /opt/conda/envs/darkpipe/bin/python darkpipe-operator:0.3.0 \
  "$PWD/reports/wave-pose-prior-20261027/scripts/probe_candidate.py"
# 同一命令追加 --wave-short new-07-standing-veto，进行亮场入口核验。
```

所有推理为有限离线测试；没有新增实时吞吐、断流、NFS或无限流的稳定性结论。
