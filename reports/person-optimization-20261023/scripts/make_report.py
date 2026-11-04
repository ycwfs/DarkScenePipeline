"""Write the decision report from completed performance and accuracy evidence."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def main():
    summary, live = read(OUT/'summary.json'), read(OUT/'live-summary.json')
    accuracy = read(OUT/'accuracy-regression.json')
    names = ['喝水16','挥手3','留出_挥手28','多人0060','多人0054','回传留出_181324']
    table=[]
    for name in names:
        rows=[next(r for r in summary if r['name']==name and r['cards']==cards and r['backend']==backend)
              for cards,backend in [(1,'thread'),(2,'thread'),(3,'process')]]
        table.append(f"| {name} | {rows[0]['baseline_fps']:.2f} | "+' | '.join(
            f"{r['fps_mean']:.2f}±{r['fps_sd']:.2f}"for r in rows)+' |')
    live_table=[]
    for row in live:
        if row['run'].startswith('soak'):
            continue
        live_table.append(f"| {row['name']} | {row['cards']} | {row['effective_fps']:.2f} | "
            f"{row['decision_age_p95']:.3f}/{row['decision_age_max']:.3f} | "
            f"{row['completed']}/{row['submitted']} | {row['source_dropped']}/{row['source_frames']} |")
    soak = [r for r in live if r['run'].startswith('soak')]
    soak_text = '持续运行检查尚未完成，暂不作稳定性结论。'
    if soak:
        soak_text = '\n'.join(f"- {r['name']}：输入{r['source_duration']:.0f}秒，"
            f"实际{r['effective_fps']:.2f}新帧/秒，决策P95/最大{r['decision_age_p95']:.3f}/{r['decision_age_max']:.3f}秒，"
            f"完成{r['completed']}/{r['submitted']}窗口，超过1秒{r['late_decisions']}条，"
            f"丢弃过时源帧{r['source_dropped']}/{r['source_frames']}，最多{r['max_candidate_people']}个人体候选。"for r in soak)
        soak_text += '\n\n同一三卡进程连续处理两种输入，各300秒；短视频循环，跨片边界重置跟踪。它是本地混合负载稳定性测试，不是连续10分钟天然长镜头，也不是平台RTSP验收。'
    verified=[]
    for folder in ('parity','runs','process-runs'):
        verified.extend(read(OUT/folder/'verification.json'))
    live_verified = read(OUT/'evidence/live-verification.json')
    report = f'''# 逐人方案工程优化与验证（2026-10-23）

## 结论

工程优化显著改善了速度，当前没有证据要求立刻把现有VideoMamba换成轻量模型。
但不同人数、输入链和卡数的结果不同，不能宣称所有配置全面达标。

- 单卡：全帧处理的原始视频从8.33–10.11fps提升至约15–19fps；实时双人挥手通过，亮场三候选样本只有14.07个新帧/秒，尚未全面满足15fps。
- 双卡：适合本轮少人原始视频及四/八候选回传场景；极拥挤样本虽然画面17.30fps，仍有动作积压、74个待处理窗口合并，最大决策时延1.522秒，未达标。
- 三卡：一张前端卡、两张动作卡，动作卡采用独立进程。30秒拥挤回放17.73个新帧/秒、453/453个已提交窗口完成、最大决策时延0.605秒，短测通过。
- 三卡并非普遍比双卡快：八候选回传三卡19.65fps低于双卡24.80fps，跨进程复制和调度有成本，不能按卡数线性推算。
- 拥挤和四/八候选视频均为已增强回传片段，不重复增强。**没有原始拥挤视频的完整增强链实测，不能把这些数字外推为原始多人监控的验收成绩。**

本轮保持原权重、32帧400×224输入、每人约0.5秒更新、α0.7、动作阈值0.75、原图照度门75，未启用物体先验、未降频人体检测/姿态、未训练轻量模型。
这是独立逐人候选运行程序，未替换平台入口、生产权重、Box包或镜像；多人事件/片段的生产集成仍需独立实施和验收。
本轮原始视频链包括Retinexformer增强、人体/姿态检测、逐人动作融合、绘制及H.264编码；未包含平台的全部后处理、事件落盘和网络推流。

## 实际改动

1. 以固定pinned memory缓冲上传uint8人物图像，GPU执行与原实现相同的float32归一化；RGB仍使用原有FP16 autocast。
2. GPU缓存逐帧归一化ROI，键包含会话、人物ID、历史代次和源帧号。复用重叠时间窗及重复采样帧，LRU限定512帧，不复用VideoMamba隐藏状态。
3. 动态处理已到期人物窗口，单卡每批最多2人，多卡最多4人；不等待凑满批次，每批完成即发布。1/2/4/8/16批量微测显示大批吞吐增益趋于饱和，因此没有默认堆到16人。
4. CPU解码/缩放、GPU计算、CPU绘制/编码分工；各GPU只有一个推理提交者。单卡受控执行；多卡支持线程和独立进程两种后端。
5. 最新帧模式保留真实源帧索引和时间戳，输入队列只有一格；同一人物尚未开始的旧窗口可被新窗口替换，替换数量明确统计。旧代次/旧时间结果不会覆盖新人物结果。
6. 超过1秒的显示结果失效并显示“识别中”，不会将未完成计算伪装成other；已完成但低于0.75的具名动作仍显示“其他”。
7. 图像哈希等诊断默认关闭，验证模式可启用；H.264编码仍使用原libx264，尚未将NVENC或TensorRT收益计入。

入口：[run_demo.py](scripts/run_demo.py)。核心：[action_engine.py](scripts/action_engine.py)、[scheduling.py](scripts/scheduling.py)、[runtime.py](scripts/runtime.py)、[process_pool.py](scripts/process_pool.py)。

## 动作分支瓶颈证据

在16个真实验证集人物窗口上，每个batch设置两次热身、三次计时：

| RGB批量 | 原CPU归一化路径 | uint8上传＋GPU归一化 | 新路径窗口吞吐 |
|---:|---:|---:|---:|
| 1 | 147.96ms | 45.06ms | 22.2窗/秒 |
| 2 | 264.72ms | 60.89ms | 32.8窗/秒 |
| 4 | 514.27ms | 115.11ms | 34.8窗/秒 |
| 8 | 1032.96ms | 225.60ms | 35.5窗/秒 |
| 16 | 2043.40ms | 447.87ms | 35.7窗/秒 |

batch4原路径中CPU归一化约366ms，RGB前向/回读约100ms。只把所有动作调用耗时当作VideoMamba核函数耗时会误判瓶颈。
同一batch下微测概率逐位一致；完整GPU缓存与变批量后的小幅数值差另做回归见后文。
数据量从float32变uint8减少4倍不等于模型前向加速4倍；完整链仍有增强、检测、姿态、跟踪、编码及调度成本。

## 不丢帧、不合并窗口的全帧对照

每个样本、每个后端配置预热后3次计时，包含解码、模型、绘制、编码收尾，排空动作任务。表为均值±样本标准差。
双卡选线程后端、三卡选进程后端；两种后端的完整结果都保留在summary.json，不隐藏效果较差的配置。

| 样本 | 原串行单卡fps | 优化单卡fps | 优化双卡fps | 优化三卡fps |
|---|---:|---:|---:|---:|
{chr(10).join(table)}

该表的拥挤场景仍未稳定超过15fps。后面的实时表是限长输入队列、会丢源帧的另一种运行方式，不能混成同一成绩。
原视频含增强；回传片段不重复增强。短片约1秒观测窗预填阶段尚未产生动作负载，整段吞吐偏乐观。

![全帧吞吐](figures/throughput.png)

## 原帧率实时回放

源端按25/27/29/30fps独立供帧，消费者不让源降速；输入过载时覆盖尚未处理的旧帧。
这里fps=实际消费的不同源帧数/源时间，**不把重复输出帧算作处理帧**。
“最新帧→决策”截止动作worker计算完成，包含前端及任务排队和模型计算，不包含结果返回后的绘制、推流、播放器延迟，也不包含约1秒观测窗预填；它不是动作起点到识别的延迟。
四/八候选片段各20秒，其余各30秒，均由短片循环；每次跨片重置人物历史，避免把视频尾首拼成假动作。

| 样本 | GPU数 | 新帧fps | 决策P95/最大(s) | 完成/提交人物窗 | 丢源帧/源总帧 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(live_table)}

该表仅统计可组成有效观测窗的人物任务，不能解释为所有进入画面的人都立即得到动作类别。
跟踪丢失、原片很短、跨片预填、首次结果等待都会影响人物可视化覆盖率；逐人发布间隔和具有有效观测窗时的显示覆盖率在live-summary.json中完整列出。
三卡拥挤样本的逐人发布间隔P95/最大约0.575/0.643秒，未做到严格每0.5秒更新；0.5秒是源时间触发目标，实际帧间隔和执行时间会造成偏差。
该样本具备有效观测窗的人物帧中，约73.4%显示了尚未过期的识别结果，其余主要受首次结果等待、跨片重置和轨迹变化影响。453/453完成不代表画面中的每个人始终都有动作标签。
实时丢源帧改变观测采样，不能借用下面204窗的等价回归声称整个实时系统精度完全不变。拥挤0060这次所有已完成结果为other，但不构成新的泛化或漏检证明。

![动作处理和实时性能](figures/action-and-live.png)

## 持续运行检查

{soak_text}

## 正确性与误差

- 204个既有逐人测试窗全部回放，RGB类别、融合类别和分支模式0变化。
- RGB宏F1 {accuracy['rgb']['macro_f1']:.4f}、other误报{accuracy['rgb']['other_false_positives']}/154；融合宏F1 {accuracy['fused']['macro_f1']:.4f}、other误报{accuracy['fused']['other_false_positives']}/154，均与原候选一致。
- 最大RGB/融合概率绝对差{accuracy['max_rgb_error']:.6f}/{accuracy['max_fused_error']:.6f}，来自批次/数值执行差异，本次没有跨过决策阈值。
- {len(verified)}段输出完整解码，共{sum(r['decoded_frames']for r in verified)}帧，{sum(r['events']for r in verified)}条逐人事件；类别、分支、源帧和轨迹均与原演示一致。18段诊断回放还逐条核对裁剪哈希。
- 另{len(live_verified)}段实时/持续回放共{sum(r['decoded_frames']for r in live_verified)}帧完整解码，{sum(r['events']for r in live_verified)}条结果通过帧数和任务数守恒核对；这项检查不代表丢帧后分类精度等价。
- 7项CPU回归通过，覆盖缓存淘汰时不覆盖当前批、跨人/跨代次隔离、过载合并公平性、旧结果拒收、停止时失败传播和输入源不随消费者降速。
- 未更改模型权重，不需要重新训练；本轮未进行平台容器、网络推拉流或片段上传验收。

## 是否需要轻量化模型

**暂不启动模型轻量化训练。** 本轮证明了原来的大部分动作调用开销可以通过工程方式削减，而且三卡在已测短实时场景中保留原模型即可达到帧率和最新窗口帧时延目标。

若部署要求是单卡覆盖所有亮度和十几人场景，当前仍不够，轻量化可作为下一阶段候选；但先继续细分前端的人体/姿态关联、跟踪裁剪、跨进程数据复制成本。
现有增强＋YOLO＋RTMO本身也占大量时间，只缩小VideoMamba不能保证完整原始视频链达标。
后续先用实际平台配额和较长的原始多人视频验证容量；若动作GPU仍持续满载、逐人窗口处理率不足，再训练更少帧/更小输入的学生模型并单独检验喝水召回与other误报。
'''
    (OUT/'analysis-report.md').write_text(report)
    (OUT/'stats-appendix.md').write_text('''# 统计与测量约定

- 同一空闲RTX3090设备组，模型/源数据固定；GPU0/1存在其他用户计算，主机CPU/内存总线并非独占。
- 原串行结果来自同日先前基准，不是与每次优化运行交错随机化的因果试验；加速比是描述性对比。
- 线程方案1/2/3卡与进程方案2/3卡，各6输入×3次，共90次正式全帧吞吐运行；另30次原帧率FIFO排队回放、18次带诊断一致性回放。
- 吞吐给出均值、样本标准差(ddof=1)、min/max及原始数据。3次同机复测不是独立摄像头场景，不给总体95%CI或p值；不做显著性和多重比较推断。
- 动作微测每设置2次预热、3次计时，共5批量×2路径×3记录；只包括RGB处理，不等于完整融合/视频链性能。
- 15个限长输入回放为每配置/每场景一次；20/30秒短视频循环，不提供可靠SLA置信区间。持续运行部分另外报告。
- 同一批多人的结果时间相关，P95/P99为描述性NumPy线性分位数，不能把事件当作独立样本做显著性检验。
- 时延以真实源索引/源fps计算，不以处理完成时间重新给帧打时间戳；最新帧、最老观测帧和动作起始是不同口径。
- 输入总帧=不同已处理源帧+被覆盖源帧；提交动作窗=已完成+待处理窗口合并。每条守恒关系均检查。
- 准确率回归仅支持固定204窗及原6段不丢帧回放的等价性。最新帧策略改变时间采样，需另用连续动作起止标注评估漏检；这批204窗已经被查看过，后续不能作为反复调参后仍未见的新测试集。
- 四/八/十几候选人数来自检测跟踪，包含可能的误检/重复框；拥挤输入已经增强，不代表原始拥挤摄像头的完整增强成本。
- 短片跨片重置会反复经历观测窗预填，降低持续动作负载；循环测试主要检验工程稳定性，不证明全天候容量。
''')
    (OUT/'figure-catalog.md').write_text('''# 图表说明

## throughput.png / .pdf

数据：summary.json与上轮基准。用途：比较保持模型、帧和人物窗口的工程优化。
柱为3轮均值；完整SD/min/max在报告和CSV中。原始与回传输入不能混为相同增强成本。
结论：少人原始视频明显提速，拥挤全帧仍未稳定15fps；多卡收益不是线性的。

## action-and-live.png / .pdf

左图来自profile.json：真实ROI上CPU归一化和GPU归一化调用时间，证明主要数据处理瓶颈；不代表整个模型核函数同倍加速。
右图来自live-summary.json：20/30秒限长输入回放的新帧fps与最大决策年龄，虚线分别15fps和1秒；每点一个场景/配置，不是独立重复估计。
用途：同时看帧率和动作时效，避免把双卡拥挤的画面流畅误判为识别达标。
局限：源帧有丢弃、人物窗口有合并，需同时核对完成/提交与逐人更新率；原始拥挤视频成本未测。
''')
    (OUT/'README.md').write_text('''# 逐人方案优化候选

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
.venv/bin/python reports/person-optimization-20261023/scripts/run_demo.py \\
  --video real-actiondata-all/挥手3.mp4 --cards 3 --process-workers \\
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
''')
    print('Report updated;',len(soak),'soak cases available')


if __name__=='__main__':main()
