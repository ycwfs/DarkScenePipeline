"""Validate timed runs, decode outputs, and summarize descriptive performance evidence."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
RUNS = OUT/'runs'


def read(path: Path):
    return json.loads(path.read_text())


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def percentile(values: list, q: float) -> float:
    return float(np.percentile(values, q))


def main() -> None:
    assert read(RUNS/'complete.json')['completed']
    protocol = read(RUNS/'protocol.json')
    results = read(RUNS/'results.json')
    assert len(results) == 18
    names = [x['name'] for x in protocol['samples']]
    (OUT/'figures').mkdir(exist_ok=True)
    (OUT/'evidence').mkdir(exist_ok=True)
    shutil.copyfile(RUNS/'protocol.json', OUT/'evidence/protocol.json')
    shutil.copyfile(RUNS/'results.json', OUT/'evidence/results.json')
    verification = []
    summaries, records_by_name, trace = [], {}, {}
    for name in names:
        reference = read(ROOT/'reports/person-finetune-20261023/demo/events'/f'{name}.json')
        all_frames = []
        folders = [RUNS/f'repeat-{r}'/name for r in (1, 2, 3)] + [RUNS/'paced'/name]
        for folder in folders:
            result = read(folder/'result.json')
            frames = read(folder/'frames.json')
            events = read(folder/'events'/f'{name}.json')
            assert len(frames) == result['frames']
            assert sum(f['predictions'] for f in frames) == len(events) == len(reference)
            max_error, same_crops = 0., 0
            for event, expected in zip(events, reference):
                for key in ('track_id', 'frame_index', 'source_frames', 'label', 'mode', 'pose_valid'):
                    assert event[key] == expected[key], (folder, key)
                same_crops += event['crop_sha256'] == expected['crop_sha256']
                for key in ('rgb_probs', 'skeleton_probs', 'fused_probs'):
                    if expected[key] is None:
                        assert event[key] is None
                    else:
                        error = float(np.max(np.abs(np.array(event[key])-expected[key])))
                        max_error = max(max_error, error)
                        assert error < .002, (folder, key, error)
            # Decode the entire output: codec headers alone do not prove all frames exist.
            cap = cv2.VideoCapture(str(folder/result['video']))
            assert cap.isOpened()
            decoded = 0
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                assert list(image.shape[1::-1]) == result['output_size']
                decoded += 1
            cap.release()
            assert decoded == result['frames']
            verification.append(dict(run=str(folder.relative_to(OUT)), frames=decoded,
                events=len(events), identical_crop_hashes=same_crops,
                max_probability_abs_error=max_error, labels_modes_frame_ids_equal=True))
            if folder.parent.name != 'paced':
                all_frames.extend(frames)
            else:
                trace[name] = frames
        repeats = [r for r in results if r['name'] == name]
        fps = [r['processing_fps'] for r in repeats]
        core_fps = [r['frames']/sum(r['timings'][k] for k in
                    ('preprocess', 'detection_pose', 'person_actions')) for r in repeats]
        paced = trace[name]
        newest = [age for f in paced for age in f.get('newest_frame_age_seconds', [])]
        oldest = [age for f in paced for age in f.get('oldest_frame_age_seconds', [])]
        burst = [f['action_seconds'] for f in all_frames if f['predictions']]
        processing_decision = [f['decision_time']-f['start_time'] for f in paced if f['predictions']]
        total_frames = len(all_frames)
        stage_ms = {k: 1000*sum(f['stages'].get(k, 0) for f in all_frames)/total_frames
                    for k in ('enhancement', 'detection', 'pose', 'render')}
        stage_ms['action'] = 1000*sum(f['action_seconds'] for f in all_frames)/total_frames
        stage_ms['encode_write'] = 1000*sum(f['encode_write_seconds'] for f in all_frames)/total_frames
        total_ms = 1000*sum(r['elapsed'] for r in repeats)/total_frames
        stage_ms['other'] = total_ms-sum(stage_ms.values())
        assert stage_ms['other'] >= 0
        people = [f['people'] for f in all_frames]
        summary = dict(name=name, processed=repeats[0]['processed'], frames=repeats[0]['frames'],
            source_fps=repeats[0]['fps'], input_duration=repeats[0]['duration'], repeats=3,
            fps_mean=float(np.mean(fps)), fps_sd=float(np.std(fps, ddof=1)),
            fps_min=min(fps), fps_max=max(fps), fps_values=fps,
            shortfall_vs_15_fps=15-float(np.mean(fps)), required_speedup=15/float(np.mean(fps)),
            optimistic_serial_core_fps_max=max(core_fps),
            people_median=percentile(people, 50), people_p95=percentile(people, 95), people_max=max(people),
            predictions_per_pass=repeats[0]['predictions'], modes=repeats[0]['modes'],
            action_burst_median=percentile(burst, 50), action_burst_p95=percentile(burst, 95),
            action_burst_max=max(burst), action_people_max=max(f['predictions'] for f in all_frames),
            paced_decision_age_median=percentile(newest, 50), paced_decision_age_p95=percentile(newest, 95),
            paced_decision_age_max=max(newest), paced_decisions_within_1s=sum(a <= 1 for a in newest),
            paced_decisions_total=len(newest), paced_oldest_age_p95=percentile(oldest, 95),
            first_decision_wall_seconds=next(f['decision_time'] for f in paced if f['predictions']),
            first_decision_media_seconds=next(f['media_time'] for f in paced if f['predictions']),
            processing_decision_p95=percentile(processing_decision, 95),
            final_output_age=paced[-1]['output_age_seconds'],
            stage_mean_ms_per_frame=stage_ms, total_ms_per_frame=total_ms,
            peak_torch_allocated_gib=max(r['peak_torch_allocated_bytes'] for r in repeats)/1024**3,
            peak_torch_reserved_gib=max(r['peak_torch_reserved_bytes'] for r in repeats)/1024**3)
        summaries.append(summary)
        records_by_name[name] = all_frames
    save(OUT/'summary.json', summaries)
    save(OUT/'evidence/verification.json', verification)
    save(OUT/'evidence/paced-frames.json', trace)
    save(OUT/'evidence/throughput-frames.json', records_by_name)
    flat_keys = [k for k, value in summaries[0].items() if not isinstance(value, (dict, list))]
    with (OUT/'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=flat_keys)
        writer.writeheader()
        writer.writerows({k: r[k] for k in flat_keys} for r in summaries)

    labels = ['Drink16\nraw', 'Wave3\nraw', 'Wave28\nraw', 'Crowd0060\nprocessed',
              'Crowd0054\nprocessed', 'Feedback181324\nprocessed']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    x = np.arange(len(summaries))
    means = np.array([r['fps_mean'] for r in summaries])
    low = means - np.array([r['fps_min'] for r in summaries])
    high = np.array([r['fps_max'] for r in summaries])-means
    ax.bar(x, means, color=['#54789c']*3+['#bd8954']*3, yerr=[low, high], capsize=4)
    for i, r in enumerate(summaries):
        ax.scatter([i]*3, r['fps_values'], s=18, color='#263238', zorder=3)
        ax.text(i, r['fps_max']+.55, f"{r['fps_mean']:.2f}", ha='center')
    ax.axhline(15, color='#ae3e37', ls='--', label='Requirement: 15 fps')
    ax.set(xticks=x, xticklabels=labels, ylabel='Actual processing throughput (frames/s)',
           ylim=(0, max(19, means.max()+3)), title='Current per-person demo on one RTX 3090')
    ax.legend(loc='upper left')
    for ext in ('png', 'pdf'):
        fig.savefig(OUT/f'figures/throughput.{ext}', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    bottom = np.zeros(len(summaries))
    stages = ['enhancement', 'detection', 'pose', 'action', 'render', 'encode_write', 'other']
    colors = ['#78a6b7', '#6d8a58', '#adba83', '#bc6658', '#9589b3', '#cfb680', '#c5c7c8']
    for stage, color in zip(stages, colors):
        values = np.array([r['stage_mean_ms_per_frame'][stage] for r in summaries])
        axes[0].bar(x, values, bottom=bottom, label=stage, color=color)
        bottom += values
    axes[0].axhline(1000/15, color='#ae3e37', ls='--')
    axes[0].set(xticks=x, xticklabels=[s.split('\n')[0] for s in labels],
                ylabel='Amortized milliseconds per video frame', title='Where serial processing time is spent')
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].legend(fontsize=8)
    for name, label in zip(names, labels):
        fs = trace[name]
        axes[1].plot([f['media_time'] for f in fs], [f['output_age_seconds'] for f in fs],
                     label=label.replace('\n', ' / '))
    axes[1].axhline(1, color='#ae3e37', ls='--')
    axes[1].set(xlabel='Source video time (s)', ylabel='Frame age at encoder input (s)',
                title='FIFO paced replay: delay accumulates')
    axes[1].legend(fontsize=8)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT/f'figures/cost-and-delay.{ext}', dpi=180)
    plt.close(fig)

    throughput_table = '\n'.join(
        f"| {r['name']} | {'已增强回传' if r['processed'] else '原视频＋增强'} | "
        f"{r['people_median']:.0f}/{r['people_max']} | {r['fps_mean']:.2f}±{r['fps_sd']:.2f} | "
        f"{r['fps_min']:.2f}–{r['fps_max']:.2f} | {r['optimistic_serial_core_fps_max']:.2f} |"
        for r in summaries)
    latency_table = '\n'.join(
        f"| {r['name']} | {r['action_burst_p95']:.3f}/{r['action_burst_max']:.3f} | "
        f"{r['paced_decision_age_p95']:.2f}/{r['paced_decision_age_max']:.2f} | "
        f"{r['paced_decisions_within_1s']}/{r['paced_decisions_total']} | {r['final_output_age']:.2f} |"
        for r in summaries)
    stage_table = '\n'.join(f"| {r['name']} | " + ' | '.join(
        f"{r['stage_mean_ms_per_frame'][s]:.1f}" for s in stages) + ' |' for r in summaries)
    speedups = '\n'.join(f"- {r['name']}：距15 fps相差 {r['shortfall_vs_15_fps']:.2f} fps，"
                         f"达到该帧率需当前吞吐的 {r['required_speedup']:.2f} 倍。"
                         for r in summaries if r['fps_mean'] < 15)
    matched = sum(r['identical_crop_hashes'] for r in verification)
    checked = sum(r['events'] for r in verification)
    analysis = f'''# 逐人方案性能核验（2026-10-23）

## 裁决与适用范围

**当前独立逐人实现不能作为满足“≥15 fps、识别时延≤1秒”的方案交付。**
这是本机单张RTX3090上的实测；尚未把逐人方案接入平台多卡服务，不能据此宣称多卡也必然失败或已经达标。
本次沿用用户裁定：暂不使用物体检测先验；YOLO仍负责人体框定位。

## 固定配置与测量方法

- 最新逐人微调epoch-07，SHA256 `{protocol['model_sha256'][str(Path(protocol['checkpoint']).relative_to(ROOT))]}`。
- RTX3090 24GB物理GPU2（CUDA可见编号0）；测前空闲，GPU0/1有其他任务，因此主机CPU/内存总线并非完全独占。
- 处理长边720，RGB每人32帧400×224，batch=2；每个人约每0.5秒识别一次，原图照度门75，α=0.7，阈值0.75。
- 原视频照常增强，再检测/姿态/跟踪/逐人RGB与骨架融合/绘制/H.264落盘；回传片段裁旧标签条且不重复增强。
- 直接调用原 `demo.run_sample()`，计时封装不改变模型、跟踪、采样、预测或输出。两次完整预热组合（喝水16＋多人0060），之后每段三次交错复测。
- 初始化 {protocol['initialization_seconds']:.2f} 秒单独记录；正式吞吐包含解码、计算、绘制、编码收尾及原演示调试输出，不包含模型加载。
- 实际处理fps=处理帧数/墙钟耗时。保存MP4的25/29/30 fps是播放帧率，不是算法速度。
- 每段另做一次原视频帧率的FIFO节奏回放：按 `t0+frame_index/fps` 定义到达时刻，处理慢时不丢帧、不让源降速；这是本地排队模拟，不是RTSP/平台端到端测量。

## 三次吞吐实测

| 视频 | 输入链 | 候选人体数中位/最大 | fps均值±样本SD | 三次范围 | 仅三项计算的最乐观fps* |
|---|---|---:|---:|---:|---:|
{throughput_table}

*只累计原程序preprocess、detection_pose、person_actions耗时，剔除解码、跟踪、绘制、编码和其他成本，再取三轮最高值。
这是当前串行实现的乐观上界估计，**不是另一种实现的实测fps**。
人体数是检测/跟踪的候选数量，可能含重复框或误检，不等同于真实人数。
六段视频为代表性负载，未对完整数据集做性能分布统计；短片约1秒的历史预填期间尚未执行动作识别，会使整段平均吞吐偏乐观。

![三轮吞吐](figures/throughput.png)

图中点是三次实测、误差棒为最小至最大值。原视频与回传片段处理链不同，不可直接把两组差异都归因于人数。
原视频与拥挤回传样本均未达到15 fps；少人数的回传短片若达到15 fps，只能说明这一段较轻负载通过吞吐项。

## 时延与积压

| 视频 | 一次动作调用耗时P95/最大(s) | 最新输入帧→决策P95/最大(s) | ≤1秒的逐人决策数 | 末帧送编码器时年龄(s) |
|---|---:|---:|---:|---:|
{latency_table}

动作调用耗时来自三轮无节奏吞吐测试，一次调用处理该帧上所有到期的人（内部每批2人）；不是单个人时延。
决策时延来自一次节奏回放，按每条逐人输出统计，包含增强、检测、处理和排队，**不含前面收集约1秒观测窗的时间**。
同一批多人结果有相同完成时间，样本并不独立；P95只是描述性分位数，不是长期服务SLA保证。
更严格的“动作开始→识别”没有逐帧动作起始真值，本次不能验证；最老观测帧时延、首个结果到达时刻保存在summary.json。
编码器输入时间也不是远端播放器显示时间，实际网络/播放还会增加开销。

## 主要耗时（毫秒/视频帧，三轮合并平均）

| 视频 | 增强 | 人体检测 | 姿态 | 逐人动作 | 绘制 | 写编码器 | 其他 |
|---|---:|---:|---:|---:|---:|---:|---:|
{stage_table}

“其他”包含解码、缩放、跟踪/裁剪、调试输出、编码收尾等；姿态为每帧一次RTMO后关联到各人，
不是每个人再跑一遍RTMO。动作项包含每个人VideoMamba、骨架分类、融合及当前调试哈希。

![耗时与排队](figures/cost-and-delay.png)

图左说明需优先处理的计算成本；图右的阶跃对应到期人物动作批次阻塞串行帧处理。
即使降低或省去编码开销，原视频和拥挤样本的计算耗时仍不足以支撑当前15 fps目标。

## 验证与下一步

- 24段正式输出全部完整解码，共 {sum(r['frames'] for r in verification)} 帧；逐人事件 {checked} 条。
- 所有事件的帧号、轨迹、采样帧、类别、分支模式与已有最新模型演示一致；{matched}/{checked} 个输入crop哈希完全一致。
- 最大概率差 {max(r['max_probability_abs_error'] for r in verification):.8f}；固定模型与现有算法的等价性已核对，没有通过少识别人来提升fps。
- 需要优化的是逐人RGB的批处理/预处理及增强与识别的并行调度。多卡分工、增大批量、减少冗余拷贝可作为下一步实测方向，收益尚未验证。
- 仅降低输出视频标称fps不能让计算变快；仅丢帧也可能破坏逐人时间窗，需要同时验证有效动作输出频率和准确率。
- 本轮只增加性能测试和证据，现役算子、模型、Box包、镜像均未修改，无需重新上传。
'''
    (OUT/'analysis-report.md').write_text(analysis)
    (OUT/'stats-appendix.md').write_text(f'''# 统计口径

- 六个固定输入，每个3次热身后的同进程复测；共18次吞吐运行和6次单轮节奏回放。
- 吞吐报告算术均值、样本标准差（ddof=1）与观测最小–最大范围；不只报告最佳值。
- 范围是实际重复波动区间，不是95%置信区间。三次同机同片回放不能独立代表摄像头场景总体；不计算总体置信区间、不做显著性检验或宣称统计泛化。
- 相对15fps的绝对差与所需加速倍数如下，是工程差距而非模型因果效应。无假设检验族，故不进行多重比较校正。
{speedups}
- 时延P95用NumPy线性分位数，按逐人事件计数，同一帧多人高度相关；原始帧/调用记录完整保留。
- 原始场景含增强，平台回传不重复增强；两组不是控制全部其他变量的人数因果对比。
- 输入片段2–6.4秒，包含约1秒动作窗预填；不证明长期稳定吞吐、RTSP/NFS/网络性能或动作起始到识别的准确时延。
- 本机单GPU与平台现役多GPU整帧服务不同；没有借用旧整帧平台测速证明新逐人服务达标。
- PyTorch峰值显存不含ONNXRuntime全部分配；GPU快照见协议和每轮记录，不应将其当作精确进程峰值。

精确数表：summary.csv、summary.json；原始18轮：evidence/results.json；节奏回放：evidence/paced-frames.json。
''')
    (OUT/'figure-catalog.md').write_text('''# 图表说明

## throughput.png / throughput.pdf

- 目的：判断当前实现能否达到15 fps。数据：evidence/results.json，6样本×3复测。
- 柱为均值，散点为每轮结果，误差棒为观测min–max；红虚线15 fps。非95%CI。
- 阅读结论：原视频和拥挤输入均失败；轻负载回传结果不可外推至完整增强链或其他人数。
- 检查项：原始/已增强两类输入明确标记；单位为墙钟处理fps，不是输出视频头的fps。

## cost-and-delay.png / cost-and-delay.pdf

- 目的：定位瓶颈以及观察FIFO输入时延是否积累。
- 左：三轮逐帧平均耗时堆叠，虚线66.67ms/帧；右：一次paced回放帧年龄，虚线1s。
- 数据：evidence/throughput-frames.json、evidence/paced-frames.json及results.json。
- 阅读结论：多人动作批次耗时增加；达到期批次时出现时延跃升，串行回放不能及时消化输入。
- 检查项：RTMO是全帧一次；动作耗时是所有到期人物合计。右图在编码器输入处采样，不包含网络或播放器。
- 不能据此宣称多GPU提升倍数、因果人数曲线或长期服务SLA。
''')
    (OUT/'README.md').write_text('''# 逐人方案性能验证

结论：**当前独立逐人实现未满足 ≥15 fps、识别时延≤1秒。**
最新ROI VideoMamba、α0.7、阈值0.75、照度门75；不用物体先验。单张RTX3090、六段视频、每段三次吞吐复测及一次节奏回放。

- [详细结论与精确数表](analysis-report.md)
- [统计口径与限制](stats-appendix.md)
- [图表说明](figure-catalog.md)
- [结构化汇总](summary.json)、[CSV](summary.csv)
- [输入/代码/模型哈希与环境](evidence/protocol.json)
- [视频解码和算法等价核验](evidence/verification.json)

原始日志在benchmark.log，完整运行与视频在runs/（大文件本地保留）。生产算子、权重和交付包未改。

复现（仓库根目录，选择空闲GPU；默认runs/已存在时脚本拒绝覆盖）：

```bash
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \\
LD_LIBRARY_PATH="$PWD/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$PWD/.venv/lib/python3.10/site-packages/nvidia/cublas/lib:$PWD/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib" \\
.venv/bin/python -u reports/person-performance-20261023/scripts/benchmark.py --output /tmp/person-performance-replay
.venv/bin/python reports/person-performance-20261023/scripts/analyze.py
```

分析脚本读取本目录runs/，复测到其他目录时需明确选择对应分析输入，不能将两批结果混用。
''')
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f'PASS: {len(verification)} outputs, {checked} events, {matched} identical crops')


if __name__ == '__main__':
    main()
