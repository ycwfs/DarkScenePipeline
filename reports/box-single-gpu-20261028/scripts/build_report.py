"""Descriptive analysis only: one finite live run per configuration, no inferential claims."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1]

def read(name):
    return json.loads((OUT/name).read_text())

def main():
    base, final = read('live-baseline-verification.json'), read('live-final-verification.json')
    demos, faults = read('demos-verification.json'), read('fault-verification.json')
    phases = {name:read(name+'-by-scene.json') for name in ('live-baseline','live-final')}
    figures=OUT/'figures'; figures.mkdir(exist_ok=True)
    metrics=[('Processed frames / source second',lambda d:d['main_frames_processed']/300),
        ('Median processing FPS',lambda d:d['fps_proc']['median']),
        ('Capture to output P95 (ms)',lambda d:d['capture_to_output_ms']['p95']),
        ('Action age P95 (ms)',lambda d:1000*d['action_result_age_seconds']['p95'])]
    fig, axes=plt.subplots(1,4,figsize=(13,3.3),layout='constrained')
    for ax,(label,fn)in zip(axes,metrics):
        values=[fn(base),fn(final)]
        ax.bar(['Original','Box candidate'],values,color=['#7c8795','#207c91'])
        ax.set_title(label,fontsize=10)
        for x,y in enumerate(values): ax.text(x,y,f'{y:.2f}',ha='center',va='bottom',fontsize=10)
        ax.set_ylim(0,max(values)*1.22);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('One RTX 3090; same 300 s RTSP source; original 1440 vs candidate 720 output',fontsize=11)
    for ext in ('png','pdf'):fig.savefig(figures/f'01-live-comparison.{ext}',dpi=170)
    plt.close(fig)
    labels=['Few people','Crowd / child','Crowd / walking','Crowd / phone','Crowd / camera']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    x=np.arange(5)
    for j,(name,title,color)in enumerate([('live-baseline','Original','#7c8795'),('live-final','Box candidate','#207c91')]):
        values=[v['fps']['median']for v in phases[name]]
        axes[0].bar(x+(j-.5)*.36,values,width=.36,label=title,color=color)
    axes[0].set_xticks(x,labels,rotation=25,ha='right');axes[0].set_ylabel('Median processing FPS')
    axes[0].axhline(15,color='#b34c40',ls='--',lw=1,label='15 FPS');axes[0].legend(fontsize=8)
    for name,title,color in [('live-baseline','Original','#7c8795'),('live-final','Box candidate','#207c91')]:
        timeline=read(name+'/source-timeline.json');begin=timeline['source_start_monotonic']
        rows=[json.loads(t) for t in (OUT/name/'frames.jsonl').read_text().splitlines()]
        rows=[r for r in rows if begin<=r['origin']+r['timestamp']<=timeline['source_end_monotonic']]
        axes[1].plot([r['output_monotonic']-begin for r in rows],[r['rss_bytes']/1024**3 for r in rows],label=title,color=color,lw=1)
    axes[1].set_xlabel('Source playback time (s)');axes[1].set_ylabel('Main process host RSS (GiB)')
    axes[1].legend();axes[1].set_title('Host RSS only; not Jetson unified memory',fontsize=10)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    for ext in ('png','pdf'):fig.savefig(figures/f'02-scene-memory.{ext}',dpi=170)
    plt.close(fig)
    def pair(label,fn,fmt='.2f'):
        a,b=fn(base),fn(final)
        return f'| {label} | {a:{fmt}} | {b:{fmt}} |'
    table='\n'.join(['| 指标 | 原 Box | 优化候选 |','|---|---:|---:|',
        pair('实际处理帧数 / 7500',lambda d:d['main_frames_processed'],'d'),
        pair('300 秒平均实际处理 FPS',lambda d:d['main_frames_processed']/300),
        pair('健康接口处理 FPS 中位数',lambda d:d['fps_proc']['median']),
        pair('健康接口处理 FPS 最低值',lambda d:d['fps_proc']['minimum']),
        pair('采集交付→输出 P95，ms',lambda d:d['capture_to_output_ms']['p95']),
        pair('动作结果年龄 P95，s',lambda d:d['action_result_age_seconds']['p95'],'.3f'),
        pair('动作结果年龄最大，s',lambda d:d['action_result_age_seconds']['maximum'],'.3f'),
        pair('输出最大间隔，s',lambda d:d['output_intervals_seconds']['maximum'],'.3f'),
        pair('主进程 RSS 峰值，MiB',lambda d:d['parent_rss_bytes']['maximum']/1024**2,'.1f'),
        pair('被新窗口替换的待处理任务',lambda d:d['coalesced_action_tasks'],'d'),
        pair('保存动作片段',lambda d:d['clips_saved'],'d'),
        pair('前后上下文完整的片段',lambda d:d['clips_context_complete'],'d')])
    scene_table='\n'.join(['| 输入时段 | 原 FPS 中位数 | 候选 FPS 中位数 | 候选最多检测人数 |','|---|---:|---:|---:|']+
        [f"| {label} | {b['fps']['median']:.2f} | {f['fps']['median']:.2f} | {f['people_max']} |" for label,b,f in zip(labels,phases['live-baseline'],phases['live-final'])])
    candidate_rate=final['main_frames_processed']/300
    gain=(candidate_rate/(base['main_frames_processed']/300)-1)*100
    probe_table='\n'.join(['| 同源等长离线探针 | 帧数 | 原FPS | 候选FPS | 候选GPU MiB |','|---|---:|---:|---:|---:|']+
        [f"| {s['id']} | {s['frames']} | {s['baseline_fps']:.2f} | {s['candidate_fps']:.2f} | {s['candidate_gpu_mib']} |"for s in read('probe-comparison.json')])
    demo_table='\n'.join(['| 样本 | 平均原L | 最多检测人 | 离线FPS | 片段 | GPU MiB | 主进程RSS MiB |','|---|---:|---:|---:|---:|---:|---:|']+
        [f"| {s['id']} | {format(s['brightness'], '.1f') if s['brightness'] is not None else '未知'} | {s['people_max']} | {s['fps']:.2f} | {s['clips']} | {s['gpu_mib']} | {s['rss_mib']:.0f} |"for s in demos['samples_detail']])
    recovery_table='\n'.join(['| 故障恢复动作 | health 恢复正常，秒 | 接收端重新收到画面，秒 |','|---|---:|---:|']+
        [f"| {k} | {v['health_ok_seconds']:.3f} | {v['receiver_activity_seconds']:.3f} |" for k,v in faults['recoveries'].items()])
    report=f'''# NVIDIA-Box 单卡优化分析 · 2026-10-28

本轮完成 Box 专用候选与开发机单卡验证。**目标 Orin Nano 4GB Super 尚未接入，不能称为 Orin 真机验收或保证 4GB 内存足够。**
用户提供的是 ARM64、34TOPS、CPU/GPU 共享 4GB 内存及 256GB NVMe。NVMe 容量不等于 GPU 内存；TOPS 也不能直接换算此模型组合的 FPS。JetPack 版本未知，部署脚本当前目标为 JetPack 6 / Python 3.10 / TensorRT 10+，仍需本机验证。

## 采用的修改

- 只改 Box 运行时及其打包/部署工具。平台源码、四个交付归档和原 0.3.0 镜像另以 SHA/镜像 ID 核对。
- 六个模型及现场权重保持：Retinexformer、人体 YOLO26m、RTMO-s、逐人 ROI VideoMamba-Tiny、ST-GCN、七类物体 YOLO26m。本轮没有训练或采用新轻量学生模型。
- 人物 RGB 帧缓存从 512 帧 float32 改成 128 帧 uint8，缓存本体约 **525 MiB→33 MiB**；取批次后执行原归一化。此数字不包括模型/激活/解码等总内存。
- 人框/骨架匹配和人物重叠检测使用向量化计算。单卡实时独立动作线程、每人公平轮转、待处理窗口同人合并，批次上限 2。合并计数显式可见；离线逐帧排空。
- ARM64 自动 TensorRT：人体/物体 FP16、姿态 FP32（关闭 TF32），128 MiB builder 工作区；缓存绑定 ONNX 哈希、GPU、架构和 TRT 版本。只交付 ONNX，engine 必须板端构建。x86 的 auto 为 ORT CUDA，测试 TensorRT 时显式指定。
- RTMO 的 rank-1 ArgMax 做等价 Unsqueeze/ArgMax/Squeeze 改写；10 帧原/改 ONNX 在 ORT 下最大误差 0。此检查不证明 TensorRT 全链逐位相等。
- 增强保持原 PyTorch FP16，CUDA Graph 默认关闭。默认直接输出 720 长边，原版默认显示放大至 1440；**模型处理输入均为 720**。可加 `--sr bicubic --sr_scale 2` 恢复显示放大，速度另测。
- 原参数维持 alpha .972 / 原图门35 / 动作 .925 / 物体 .581；挥手同腕3真实帧、跑动 .35 框高/秒；物体框、人物框、骨架、人/动作并发片段、前后2秒均保留。

## 真实单卡推拉流

同一 GPU0（RTX3090）、同一 300 秒 25FPS/7500 帧输入，先 180 秒不同亮度/少人，后 120 秒四类多人场景；RTSP 输入→实际 RTMP 主动推送→外部 FFmpeg 录制，同时测试 MJPEG/FLV/HLS 观看者重连。没有像素条码。原版输出1440，候选720，故这是整个部署配置的比较，不能将收益全部归于 TensorRT。

{table}

按 300 秒实际处理源帧计算，吞吐提高 **{gain:.1f}%**。丢旧输入以控制延迟：候选消费 {final['main_frames_processed']}/7500 帧，未逐帧执行所有输入。编码器设定15FPS不等于模型15FPS。动作窗口合并不会排除某个人，但中间窗口不再逐一识别；短促动作召回尚未全面验收。

![总体比较](figures/01-live-comparison.png)

这张图同时展示速度与延迟，避免只报更高显示帧率。候选仍不能保证所有场景达到15FPS或每次动作结果小于1秒。

{scene_table}

相同输入帧数的5个离线探针补充如下；各配置每样本仅一次，非独立重复实验，不与实时FPS混算：

{probe_table}

![人数分段与内存](figures/02-scene-memory.png)

拥挤场景仍是主要吞吐限制。RSS 是主 Python 进程，不包括所有编码/系统进程。RTX 独立显存与主机 RSS 不能直接相加当作 Jetson 统一内存；设备总容量、交换、温度及功耗需通过随包 tegrastats/benchmark 工具验证。

主流测试正常 SIGTERM 收尾，{final['decoded_videos']} 份视频严格完整解码通过；保存 {final['clips_saved']} 个片段，其中 {final['clips_context_complete']} 个有请求的前后上下文。写入失败、强制释放、放弃、写队列丢帧均0，输入重连 {final['reconnects']}、主动推流重启 {final['push_restarts']}、HLS 重启 {final['hls_restarts']}。health 状态分布为 {final['health_statuses']}。输出间隔超过1秒次数 {final['output_gaps_over_1s']}。健康接口可能把短时正在进行的编码写入标为 degraded；保留原始记录，不把它等同已证实断流，也不删除该状态。

本次唯一上下文不足片段在15秒分段上限处结束，前卷1.921秒、后卷0.719秒，元数据未把它标成完整。未裁的外部录制保留初始化及尾部黑画面；另生成299.933秒展示版full-output-5min.mp4，只裁两端连续黑画面，中间全部保留，另行严格解码。原始录制及裁剪边界见stream-trim.json。

延迟起于 OpenCV 交付解码帧，不包含相机网络缓冲；动作年龄统计包含本次会话启动尾部事件。只代表本机 loopback、本地盘、有限时间测试，不是无限流、Orin、平台 vGPU 或 NFS 保证。不同处理速度改变窗口相位和跟踪，片段/触发数量不能作为误报率对照。

## 故障与恢复

独立110秒测试，源暂停8秒、输出中继停止8秒、输入断开8秒以及观看者重连，均用本轮归档单GPU运行。

{recovery_table}

{faults['clips_saved']} 个片段，{faults['clips_context_complete']} 个上下文完整，其余断流/边界不足按元数据明示；写失败/强制释放/放弃均0，写队列丢帧 {faults['clip_frame_drops']}。采集代次 {faults['epochs']}，正常退出后自有进程残留0。接收端恢复活动包括警告画面或重复末帧，不等于新鲜源画面的端到端恢复时间。缓冲帧在暂停后仍可能完成，详见 fault-verification.json。

## 识别与可视化检查

- 独立解包版本 **107 项 CPU 回归通过**；真实启动器、六模型自检记录见 archive-entry/ 与 archive-check-env.log。
- 固定人物输入204窗的缓存优化检查：最终标签0变化，RGB概率最大绝对差0.00233424。仅验证同ROI/同姿态的动作路径，不是检测/跟踪/异步调度的全链无损证据。
- 最终姿态FP32配置完整跑23历史样本，{demos['frames']} 帧、{demos['events']} 人物窗、{demos['clips']} 个自动片段；{demos['decoded_videos']} 份完整输出/片段严格解码。覆盖原平均L约7–154、最多{max(s['people_max'] for s in demos['samples_detail'])}个检测人物；3份回传已增强，原L未知。样本含历史训练/反馈，非独立泛化准确率。
- **已知误过滤变化保留**：喝水27原基准5个drink窗，TensorRT候选3个，仍保留该短片的喝水动作片段。两个窗口因人体/姿态局部ROI与检测数值变化失去物体支持；原瓶分数约.5834紧邻.581阈值。将姿态恢复FP32未消除这项差异，因此不能声称全部精度不变，也未擅自降低阈值掩盖它。
- 手机/相机误认瓶、部分旧误跑等既有问题并未通过执行优化解决。23样本没有完整逐帧逐人GT；触发数量下降不记作纠错。
- 与历史正式23输入相比，18份样本最终类别计数一致。5份有变化，其中crowd-camera的喝水1窗变0，lighting-06的跑动1窗变0；另有多人轨迹导致other窗口数变化。详见historical-label-counts.json。这些计数不构成同窗精度或误报率统计；即使上述两个历史误触发在本次短片中消失，也不能称已解决该类误报。

{demo_table}

全部输出和片段：[index.html](index.html)。正式规则默认及每个模型训练来源见包内 models.json / box-models.json。

## 未采用或失败的候选

1. 首版 CUDA Graph 因 autocast 临时权重地址失效出现非法内存访问；关闭捕获时的缓存后修复，5样本197窗标签一致，但额外图池不适合4GB默认，故关闭。
2. TensorRT增强：原 ONNX 对 PyTorch 一个检查帧 PSNR67.55dB/MAE.0114；60帧 TensorRT FP16 最低PSNR27.88dB、最大MAE7.82，部分敏感层FP32后仍34.04dB/3.94，未达等价要求，拒绝进入交付。128MiB增强构建工作区不足的失败日志也保留。
3. 单卡动作batch1改善部分前端吞吐，但动作年龄P95升至1.119秒、max2.320秒，合并643窗；默认恢复batch2。
4. TRT姿态FP16是早期候选，live-trt-balanced/ 与 trt-final-demos/ 目录名虽有final字样，**不是当前交付精度配置**。最终依据 live-final/ 和 archive-demos/。
5. torch.compile增强仅微测25.42→18.03ms、首编译20.67秒；未集成，不计入本版收益。ORT FP16也只是微测。
6. TensorRT原RTMO rank-1 ArgMax解析失败已作等价图改写；TensorRT初始化改变locale导致中文读文件失败，已恢复locale并强制UTF-8。

## 可交付边界与下一步

本轮为功能完整的 Box 性能候选，**不是已验收的 Orin4GB 实时方案**。完整模型保留，x86单卡仍未全面达15FPS；仅凭这一轮优化不足以宣布4GB Orin目标达标。包内提供Jetson安装分支、本地engine构建、自检及测速工具；需匹配本机JetPack的ARM64 PyTorch/Triton/Mamba扩展。

若板端容量或速度仍不达标，下一轮应在固定现场验证集和误报约束下做增强蒸馏、人体/物体检测m→s及动作学生模型，分别验证漏报和误报后再替换。没有把尚未训练/验证的轻量模型作为当前功能完备的证明。
'''
    (OUT/'analysis-report.md').write_text(report)
    (OUT/'stats-appendix.md').write_text(f'''# 统计与协议边界

问题：相同源视频与单GPU条件下，Box部署候选能否提高实际处理吞吐、降低常驻内存，保持可用推流/保存。

比较单位是一次完整300秒运行：原版n=1、最终候选n=1。中间配置仅作失败/取舍记录，未挑最好一次当最终。每种配置没有独立seed或重复run；234等health点、7500源帧高度时间相关，不能当作独立重复。

{table}

效果量：按源时间的吞吐提升{gain:.6f}%；这是描述性差值，不是有统计保证的期望提升。
未做显著性检验、p值、运行间标准差或95%置信区间：n=1不支持估计它们，不用逐帧伪重复构造窄区间。P95是本次时间分布分位数，不是95%置信区间。没有统计假设检验族，因此不作多重比较校正。

原输出1440，候选720；模型输入同为720；缓存/调度/后端/显示编码同时变化，无法用这些数据单独归因。启动和engine构建不计入主300秒性能，源有75秒预卷；离线报告同时保存elapsed_with_startup和处理FPS。单次短片FPS可能受CUDA缓存/热身影响。

23输入为历史诊断覆盖，不是随机抽样或新独立测试；其中旧训练来源与线上反馈来源均公开，input hashes见inputs.json。没有完整逐人GT，不能从最终标签计数推出准确率/F1或误报率提升。暗图/已增强回传的照度语义分开。

内存：offline每.5秒采样nvidia-smi进程显存及/proc RSS，峰值为采样峰值；另记录torch allocated/reserved。live只记录主进程RSS。Orin CPU/GPU统一内存及全部系统进程尚无观测；不能将RTX两项直接相加或认定4GB一定足够。

再现脚本：run_live.py、instrumented_box.py、analyze_live.py、bench_offline.py、review_demos.py、run_faults.py、analyze_faults.py、build_report.py。最终独立解包运行时位于release-extracted/NVIDIA-Box；测试解释器native-test-env；TensorRT10.3仅装在实验目录。无平台镜像依赖变更。
''')
    (OUT/'figure-catalog.md').write_text('''# 图表目录

- `figures/01-live-comparison.png/.pdf`：原版与候选的300秒实际吞吐、health中位FPS、采集后延迟与动作年龄P95。来源live-baseline/final-verification.json。无误差线：每组只有一次运行。观察吞吐是否提高及动作延迟是否同时恶化；不能把编码15FPS当算法速度，不能把P95当置信区间。输出尺寸变化属于配置比较。
- `figures/02-scene-memory.png/.pdf`：分场景处理FPS及随时间的主进程RSS。来源by-scene.json与frames.jsonl。用来识别拥挤瓶颈和内存压力，指导板端验收；曲线不包含所有子进程，也不等于Jetson统一内存。每段帧是相关观测，不作独立显著性样本。
- `analysis-report.md`内精确数表与`summary-table.csv`：对应每次运行原始值。完整演示是模型推理输出，无生成式替代图片。
''')
    with (OUT/'summary-table.csv').open('w')as f:
        w=csv.writer(f);w.writerow(['metric','original','candidate'])
        for label,fn in metrics:w.writerow([label,fn(base),fn(final)])
    (OUT/'README.md').write_text(f'''# Orin Nano 4GB Box优化候选

实测为单张RTX3090，不是Orin真机验收。完整模型和规则保留，优化缓存/调度/后端及720直接输出。

- [完整分析与逐项数字](analysis-report.md)
- [5分钟实际推流视频、全部动作片段、23样本](index.html)
- [统计边界](stats-appendix.md) · [图表及解释](figure-catalog.md)
- [主流验收JSON](live-final-verification.json) · [故障恢复JSON](fault-verification.json) · [短片验收JSON](demos-verification.json)
- [Box安装与真机测试说明](../../NVIDIA-Box/JETSON.md)
- [交付候选ZIP](../../NVIDIA-Box_Orin4GB_candidate_20261028.zip)

300秒平均实际处理{base['main_frames_processed']/300:.2f}→{candidate_rate:.2f}FPS（+{gain:.1f}%）；health FPS中位{base['fps_proc']['median']:.2f}→{final['fps_proc']['median']:.2f}。未达到所有场景15FPS。{final['clips_saved']}动作片段，{final['clips_context_complete']}上下文完整，写失败0；故障测试另列。

已知精度代价：喝水27的5个原drink窗变3个，仍有动作片段；不是精度完全无损。4GB容量、ARM64扩展安装和真机FPS待设备端测量，包名保留candidate。历史FP16姿态目录trt-final-demos不是最终结果；以archive-demos和live-final为准。
''')
    print('Analysis, figures and report index written')

if __name__=='__main__':main()
