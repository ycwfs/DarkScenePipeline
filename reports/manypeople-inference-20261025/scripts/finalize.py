"""Publish a compact file index after all source and result checks have finished."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

REPORT = Path(__file__).resolve().parents[1]


def main() -> None:
    while True:
        path=REPORT/'verification.json'
        verification=json.loads(path.read_text()) if path.exists() else {}
        if verification.get('finished'):
            break
        time.sleep(15)
    completion=json.loads((REPORT/'completion.json').read_text())
    if not completion['success'] or not verification['passed']:
        raise RuntimeError('Batch or output verification did not pass; inspect job logs.')
    inputs=json.loads((REPORT/'input-verification.json').read_text())
    assert len(inputs)==9 and all(r['pass'] for r in inputs)
    items=verification['items']
    expected=dict(person_mode=True,input_processed=False,reco_min_conf=.75,lum_gate=30.,
        skel_fusion_alpha=.95,skel_no_person='soft',fuse_mode='alpha',proc_max_side=720,
        sr='bicubic',sr_scale=2,clip_pre_sec=2.,clip_post_sec=2.,clip_max_sec=15.,
        pose_overlay=True,label_bar=False,enhance='retinexformer')
    config_checks=[]
    for item in items:
        args=json.loads((REPORT/'jobs'/f'{item["number"]:02d}'/'effective-arguments.json').read_text())
        mismatches={k:args.get(k) for k,v in expected.items() if args.get(k)!=v}
        assert not mismatches,(item['number'],mismatches)
        config_checks.append(dict(number=item['number'],gpu_ids=args['gpu_ids'],configuration_matches=True))
    (REPORT/'config-verification.json').write_text(json.dumps(dict(expected=expected,jobs=config_checks),
        ensure_ascii=False,indent=2)+'\n')
    videos=REPORT/'videos';videos.mkdir(exist_ok=True)
    events=REPORT/'events';events.mkdir(exist_ok=True)
    counts=Counter();clips=0;total_frames=0;complete_context=0
    for item in items:
        counts.update(item['label_counts']);clips+=item['clips_total'];total_frames+=item['frames']
        complete_context+=sum(bool(c['metadata'].get('context_complete')) for c in item['clips'])
        for source,target in ((Path(item['output']),videos/f'{item["number"]:02d}_处理后.mp4'),
                              (Path(item['events']),events/f'{item["number"]:02d}_events.json')):
            if not target.exists(): os.link(source,target)
    package=json.loads((REPORT/'package-verification.json').read_text())
    archive=Path(package['archive'])
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==package['sha256']
    rows=[]
    for item in items:
        rows.append(f'| {item["number"]:02d} | {Path(item["source"]["absolute_path"]).name} | '
            f'{item["duration"]:.2f} | {item["frames"]} | {item["events_total"]} | '
            f'{item["clips_total"]} | [播放](videos/{item["number"]:02d}_处理后.mp4) |')
    total_seconds=sum(i['duration'] for i in items)
    short_context='；'.join(f'编号 {item["number"]:02d}，{clip["metadata"]["closed_because"]}，'
        f'后置 {clip["metadata"]["post_context_available"]:.2f} 秒'
        for item in items for clip in item['clips'] if not clip['metadata']['context_complete'])
    report=f'''# 10-25 现场多人视频推理：已完成

[逐段播放、下载和动作片段检查页](index.html)。[完整结果视频目录](videos/)按 01–09 编号；[逐人事件目录](events/)与编号对应。简短文件名是已验证结果的硬链接，不重复编码。原始完整目录及片段元数据保留在 jobs/NN/results/。

全部 **9 段、{total_seconds/60:.2f} 分钟、{total_frames:,} 帧**已完成，输出 1440×808 H.264、保持各自源帧率，无音频、无测流黑白条。共 {sum(i['events_total'] for i in items):,} 条人物窗口预测、{clips} 段自动动作片段。预测次数不是独立动作次数，T 不是固定身份。

| 编号 | 输入原名 | 秒 | 帧数 | 事件 | 片段 | 完整结果 |
|---|---|---:|---:|---:|---:|---|
{chr(10).join(rows)}

使用当前 dark-demo-v2_person_tau075_20261024.tar.gz 独立解包；{package['files_verified']} 个载荷文件逐一核验。原 darkpipe-operator:0.3.0 镜像，ID `{completion['image_id']}`。结束后归档 SHA256 再次核对为 `{package['sha256']}`，未修改平台代码包、模型或镜像。

动作阈值 **0.75**、亮度门 **30**、RGB 权重 **0.95**、soft；每人 ROI VideoMamba＋同人骨架 ST-GCN；当前现场微调人体、姿态与动作模型，公开 NTIRE 增强权重；无骨架平滑、无物体先验。处理长边 720，输出放大 2 倍；动作片段前后各 2 秒，15 秒重叠分段。

独立文件并行，单个视频保持完整连续时间线。使用物理 GPU 0/1，每段分别以一张卡计算前端、另一张卡计算动作，具体分配见各任务参数；调度器逐步增加独立视频并发；接管调度时保留已有推理进度。使用正式离线入口逐帧处理并等待动作结果，不丢源帧；与直播为限制延迟而跳过旧帧的调度不同。本机有其他计算任务，本轮不作为实时速度验收。

## 核验

- 原始 9 段均通过 FFmpeg 严格完整解码；80,316 帧与容器记录匹配。
- 9 段完整输出逐帧解码，帧数与输入及推理统计一致，时长差小于 0.12 秒。
- {clips} 段动作视频均通过严格完整解码；逐段事件 JSON 数量和 schema_version=3 校验通过。
- 片段保存失败、强制释放、未完成和写入错误检查通过；上下文覆盖完整 {complete_context}/{clips} 段。其余记录：{short_context or '无'}。这不等于真实动作边界的保证。
- 全部进程正常退出，代码包哈希保持不变。详见 [完整核验](verification.json)、[输入核验](input-verification.json)、[结果索引 CSV](summary.csv)。

数据没有逐人动作真值标注，本次仅推理和输出完整性核验，**不把预测当作准确率**。全部预测和片段保留供人工检查，未删除疑似误报或漏识区间。人物窗口类别统计：`{json.dumps(dict(counts),ensure_ascii=False)}`。

原始压缩包未改，解压映射和哈希见 ../../data-10-25-manypeople/extraction-manifest.json。[输入概览](input-contact-sheet.jpg)／[结果概览](output-contact-sheet.jpg)。启动参数、运行日志、完成状态保存在 jobs/NN/，整批命令与状态见 protocol.json、completion.json；[九份实际配置核对](config-verification.json)。

scripts/run_observed.py 仅记录标量进度和完成项，不改变模型输入、概率、绘制或保存逻辑。diagnostic-sequential/ 是主动停止的顺序试运行记录，不纳入最终结果。复现需使用新结果目录，执行 scripts/launch_parallel.py，再执行 scripts/review.py --watch 和 scripts/finalize.py。
'''
    (REPORT/'README.md').write_text(report)
    (REPORT/'STATUS.md').write_text(f'已完成：9/9 段，{total_frames} 帧，{clips} 个动作片段。入口：index.html。\n')
    summary=dict(files=9,frames=total_frames,seconds=total_seconds,clips=clips,
        events=sum(i['events_total'] for i in items),context_complete=complete_context,
        label_counts=dict(counts),package_unchanged=True,verified=True)
    (REPORT/'final-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
