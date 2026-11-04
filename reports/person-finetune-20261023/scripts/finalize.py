"""Publish a local experiment checkpoint and compact manifest after real evaluation/demo checks."""
import hashlib
import json
from pathlib import Path
import shutil
import torch

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    selection=json.loads((OUT/'selection.json').read_text())
    result=json.loads((OUT/'analysis/results.json').read_text())
    dataset=json.loads((OUT/'training-data.json').read_text())
    completion=json.loads((OUT/'test-completion.json').read_text())
    assert completion['completed'] and completion['selection_epoch']==selection['epoch']
    assert completion['data_sha256']==sha(OUT/'training-data.json')
    demos=json.loads((OUT/'demo-comparison.json').read_text())
    assert len(demos)==10 and all(r['matched_person_windows'] and r['skeleton_unchanged']for r in demos)
    assert len(json.loads((OUT/'training-history.json').read_text()))==9
    source=Path(selection['path']);assert sha(source)==selection['sha256']
    models=OUT/'models';models.mkdir(exist_ok=True)
    name='videomamba_t_behavior_32f_person_roi_20261023.pth'if selection['epoch']else 'selected_unchanged_pn2.pth'
    target=models/name
    if target.exists():assert sha(target)==sha(source)
    else:shutil.copy2(source,target)
    model=torch.load(target,map_location='cpu',weights_only=False)
    if selection['epoch']:
        assert model['input_scope']=='tracked_person_roi'
        assert model['training_data_sha256']==sha(OUT/'training-data.json')
        assert set(model['seen'])=={r['label']for r in dataset['records']if r['split']=='train'}
    production=ROOT/'platform/dark-demo-v2/ckpts/videomamba_t_behavior_32f_realactall_full_pn2.pth'
    protocol=json.loads((OUT/'protocol.json').read_text());assert sha(production)==protocol['initial_sha256']
    card=dict(model=target.relative_to(OUT).as_posix(),sha256=sha(target),selected_epoch=selection['epoch'],
        input_scope='tracked_person_roi',input_shape='RGB uint8 32x224x400x3 before normalization',
        normalization='ImageNet mean/std; network tensor Bx3x32x224x400',
        initialization=protocol['init'],initialization_sha256=protocol['initial_sha256'],
        person_crop=dict(expansion=.15,letterbox=[400,224],padding=114,window_seconds=1.,frames=32),
        fixed_fusion=dict(alpha=.7,reco_min_conf=.75,raw_luma_gate=75),
        data_summary=dataset['summary'],test_metrics=result['selected'],
        baseline_test_metrics=result['baseline'],
        artifacts=dict(protocol_sha256=sha(OUT/'protocol.json'),training_data_sha256=sha(OUT/'training-data.json'),
                       annotation_sha256=sha(OUT/'annotations.json'),splits_sha256=sha(OUT/'splits.json')),
        limitations=['Historical field videos were mostly seen by initial pn2.',
                     'Single seed; classification metrics conditional on usable detections and reviewed labels.',
                     'Drink recall regressed from 11/11 to 8/11; all three misses are from drink34.',
                     'No local positive fight/talk/shake_hands; one training source each for throw/fall.',
                     'Fixed skeleton fusion can suppress a correct RGB action.',
                     'Do not substitute this ROI checkpoint into the whole-frame production pipeline.'],
        production_changed=False,image_rebuilt=False)
    (models/'model-card.json').write_text(json.dumps(card,ensure_ascii=False,indent=2)+'\n')
    (models/(name+'.sha256')).write_text(sha(target)+'  '+name+'\n')
    a,b=result['baseline'],result['selected']
    table='| 指标 | 原 pn2 | 逐人候选 |\n|---|---:|---:|\n'
    for mode,label in [('rgb','RGB'),('fused','融合')]:
        table+=f'| {label}宏F1 | {a[mode]["macro_f1"]:.4f} | {b[mode]["macro_f1"]:.4f} |\n'
        table+=f'| {label} other误报 | {a[mode]["other_false_positives"]}/{a[mode]["other_count"]} | {b[mode]["other_false_positives"]}/{b[mode]["other_count"]} |\n'
    val=dataset['summary']['val'];train=dataset['summary']['train'];test=dataset['summary']['test']
    text=f'''# 逐人 VideoMamba 适配（2026-10-23）

已完成现场数据处理、逐人标注核对、8轮微调、固定测试集配对评估与10段新旧对照演示。
按验证集选择第{selection['epoch']}轮；阈值0.75、alpha0.7、原图照度门75保持固定。
正式结果来自同一批逐人裁剪输入；旧整帧训练pn2也使用这些逐人输入作基线。

{table}

本表为{test['windows']}个人物窗口、{test['videos']}段视频的本次微调留出测试。
宏F1只平均有测试真值的wave、chase、drink、pick_up、other五类。
旧pn2已见过大部分历史现场数据，不能将本表称为全新场景泛化；原反馈留出会话单列在完整报告。

本轮收益主要是other误报减少和追逐/跑动识别改善。喝水召回从11/11降到8/11，错误集中在喝水34；
挥手RGB保持11/11，但固定骨架融合仍只有7/11。融合的来源组增益区间包含0，需继续用独立数据验证。

## 查看结果

- [10段逐人结果浏览页](demo-index.html)
- [完整评估报告](analysis/analysis-report.md)、[统计附录](analysis/stats-appendix.md)
- [逐窗新旧预测CSV](analysis/paired-test-windows.csv)、[所有数值JSON](analysis/results.json)
- [图表目录](analysis/figure-catalog.md)，PNG/PDF/SVG在`analysis/figures/`
- [独立候选权重]({target.relative_to(OUT).as_posix()})、[模型说明](models/model-card.json)
- [人物与时间段标注](annotations-review.html)、[剩余不确定轨迹](annotation-uncertainties.json)

## 对照视频

左右分别为微调前、微调后；每个人的框下显示自己的融合动作，没有全局动作标签条。
同一视频的新旧轨迹、RGB像素哈希和骨架概率已全部对齐核对，所有输出已完整解码并核对帧数。

'''
    for row in demos:text+=f'- [{row["name"]}](<{row["comparison"]}>)\n'
    text+=f'''
## 数据和训练

742个现场/线上源文件，字节去重为535段，49,160帧、约31分钟；不包括公开HMDB/Kinetics/ARID。
完整处理得到4,486个人物候选窗口；310个模糊或过渡窗口排除。
最终训练{train['windows']}窗、验证{val['windows']}窗、测试{test['windows']}窗。
120段视频没有足够稳定的人物窗口，其中44段原正样本；这些失败没有被记成正确other。

用户确认的三条other轨迹已加入，增加6个训练负窗与4个验证负窗。此前刚开始的第1轮已中止归档，
正式训练从同一pn2重新开始；划分、测试集、阈值和超参数未改变。证据见`label-amendment.json`。
训练协议在`protocol.json`，复现命令和输入约定见[RUNBOOK](RUNBOOK.md)。
`selection.json`保留选定模型时的快照（当时尚未测试）；测试完成状态另存`test-completion.json`，不反向改写选择记录。

## 使用边界

这是逐人方案的独立候选，必须使用跟踪后的逐人ROI输入，不能直接覆盖现有整帧生产权重。
生产算子、NVIDIA Box包、默认模型和镜像均未更新。
RTMO/ST-GCN未训练；亮场没有合格骨架时，当前融合规则仍会强制显示其他。
交谈、打架、握手无本轮正样本，投掷/跌倒只有单源训练样本，不能声称已验证全部十类能力。
'''
    (OUT/'README.md').write_text(text)
    artifacts=[target,models/'model-card.json',OUT/'README.md',OUT/'selection.json',OUT/'protocol.json',
               OUT/'training-history.json',OUT/'analysis/results.json',OUT/'demo-comparison.json',
               OUT/'test-completion.json',OUT/'model-environment-provenance.json',OUT/'source-code-hashes.json',
               OUT/'RUNBOOK.md',OUT/'demo-index.html',OUT/'annotations-review.html',
               OUT/'annotations.json',OUT/'splits.json',OUT/'training-data.json',OUT/'label-amendment.json']
    artifacts+=list((OUT/'analysis').glob('*.md'))+list((OUT/'analysis').glob('*.csv'))
    artifacts+=list((OUT/'analysis/figures').glob('*'))+list((OUT/'comparisons').glob('*.mp4'))
    for folder in ['demo','demo-baseline']:
        artifacts+=list((OUT/folder/'videos').glob('*.mp4'))+list((OUT/folder/'events').glob('*.json'))
    (OUT/'artifact-manifest.json').write_text(json.dumps({str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=sha(p))for p in artifacts},ensure_ascii=False,indent=2)+'\n')
    print(table,flush=True)
    print('Selected checkpoint:',target,'SHA256:',sha(target),flush=True)


if __name__=='__main__':main()
