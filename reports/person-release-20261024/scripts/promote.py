"""Apply the validation-locked models/defaults and write authoritative delivery docs."""
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]
OP=ROOT/'platform/dark-demo-v2'


def main():
    lock=json.loads((OUT/'selection-lock.json').read_text());c=lock['config']
    assert (OUT/'results.json').is_file()
    shutil.copy2(OUT/'models/stgcn_selected_20261024.pth',OP/'ckpts/stgcn_person_20261024.pth')
    defaults={'lum_gate':30.0,'reco_min_conf':c['threshold'],'skel_fusion_alpha':c['alpha'],'skel_no_person':c['no_pose']}
    for rel in ['main.py','darkpipe/config.py']:
        p=OP/rel;s=p.read_text()
        for key,value in defaults.items():
            if rel=='main.py':
                pattern=r'(p\.add_argument\("--'+key+r'",(?: type=[^,]+,)? default=)[^,\n]+'
            else:
                pattern=r'(    '+key+r': (?:float|str) = )[^\n]+'
            s,n=re.subn(pattern,lambda m:m[1]+repr(value),s,count=1)
            assert n==1,(rel,key)
        p.write_text(s)
    p=OP/'darkpipe/person/core.py';s=p.read_text()
    s=re.sub(r'DEFAULT_ACTION_THRESHOLD = [\d.]+',f'DEFAULT_ACTION_THRESHOLD = {c["threshold"]}',s)
    s=s.replace('alpha=.7, threshold=',f'alpha={c["alpha"]}, threshold=');p.write_text(s)
    p=OP/'suanzi.json';doc=json.loads(p.read_text())
    for item in doc['inputs']:
        if item['name'] in defaults:item['default']=defaults[item['name']]
    doc['description']='逐人低照度增强与行为识别：现场人体/姿态/逐人RGB及骨架模型，亮度门30，同人融合，按人物和动作并发保存。'
    p.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
    models={
      'enhancer':dict(file='NTIRE.pth',architecture='Retinexformer',onsite_finetuned=False,
        training_data='Public NTIRE low-light pretrained checkpoint; previous onsite enhancement fine-tuning rejected after action degradation'),
      'detector':dict(file='yolo26m_person_20261024.onnx',architecture='YOLO26m',onsite_finetuned=True,
        training_data='1626 onsite dark frames/1741 person boxes plus72 bright frames/89 real-person boxes after excluding human-shaped exhibits; raw/enhanced views',
        source='reports/detector-clean-finetune-20261024/models/yolo26m_clean_candidate.onnx'),
      'pose':dict(file='rtmo-s_person_20261024.onnx',architecture='RTMO-s COCO17',onsite_finetuned=True,
        training_data='Onsite dark auxiliary person/COCO17 sequence labels, quality-filtered for training; no inference smoothing; not independent human GT',
        source='reports/dark-finetune-20261024/models/rtmo-s_dark_candidate.onnx'),
      'rgb':dict(file='videomamba_t_behavior_32f_person_roi_20261023.pth',architecture='VideoMamba-Tiny32f personROI',onsite_finetuned=True,
        training_data='All onsite video plus two online other batches;535 unique sources, reviewed person/time labels;3789 train windows in the10-23 fixed input dataset',
        source='reports/person-finetune-20261023/models/videomamba_t_behavior_32f_person_roi_20261023.pth'),
      'skeleton':dict(file='stgcn_person_20261024.pth',architecture='ST-GCN COCO17',onsite_finetuned=True,
        training_data='Historical onsite real-actiondata-all skeleton cache; old fitting line used video labels and merged person logits; applied per person at inference',
        source='platform/dark-demo-v2/ckpts/stgcn_behavior_32f_realactall_full.pth',
        selection='New2507-window person adaptation trained20epochs/val-selected13; full validation fusion comparison preferred the historical onsite model; retained honestly')}
    if c['model']=='adapted':
        models['skeleton'].update(training_data='Frozen onsite person/time labels;2507 eligible training windows from latest person/pose models',
            source='reports/person-release-20261024/models/stgcn_person_20261024.pth',selection='Validation selected adapted checkpoint')
    for value in models.values():
        path=OP/'ckpts'/value['file'];value['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    record=dict(version='20261024-person-gate30',defaults=defaults,person_mode=True,
        models=models,selection_lock_sha256=hashlib.sha256((OUT/'selection-lock.json').read_bytes()).hexdigest(),
        smoothing=False,object_prior=False)
    (OP/'models.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    p=ROOT/'scripts/package_nvidia_box.py';s=p.read_text()
    s=re.sub(r"'reco_min_conf': [\d.]+",f"'reco_min_conf': {c['threshold']}",s)
    s=re.sub(r"'skel_fusion_alpha': [\d.]+",f"'skel_fusion_alpha': {c['alpha']}",s)
    s=s.replace("'lum_gate': 30, 'fuse_mode': 'alpha',", "'lum_gate': 30, 'fuse_mode': 'alpha', 'person_mode': True, 'clip_denoise': 'off',")
    p.write_text(s)
    modeltable='| 模型 | 训练状态／数据 |\n|---|---|\n'
    for key,title,text in [
        ('enhancer','Retinexformer','公开 NTIRE 预训练，未采用已退化的现场增强微调'),
        ('detector','YOLO26m','现场暗场人体框＋剔除展品的亮场真人框微调'),
        ('pose','RTMO-s','现场暗场辅助人体／17点骨架标注微调；推理不平滑'),
        ('rgb','VideoMamba-Tiny','现场逐人ROI＋两批线上other，逐人时间段标签微调'),
        ('skeleton','ST-GCN','现场历史骨架微调；本轮新适配未胜出，保留旧现场权重' if c['model']=='old' else '本轮现场逐人骨架适配')]:
        modeltable+=f'| {title} | {text}；`{models[key]["file"]}` |\n'
    readme=f'''# 暗光逐人动作识别算子 · 20261024

本包默认使用逐人方案。视频输入→原图亮度→增强→人体框与跟踪ID→每个人自己的RGB ROI和骨架分类→同人融合→框下显示动作→实时输出与人物动作片段。没有全画面动作底栏，不使用物体先验，不做骨架平滑。

## 默认参数

| 参数 | 当前值 |
|---|---|
| person_mode | true；false仅用于旧整帧流程回滚 |
| proc_max_side | 720；1080p缩为720×404，默认bicubic×2输出1440×808 |
| lum_gate | 30；提亮前原图0–255平均亮度，不是lux |
| reco_min_conf | {c['threshold']:g}；非other最高概率低于门限只显示其他 |
| skel_fusion_alpha | {c['alpha']:g}，该人RGB权重；骨架权重{1-c['alpha']:g} |
| skel_no_person | {c['no_pose']}；无合格骨架按其权重给other票 |
| reco_span_sec | 1秒，32帧，人物窗每0.5秒提交一次 |
| gpu_ids | 0,1,2；实时前两卡交替前端，其余独立动作进程；双卡一前端一动作 |
| clip_pre_sec / clip_post_sec | 2 / 2秒 |
| clip_max_sec / clip_denoise | 15秒重叠分段 / off |
| pose_overlay / label_bar | true / false；逐人模式不显示全局底栏 |
| input_processed | false；仅历史已增强带旧底栏切片设true，自动裁条、不再增强，原L记未知 |

逐帧原L>=30才提取姿态；32帧窗口平均L>=30才允许双分支融合，低于30只使用该人的RGB。无骨架时的soft策略会降低非other分数。每人框旁只显示自己的融合动作；结果未满窗或超过1秒显示识别中。跟踪ID不是持久身份，遮挡和交叉可能换ID。

## 模型与实验

{modeltable}

完整文件哈希、模型是否现场训练及来源见 `models.json`。阈值和权重仅用冻结验证集选定，锁定后测试；本批测试214个人物窗，宏F1 {json.loads((OUT/'results.json').read_text())['scores']['test']['selected']['macro_f1']:.4f}，other误报2/163，正动作正确37/51。仍有误报与漏识，验证覆盖不齐全；历史初始化见过部分旧视频，不能作为全新场景泛化证明。完整实验、分亮度结果、演示和实时验收见仓库 `reports/person-release-20261024/`。

## 平台运行与输出

更新算子代码包并重新导入 `suanzi.json`，旧平台登记参数不会随代码默认值自动变化。检查启动日志 `person_mode=True lum_gate=30 alpha={c['alpha']:g} reco_min_conf={c['threshold']:g}`。容器卡号相对于可见GPU；三卡是默认申请，单卡可运行但拥挤场景不保证实时达标。

文件、目录输入离线处理一次，产出整段H.264视频、schema_version=3逐人事件JSON及动作片段；流输入常驻提供 `/stream`、`/live.flv`、可选 `/hls/index.m3u8`、`/events`、`/health`、`/config`。主动推送填写 `rtmp_push_url`（RTMP或RTSP目标）。输出无音频。

```bash
/opt/conda/envs/darkpipe/bin/python -u main.py --video_path /data/input.mp4 \\
  --gpu_ids 0,1,2 --clip_dir /data/results --session_json /data/session.json --extra_hosts ''
```

逐人事件保留 `track_id`、`history_generation`、`capture_epoch`编码于generation高位、源帧索引、完整概率、实际阈值与分支。不能再把某一条事件当全画面结论。人体框、姿态及RGB来自同一人物轨迹。

## 保存与稳定性

按人物ID＋动作分别保存完整画面，文件名含E/T/G/R编号；同时动作允许时间重叠，不会切掉其他人的片段。直播不等待保存：保存侧缓冲1.2秒，按源时间对齐异步结果并重绘该帧标签。前后2秒相对识别触发／最后同类事件；不等同于已知真实动作起止的保证。长动作按15秒重叠续段。断流、开头缓存不足、模型漏识、强杀或过载可能导致上下文不足，同名JSON记录覆盖范围与 `context_complete`。

临时编码在本地/tmp，完成后移动至clip_dir（可挂NFS）。多人路径隔离，end收尾消息不丢弃；真实写盘错误明确失败，临时拥塞有掉帧／弃段计数。WebHDFS上传有独立队列，失败保留本地并计数；离线交付失败仍非零退出。实时输入打开/读取超时3/2秒并重连；输入失联显示重连提示；每种推流由FFmpeg独立恢复。有限本机回环测试不能保证平台网络或无限时长零故障。

## 打包、镜像与回滚

运行镜像仍为 `darkpipe-operator:0.3.0`，新代码不依赖SciPy。镜像是否需重传以本次交付验收记录为准。模型在算子包ckpts中，入口不从镜像旧ckpts替换它们。包中有12份权重（5份当前模型与旧版回滚权重，Retinexformer共享）。

```bash
python3 platform/package_v2.py --check
python3 platform/package_v2.py --output platform/dark-demo-v2_person_20261024.tar.gz
python3 platform/package_v2.py --output platform/dark-demo-v2_person_20261024.zip
python3 scripts/package_nvidia_box.py --sync --output NVIDIA-Box_20261024.zip
```

tar.gz含dark-demo-v2外层；zip根目录是main.py和suanzi.json；manifest和sha256与最终文件配套。旧交付归档保留。旧整帧说明移至 `README_legacy_20261022.md`，其中性能与默认值不适用于当前逐人方案。
'''
    (OP/'README.md').write_text(readme)
    box=ROOT/'NVIDIA-Box'
    old=(box/'README_legacy_20261022.md').read_text()
    installation=old[old.index('## 1. 安装环境'):old.index('## 2. 离线视频与目录')]
    (box/'README.md').write_text(f'''# NVIDIA Box 完整逐人方案 · 20261024

runtime代码和12份平台权重逐文件同步；默认单张可见GPU0，可显式使用多卡。最新流程是每人ROI VideoMamba与同人骨架融合，不再整帧RGB。默认原图亮度门30、RGB权重{c['alpha']:g}、动作阈值{c['threshold']:g}、不平滑；每人框下显示动作或其他，无全局底栏。模型训练状态及数据来源见models.json；增强模型仍是公开NTIRE预训练，其余主模型有现场微调来源。

{installation}
## 运行

```bash
bash run.sh --video_path /data/input.mp4 --output-dir /data/results
bash run.sh --video_path 'rtsp://camera/live' --gpu_ids 0,1,2 \\
  --output-dir /data/results --stream_formats mjpeg,flv,hls \\
  --rtmp_push_url 'rtmp://media/live/darkpipe'
```

文件/目录离线只处理一次，视频与逐人事件、片段均落本地；实时接口 /stream、/live.flv、/hls/index.m3u8、/events、/health、/config。默认没有HDFS上传或平台hosts配置。输入历史已增强带底栏切片时才设--input_processed true。

单卡共享全部模型；实时≥3卡前两张交替增强/检测/姿态，其余独立动作进程；双卡是一前端一动作。离线第一卡前端，其余动作，保持一条源时间轴。处理长边720，默认×2后1080p源输出1440×808，无音频。单卡拥挤性能不保证15fps；实际Box硬件需实测。

片段按人物ID＋动作独立保存完整画面，前后各2秒，15秒重叠分段；clip_denoise默认off。保存侧等待1.2秒对齐异步结果并重绘标签。断流或EOF可缺上下文，sidecar记录完整性；真实写盘错误仍明确失败。使用Ctrl+C/SIGTERM正常收尾。

check_env.py验证最新逐人模型链；check_models.py和behavior_infer.py为历史单模型工具，仅供回滚研究，不代表当前逐人入口。完整实验与验收见仓库reports/person-release-20261024；本次在开发机Linux x86_64/RTX3090核验，未连接实体Box。

更新入口：python3 scripts/package_nvidia_box.py --sync --output NVIDIA-Box_20261024.zip。默认NVIDIA-Box.zip、manifest、sha256配套。历史说明保留于README_legacy_20261022.md，旧版本成绩不能用于当前算法。
''')
    print('promoted',defaults)


if __name__=='__main__':
    main()
