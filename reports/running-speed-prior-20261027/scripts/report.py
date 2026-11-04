"""Exact frozen comparison, independent candidate regression, and video checks."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import html
import json
import subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import OUT,ROOT,WAVE,save,digest
from choose_threshold import pass_rule


def decode(path):
    result=subprocess.run(['ffmpeg','-nostdin','-v','error','-xerror','-threads','2','-i',str(path),'-f','null','-'],capture_output=True,text=True)
    assert result.returncode==0 and not result.stderr,(path,result.stderr)
    result=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,nb_frames,duration','-of','json',str(path)],capture_output=True,text=True,check=True)
    return dict(path=str(path.relative_to(OUT)),sha256=digest(path),stream=json.loads(result.stdout)['streams'][0],decode_passed=True)


def main():
    selection=json.loads((OUT/'selection.json').read_text());tau=selection['threshold'];policy=selection['missing_policy']
    old=json.loads((OUT/'old-calibration-evidence.json').read_text())+json.loads((OUT/'old-test-evidence.json').read_text())
    new=json.loads((OUT/'long-calibration-evidence.json').read_text());summary={}
    for split in ['train','val','test']:
        rr=[r for r in old if r['split']==split]
        true=[r for r in rr if r['truth']=='chase'];tp=[r for r in true if r['baseline']=='chase'];fp=[r for r in rr if r['truth']is not None and r['truth']!='chase'and r['baseline']=='chase']
        summary[split]=dict(true_total=len(true),true_before=len(tp),true_after=sum(pass_rule(r['evidence'],tau,policy)for r in tp),
            true_bypass=sum(r['evidence']['speed']is None for r in tp),false_before=len(fp),false_after=sum(pass_rule(r['evidence'],tau,policy)for r in fp),
            unknown_before=sum(r['truth']is None and r['baseline']=='chase'for r in rr),
            unknown_after=sum(r['truth']is None and r['baseline']=='chase'and pass_rule(r['evidence'],tau,policy)for r in rr))
    # Complete 4234-window output replay: only original running labels can change.
    old_lookup={r['id']:r for r in old}
    all_old=json.loads((WAVE/'old-window-results.json').read_text());paired=[]
    for r in all_old:
        baseline=r['variants']['3_total'];label=baseline
        if label=='chase':
            assert r['id']in old_lookup
            if not pass_rule(old_lookup[r['id']]['evidence'],tau,policy):label='other'
        paired.append(dict(id=r['id'],truth=r['truth'],split=r['split'],baseline=baseline,candidate=label))
    save(OUT/'old-paired-results.json',paired)
    integrated=[];videos=[];independent=[]
    for directory in sorted((OUT/'long-videos').glob('*'))+sorted((OUT/'short-videos').glob('*')):
        if not directory.is_dir():continue
        d=json.loads((directory/'summary.json').read_text());assert d['stats']['frames']==d['source']['frames']
        rows=json.loads((directory/'events.json').read_text());assert len(rows)==d['events']
        clips=d['stats']['clips']
        for k in ['clips_failed','clips_abandoned','clips_forced','clips_unfinished','frames_dropped','write_errors']:
            assert clips[k]==0,(directory,k,clips[k])
        clip_files=list((directory/'clips').rglob('*.mp4'));assert len(clip_files)==clips['clips_saved']
        videos += [directory/'output.mp4']+clip_files
        record=dict(directory=str(directory.relative_to(OUT)),source=d['source'],frames=d['stats']['frames'],events=len(rows),
            actions=dict(Counter(r['label']for r in rows)),running_before=d['running_before'],running_after=d['running_after'],
            speed_vetoes=d['speed_vetoes'],clips=clips['clips_saved'],clip_context_incomplete=sum(not c.get('context_complete',False)for c in clips['clips']))
        if directory.parent.name=='long-videos':
            ref=[json.loads(s)for s in (WAVE/'long-videos'/directory.name/'candidate-events.jsonl').read_text().splitlines()]
            keys=['track_id','history_generation','frame_index','source_frames','bbox']
            match=len(ref)==len(rows)and all(all(a[k]==b[k]for k in keys)for a,b in zip(rows,ref))
            result=dict(source=directory.name,reference_events=len(ref),current_events=len(rows),same_geometry=match)
            if match:
                diffs={k:0. for k in ['rgb_probs','skeleton_probs','fused_probs']};changes=[]
                evidence={x['event']['frame_index']:x for x in json.loads((OUT/'long-running-evidence.json').read_text())}if directory.name=='new-01'else {}
                for a,b in zip(rows,ref):
                    for k in diffs:
                        if a[k]is not None and b[k]is not None:diffs[k]=max(diffs[k],float(np.max(np.abs(np.array(a[k])-b[k]))))
                    if a['label']!=b['label']:changes.append(dict(frame=a['frame_index'],track_id=a['track_id'],before=b['label'],after=a['label'],running_prior=a.get('running_prior')))
                result.update(max_probability_difference=diffs,label_changes=changes)
            independent.append(result)
        integrated.append(record)
    assert len(integrated)==13,len(integrated)
    with ThreadPoolExecutor(max_workers=3)as pool:decoded=list(pool.map(decode,videos))
    for v in decoded:
        p=OUT/v['path'];assert v['stream']['codec_name']=='h264'
        if p.name=='output.mp4':assert int(v['stream']['nb_frames'])==json.loads((p.parent/'summary.json').read_text())['source']['frames']
    save(OUT/'verification.json',dict(passed=True,integrated_sources=integrated,independent_comparison=independent,
        decoded_videos=decoded,video_count=len(decoded),rules_tests=15,old_replay_windows=len(paired),exact_frontend_windows=10,
        scope='Finite offline files; no new real-time, NFS, infinite-stream or throughput guarantee.'))
    save(OUT/'analysis-results.json',dict(threshold=tau,missing_policy=policy,old=summary,long=selection['calibration']['long'],integrated=integrated))
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(10,4.7));groups=[('Old train true run',[r for r in old if r['split']=='train'and r['truth']=='chase']),('Old val true run',[r for r in old if r['split']=='val'and r['truth']=='chase']),('Old test true run',[r for r in old if r['split']=='test'and r['truth']=='chase']),('Long child run',[r for r in new if r['truth']=='chase']),('Long non-run',[r for r in new if r['truth']=='other'])]
    for i,(name,rows)in enumerate(groups):
        values=[r['evidence']['speed']for r in rows if r['evidence']['speed']is not None]
        ax.scatter(values,np.full(len(values),i)+np.linspace(-.12,.12,len(values)),s=28,alpha=.7,label=None)
    ax.axvline(tau,color='red',linestyle='--',label=f'selected {tau}');ax.set_yticks(range(len(groups)),[g[0]for g in groups]);ax.set_xlabel('Mean keypoint speed / body-box height (1/s)');ax.legend();fig.tight_layout();fig.savefig(figdir/'speed-distributions.png',dpi=180);fig.savefig(figdir/'speed-distributions.pdf');plt.close(fig)
    sweep=[r for r in json.loads((OUT/'threshold-sweep.json').read_text())if r['missing_policy']=='bypass'and r['threshold']<=1.]
    fig,ax=plt.subplots(figsize=(9,4));xx=[r['threshold']for r in sweep]
    for field,name in [('true_kept','Long child TP (4)'),('false_kept','Long non-run FP (6)')]:ax.plot(xx,[r['long'][field]for r in sweep],marker='.',label=name)
    ax.plot(xx,[r['val']['true_kept']for r in sweep],label='Old val TP (8)');ax.axvline(tau,color='gray',linestyle='--');ax.set_xlabel('Speed threshold (body-box heights / second)');ax.set_ylabel('Retained windows');ax.legend();fig.tight_layout();fig.savefig(figdir/'threshold-tradeoff.png',dpi=180);fig.savefig(figdir/'threshold-tradeoff.pdf');plt.close(fig)
    table='\n'.join(f'|旧{split}|{s["true_before"]}/{s["true_total"]}|{s["true_after"]}/{s["true_total"]}|{s["false_before"]}→{s["false_after"]}|{s["true_bypass"]}|'for split,s in summary.items())
    fulls=[r for r in integrated if r['directory'].startswith('long')];shorts=[r for r in integrated if r['directory'].startswith('short')]
    report=f'''# 跑动骨架速度先验（2026-10-27）

推荐实验值：`--running_pose_min_speed {tau}`，单位为**人体框高度/秒**，0关闭。候选已接入实际逐人ActionEngine、CLI、配置和suanzi三处参数合约，保留上轮挥手3帧、正式7类物体模型和原四参数。代码在operator/dark-demo-v2；模型链接原ckpts，尚非可直接上传归档，正式交付包未替换。

当前1秒窗口内，去重源帧，选择相距约0.2秒（0.16–0.28秒）的同人关节观测，至少5个共同可靠关节（置信度≥.5）、至少3对且覆盖≥.3秒，平均关节欧氏位移除以平均人体框高度和真实时间间隔。按实际画幅恢复像素，避免x/y归一化尺度不同。坐标不平滑，不加模型前向，不等未来帧。它不是米/秒，仍受视角、遮挡、框质量和关节点跳动影响。

照度门35以下、原亮度未知或骨架关闭时跳过。亮场缺少可靠速度时也**保留原动作结果**：一个远处儿童真跑动没有可用骨架，严格否决会新增漏检。此策略是问询后按已说明推荐项继续的候选；完整严格对照仍在threshold-sweep.json。最初严格协议及后续修订分别保留，未隐去失败。

|数据|原正确跑动|加入先验后|已知跑动误报|正确但跳过速度检查|
|---|---:|---:|---:|---:|
{table}
|新长视频已审核触发|4/4|4/4|6→2|1|

旧126个已标注真跑动窗口，原113个正确全部保持；原13个漏识别不会被过滤规则恢复。旧两个已知跑动误报未消除，其中一个低亮跳过。新三段仅01号有10个原跑动触发，07/09均0。10个对应两组儿童跑动4窗、6个非跑动触发；原图前后时序人工辅助核对，采用用户“只有小孩在跑”反馈。6→2是这批反馈校准上的改善，不是独立测试准确率或整片召回。

0.30时新误报剩3；0.35剩2且儿童4保持；0.40开始漏1个儿童；0.50虽再过滤重叠成人一窗，却丢旧val2窗、儿童1窗。残留一例重叠人体框速度.453，另一例远处缺姿态。旧val也有误报速度.572，因此单个速度下限无法完全分离走/跑。选择只用旧val＋明确授权的长视频反馈，test在锁定后才评分。

固定证据主对照复用旧535源4234窗的模型概率，并对受影响窗还原同人同框姿态；新01完整9876帧重跑前端，对10个原触发逐个验证框及32个源帧完全相同。独立实际候选另跑完整01/07/09，共{sum(r['frames']for r in fulls)}帧，另5段新摘录＋5段旧短片，合计{len(integrated)}输入、{sum(r['events']for r in integrated)}人物窗。新短片重新启动跟踪、窗口相位不同，与完整视频同现实事件不重复当独立样本。

实际长片跑动触发：{[(r['directory'],r['running_before'],r['running_after'])for r in fulls]}；10短片为{sum(r['running_before']for r in shorts)}→{sum(r['running_after']for r in shorts)}。独立加载的概率/标签差详见verification.json，不将独立前向宣称逐位相等；严格固定概率效果以上表为准。

共{sum(r['clips']for r in integrated)}自动动作片段，{len(decoded)}输出/片段完整解码，写失败/强制释放/未完成/丢帧/弃段均0。上下文不足{sum(r['clip_context_incomplete']for r in integrated)}段明示在verification.json。15个速度规则回归通过。只代表有限离线验证，不声称实时性能或无限流稳定。

浏览[index.html](index.html)，含完整输出、保存片段及原图审核；[XCLIP独立实验](../xclip-person-20261027/index.html)。
'''
    (OUT/'analysis-report.md').write_text(report)
    (OUT/'README.md').write_text('# 跑动速度先验实验\n\n默认候选 `running_pose_min_speed=0.35` 人体框高度/秒，0关闭。缺骨架/低亮跳过。原概率固定对照：新长片已审核跑动误报6→2，儿童4/4保持；旧原正确跑动113/113保持。\n\n[结果视频](index.html) · [完整报告](analysis-report.md) · [统计与限制](stats-appendix.md) · [复现](REPRODUCE.md)。正式平台/Box包和镜像不变。\n')
    (OUT/'stats-appendix.md').write_text('# 统计口径\n\n这是确定性规则的单次配对诊断，无多seed方差。来源内时间窗重叠，儿童4窗实际上仅两个动作组，不能按独立4人/4视频计算置信区间。6个非跑动反馈例也是阈值校准数据，故只报告计数及减误报4/6（66.7%），不作p值检验、泛化置信区间或显著性声明。旧test锁定后评分但早期已经查看，仍是历史回归。旧unknown不算真/假。\n')
    (OUT/'figure-catalog.md').write_text('# 图表解释\n\n- speed-distributions：同一速度公式下旧真跑动及新儿童/成人的分布，缺姿态不伪填0；应注意走跑重叠，不能声称阈值完全分开。无误差条，各点为相关时间窗。\n- threshold-tradeoff：固定校准窗的阈值/误报/真跑动保留量，展示.35到.40即发生漏检，而不是无限提高阈值。没有将测试集反向选参。\n')
    cards=[]
    for r in integrated:
        path=r['directory'];title=Path(r['source'].get('source',r['source'].get('absolute_path',''))).name
        links=' '.join(f'<a href="{html.escape(v["path"])}">片段{j+1}</a>'for j,v in enumerate(decoded)if v['path'].startswith(path+'/clips/'))
        cards.append(f'<article><h3>{html.escape(path+" / "+title)}</h3><p>跑动窗 {r["running_before"]} → {r["running_after"]}；片段{r["clips"]}。</p><video controls preload="metadata" width="1000" src="{path}/output.mp4"></video><p>{links}</p></article>')
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>跑动速度先验</title><style>body{font-family:sans-serif;max-width:1100px;margin:30px auto;background:#f5f6f8}article{background:white;padding:16px;margin:18px 0}img,video{max-width:100%}</style><h1>跑动速度先验：0.35 人体框高度/秒</h1><p>新长片审核误报6→2，儿童4/4保持；缺骨架时保留原判断。仍有重叠人体和远处缺骨架误报，完整结果全部保留。</p><p><a href="analysis-report.md">报告</a> · <a href="verification.json">完整核验</a> · <a href="../xclip-person-20261027/index.html">XCLIP实验</a></p><img src="figures/speed-distributions.png"><img src="figures/threshold-tradeoff.png"><h2>原触发人物审核</h2><img src="review/new01-running-triggers.jpg">'+''.join(cards))
    print(json.dumps(dict(passed=True,sources=len(integrated),videos=len(decoded),old=summary),ensure_ascii=False))

if __name__=='__main__':main()
