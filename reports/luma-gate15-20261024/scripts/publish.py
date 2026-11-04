"""Publish exact paired results, uncertainty and real comparison clips."""
import html
import json
import os
from pathlib import Path
import zipfile
import numpy as np
import cv2

os.environ.setdefault('MPLCONFIGDIR','/tmp/luma-gate15-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate import OUT, save, sha


def main() -> None:
    results=json.loads((OUT/'results.json').read_text());assert results['complete']
    protocol=json.loads((OUT/'protocol.json').read_text())
    pairs=json.loads((OUT/'paired-windows.json').read_text())
    allrows=json.loads((OUT/'all-events.json').read_text())
    sub=results['subsets'];figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    table=['| 数据 | 配对已标注窗 | 正动作命中：75 → 15 | other误报：75 → 15 | 全类正确：75 → 15 |',
           '|---|---:|---:|---:|---:|']
    for split,title in [('val','验证集'),('test','测试集'),('train','训练内诊断')]:
        for b,bt in [('all','全部亮度'),('15-75','15≤L<75')]:
            r=sub[split+'/'+b];a=r['gate75'];z=r['gate15']
            table.append(f'| {title} · {bt} | {a["n"]} | {a["positive_correct"]} → {z["positive_correct"]} / {a["positive_count"]} | '
                f'{a["other_false_positives"]} → {z["other_false_positives"]} / {a["other_count"]} | {a["correct"]} → {z["correct"]} / {a["n"]} |')
    numeric='\n'.join(table)
    changed=[r for r in pairs if r['before']!=r['after']]
    harmful=[r for r in changed if r['truth']is not None and r['before']==r['truth']and r['after']!=r['truth']]
    helpful=[r for r in changed if r['truth']is not None and r['before']!=r['truth']and r['after']==r['truth']]
    lost_to_missing=[r for r in harmful if r['mode15']=='rgb_soft_no_pose']
    examples=['| 来源 | 帧/原图L | 真值 | 75结果 | 15结果 | 15分支 | 骨架真类概率 |',
              '|---|---|---|---|---|---|---:|']
    example_rows=[]
    for changes in (harmful,helpful):
        seen=set()
        for r in changes:
            if r['source']not in seen:
                seen.add(r['source']);example_rows.append(r)
            if len(seen)>=8:break
    shown=set()
    for r in example_rows:
        if r['source']in shown:continue
        shown.add(r['source'])
        p=r['skeleton_probs15'];labels=['wave','throw','chase','fall','fight','talk','drink','pick_up','shake_hands','other']
        sk='无'if p is None else f'{100*p[labels.index(r["truth"])]:.1f}%'
        examples.append(f'| {r["source"]} | {r["frame"]} / {r["raw_luma"]:.2f} | {r["truth"]} | '
            f'{r["before"]} {100*r["confidence75"]:.1f}% | {r["after"]} | {r["mode15"]} | {sk} |')
        if len(shown)>=16:break
    stats=['# 配对统计口径','',
        '一次固定权重推理，未做多训练种子重复。窗口重叠，同一视频/录制家族不是独立样本，不能用窗口总数冒充独立试验次数。',
        '按原划分source group计算每组正确率，再做等权配对差值；5000次来源组bootstrap，seed2415，百分位95%CI。',
        '大量并列且差值离散，不使用正态t检验；双侧精确符号检验只用有变化的来源组。验证/测试两个全量对照做Holm校正；分亮度与训练内结果仅描述。',
        '这些区间不覆盖标注误差、摄像机域迁移或多训练种子的方差。零变化区间不能证明所有新场景等效。','',
        '| 数据 | 来源组n | 75均值±SD | 15均值±SD | 差值pp [95%CI] | 胜/负/平 | p / Holm p |','|---|---:|---:|---:|---|---|---|']
    for split,title in [('val','验证'),('test','测试')]:
        s=sub[split+'/all']['stats'];lo,hi=s['ci95']
        stats.append(f'| {title} | {s["groups"]} | {s["before_mean"]:.4f} ± {s["before_sd"]:.4f} | '
            f'{s["after_mean"]:.4f} ± {s["after_sd"]:.4f} | {100*s["difference"]:.2f} [{100*lo:.2f}, {100*hi:.2f}] | '
            f'{s["wins"]}/{s["losses"]}/{s["ties"]} | {s["sign_p"]:.4g} / {s["holm_p"]:.4g} |')
    (OUT/'stats-appendix.md').write_text('\n'.join(stats)+'\n')
    coverage=['| 划分 | 来源数 | 窗口75/15 | 未标注75/15 | 无窗口来源75/15 |','|---|---:|---:|---:|---:|']
    for split in ['val','test','train']:
        sets=[[r for r in results['coverage']if r['split']==split and r['gate']==g]for g in (75,15)]
        totals=[(sum(r['windows']for r in rs),sum(r['unknown']for r in rs),sum(r['windows']==0 for r in rs))for rs in sets]
        coverage.append(f'| {split} | {len(sets[0])} | {totals[0][0]}/{totals[1][0]} | {totals[0][1]}/{totals[1][1]} | {totals[0][2]}/{totals[1][2]} |')
    body=['# 原图照度门75与15配对验证','',
        '**门限15可以启用更暗场景的骨架，但不能视为无损下调：实测同时出现误报减少与正确动作被压为其他。当前证据不支持仅改门限就直接替换生产方案。**','',
        f'完成{len(protocol["videos"])}段、{sum(v["frames"]for v in protocol["videos"]):,}原视频帧回放；每段独立运行两套跟踪/动作状态。',
        '固定清理展品后的YOLO26m、暗场微调RTMO-s、逐人ROI VideoMamba epoch07、原ST-GCN、Retinexformer；α=0.7，动作阈值0.75，无骨架平滑。',
        '此处检验用户指定的15，不扫描阈值、不训练、不改变当前生产默认75或交付包。','',numeric,'',
        f'配对窗口中共{len(changed)}个最终标签变化；已标注部分修正{len(helpful)}个、损失{len(harmful)}个，其中{len(lost_to_missing)}个损失来自无合格骨架的soft退化。',
        '受影响亮度区间的测试集16窗全部为other，没有正动作；其100%正确率只表示这16个负窗未误报，不能证明暗场正动作召回。验证集此区间仅11个有标签的正动作窗，两档均只命中4个。',
        '注意：能看见骨架、允许调用姿态、最终融合识别正确是不同指标。无合格骨架时，现有规则以other向量混合；非other最高只剩0.7，达不到0.75动作阈值。','',
        '## 具体变化示例','',*examples,'',
        '有合格骨架也可能退化：固定ST-GCN对真动作分数不够时，会拉低RGB正确类别。例如RGB=0.96，骨架真类概率必须至少0.26，才能使0.7×RGB+0.3×骨架达到0.75。姿态模型微调并未同步适配骨架动作分类器。','',
        '[喝水12实际漏报帧](evidence/drink12-f50.jpg)：原图L≈23.15，RGB喝水0.96066、ST-GCN喝水0.04477，融合0.68589低于0.75。两档使用同一人体裁剪，右侧仍显示骨架。','',
        '## 覆盖与比较边界','',*coverage,'',
        '主要表格只比较同源、完全相同32个取帧索引和RGB裁剪字节的窗口；两档RGB概率逐位相同。跟踪因姿态改变造成的未配对窗口在results.json另列，不混入主配对表。',
        '旧逐人动作标注通过人物框IoU与时间段迁移，至少80%采样帧关联一致；新增正样本人物不能按视频名自动赋标签，未确认窗口保持unknown。额外人工视觉核对如有，保留在label-review.json。',
        '视频级other沿用已有数据标注，包含可能的误检候选；无检测或无完整窗口不能当作正确other。',
        '验证/测试沿用冻结划分；历史初始化模型见过许多现场来源，不能称全新场景泛化。训练内暗场只用于找退化原因。',
        '本次是全帧离线动作/门控验证；共享前端和RGB计算减少重复工作，不能将运行耗时当作双流生产FPS或实时平台验收。',
        '门限使用增强前原图的整帧均值，原始黑边参与；回传片段缺少原图L，门限保持跳过。窗口均值与逐帧是否提取姿态分开判断。','',
        f'{len(results["unchanged_domain_sources"])}个全程L<15、L≥75或原L未知来源，两档输入窗口与融合概率完全一致。',
        '已处理所有列入协议的来源，未删除漏检或识别失败的视频。','',
        '## 证据','',
        '- [可播放对照页](index.html) · [视频预览ZIP](gate15-review.zip)',
        '- [逐窗CSV](paired-windows.csv) · [完整统计](results.json) · [统计附录](stats-appendix.md)',
        '- [图表与解释](figure-catalog.md) · [冻结协议及模型SHA256](protocol.json)',
        '- data目录保留每帧框/姿态、两档事件和完整概率。scripts目录提供回放、分析、发布与验证。']
    (OUT/'analysis-report.md').write_text('\n'.join(body)+'\n')
    (OUT/'README.md').write_text('# 门限15验证\n\n[完整报告](analysis-report.md) · [对照视频](index.html) · [完整预览包](gate15-review.zip)\n\n'+numeric+'\n\n固定最新候选模型，仅比较75与15。训练内与留出单列；生产默认和交付包未替换。\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#7b8795','#167d73']
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,subset,title in [(axes[0],'all','All paired labeled windows'),(axes[1],'15-75','Affected band: 15 <= L < 75')]:
        for j,g in enumerate(['gate75','gate15']):
            vals=[];labels=[]
            for split in ['val','test','train']:
                r=sub[split+'/'+subset][g];vals.append(100*r['top1']if r['n']else np.nan)
                name=split+' (other only)'if split=='test'and subset=='15-75'else split
                labels.append(f'{name}\nn={r["n"]}')
            bars=ax.bar(np.arange(3)+(j-.5)*.32,vals,.32,label=g,color=colors[j]);ax.bar_label(bars,fmt='%.1f',padding=2)
        ax.set(xticks=np.arange(3),xticklabels=labels,ylim=(0,110),ylabel='Correct (%)',title=title)
        ax.legend()
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(figdir/f'01-paired-accuracy.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    bands=['0-15','15-30','30-45','45-60','60-75','75+']
    affected=[r for r in allrows if r['gate']==15 and r['raw_luma']is not None]
    bottom=np.zeros(len(bands))
    for mode,color in [('rgb_low_luma','#768493'),('rgb_skeleton','#167d73'),('rgb_soft_no_pose','#d59154')]:
        n=np.array([sum(r['brightness']==b and r['mode']==mode for r in affected)for b in bands])
        axes[0].bar(bands,n,bottom=bottom,label=mode,color=color);bottom+=n
    axes[0].set(title='Gate15 branch coverage (all replay splits)',ylabel='Person windows');axes[0].legend(fontsize=8)
    states=['Corrected','Lost: valid pose','Lost: no pose']
    for j,split in enumerate(['val','test','train']):
        vals=[sum(r['split']==split for r in helpful),sum(r['split']==split and r['mode15']!='rgb_soft_no_pose'for r in harmful),
              sum(r['split']==split for r in lost_to_missing)]
        bars=axes[1].bar(np.arange(3)+(j-1)*.25,vals,.25,label=split);axes[1].bar_label(bars,padding=2)
    axes[1].set(xticks=np.arange(3),xticklabels=states,ylabel='Labeled paired windows',title='Decision changes: gate75 to gate15');axes[1].legend()
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(figdir/f'02-coverage-and-changes.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    (OUT/'figure-catalog.md').write_text('''# 图表说明

- `01-paired-accuracy.png/pdf`：同窗正确率，固定两档同一真值与RGB字节；左为全部亮度，右为真正受影响的15～75。标明每个划分样本数；无误差线，统计CI见附录。训练内柱不能作泛化结论，测试受影响窗可能全为other。
- `02-coverage-and-changes.png/pdf`：左图为门限15各亮度档的实际分支次数，含未标注窗口与训练回放，不是骨架精度；右图区分被修正与损失动作，说明开骨架是否改变正确判断，以及无骨架soft是否成为瓶颈。统计来自all-events.json和paired-windows.json。

图表用于判断能否仅改门限；不能用更多骨架出现或更高训练内正确率证明线上完整动作识别已通过。
''')
    cards=[];video_paths=[]
    extra_ids={v['id']for v in json.loads((OUT/'extra-demos.json').read_text())['videos']}if (OUT/'extra-demos.json').exists()else set()
    for v in protocol['videos']:
        if not v['demo']and v['id']not in extra_ids:continue
        vp=f'videos/{v["id"]}.mp4';poster=f'posters/{v["id"]}.jpg';video_paths.extend([vp,poster])
        rows=[r for r in pairs if r['video_id']==v['id']];n=sum(r['before']!=r['after']for r in rows)
        changes=[r for r in harmful+helpful if r['video_id']==v['id']]
        if changes:
            cap=cv2.VideoCapture(str(OUT/vp));cap.set(cv2.CAP_PROP_POS_FRAMES,changes[0]['frame'])
            ok,image=cap.read();cap.release();assert ok
            cv2.imwrite(str(OUT/poster),image)
        cards.append(f'<article><h2>{html.escape(Path(v["source"]).name)} <small>{v["split"]}</small></h2>'
            f'<p>原图平均 L={v["raw_mean_luma"]:.2f} · 配对窗口 {len(rows)} · 标签变化 {n}</p>'
            f'<video controls preload="none" poster="{poster}" src="{vp}"></video><p><a href="{vp}" download>保存视频</a></p></article>')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>照度门15验证</title><style>body{margin:0;background:#efede5;color:#20332e;font:16px/1.65 system-ui,sans-serif}main{max-width:1320px;margin:auto;padding:40px 24px}h1{font-size:40px;margin:10px 0}h2{font-size:22px}small{font-size:14px;font-weight:400}a{color:#166c5e}article{padding:26px 0;border-top:1px solid #bac9bf}video{width:100%;background:#111}header{margin-bottom:35px}.note{padding:18px;background:#dce4da}</style><main><header>
<h1>照度门 75 → 15</h1><p>左：门限75；右：门限15。同一组最新模型，每个人单独识别，无骨架平滑。</p>
<p class="note">门限15只是允许骨架参与。仍需满足姿态质量与动作阈值0.75；无合格骨架可能被融合规则压为其他。前约1秒显示识别中。train视频属于训练内诊断。封面优先显示已标注的判断变化帧，完整视频保留全部过程。</p>
<a href="analysis-report.md">完整结果</a> · <a href="paired-windows.csv">逐窗数据</a></header>'''+''.join(cards)+'''</main><script>document.querySelectorAll('video').forEach(v=>v.addEventListener('play',()=>document.querySelectorAll('video').forEach(w=>{if(w!==v)w.pause()})))</script></html>'''
    (OUT/'index.html').write_text(page)
    members=['index.html','README.md','analysis-report.md','stats-appendix.md','figure-catalog.md','results.json','paired-windows.csv','evidence/drink12-f50.jpg']+video_paths
    members += [str(p.relative_to(OUT))for p in figdir.glob('*')]
    zip_path=OUT/'gate15-review.zip'
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=3)as z:
        for name in members:z.write(OUT/name,name)
    save(OUT/'bundle.json',dict(path=zip_path.name,sha256=sha(zip_path),bytes=zip_path.stat().st_size,members=members))
    print(numeric,flush=True)


if __name__=='__main__':main()
