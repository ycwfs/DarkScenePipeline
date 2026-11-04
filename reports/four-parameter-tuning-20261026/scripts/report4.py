"""Reviewable per-window results, source-cluster uncertainty, and real comparison figures."""
import csv
import html
from collections import Counter,defaultdict
from common import *
from tune4 import Joint,old


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    j=Joint();e=j.ev
    summary=json.loads((OUT/'summary.json').read_text());doc=json.loads((OUT/'predictions.json').read_text())
    predictions={k:np.asarray(v) for k,v in doc['predictions'].items()};base=predictions['baseline'];pick=predictions['recommended']
    setting=doc['settings']['recommended'];top,score,support=j.values(setting['alpha'],setting['gate'])
    rows=[]
    for i,r in enumerate(j.rows):
        row={k:r[k] for k in ['id','source_id','scope','split','group','track_id','timestamp','truth','is_drink','raw_luma']}
        row.update(baseline=BEHAVIORS[base[i]],recommended=BEHAVIORS[pick[i]],score=float(score[i]),
                   object_support=None if not np.isfinite(support[i]) else float(support[i]),changed=bool(base[i]!=pick[i]),
                   lost_true=bool(e.truth[i]>=0 and e.truth[i]!=j.o and base[i]==e.truth[i] and pick[i]!=e.truth[i]),
                   gained_true=bool(e.truth[i]>=0 and e.truth[i]!=j.o and base[i]!=e.truth[i] and pick[i]==e.truth[i]),
                   removed_other_fp=bool(e.truth[i]==j.o and base[i]!=j.o and pick[i]==j.o),
                   added_other_fp=bool(e.truth[i]==j.o and base[i]==j.o and pick[i]!=j.o))
        rows.append(row)
    old.csv_save(OUT/'window-results.csv',rows)
    lost=[r for r in rows if r['lost_true'] or r['gained_true'] or r['removed_other_fp'] or r['added_other_fp']]
    if lost:old.csv_save(OUT/'known-changes.csv',lost)
    per_video=[]
    for s in e.protocol['sources']:
        mask=np.array([r['source_id']==s['id'] for r in j.rows]);m0=e.metrics(base,mask);m1=e.metrics(pick,mask)
        per_video.append(dict(id=s['id'],source=s['source'],scope=s['scope'],split=s['split'],windows=int(mask.sum()),
            known=m0['known'],unknown=m0['unknown'],baseline_positive=m0['positive_correct'],recommended_positive=m1['positive_correct'],
            baseline_other_fp=m0['other_fp'],recommended_other_fp=m1['other_fp'],
            new_known_drink_fp_before=int((base[mask&e.new_false]==e.drink).sum()),new_known_drink_fp_after=int((pick[mask&e.new_false]==e.drink).sum())))
    old.csv_save(OUT/'video-results.csv',per_video)
    # Paired source-family bootstrap, descriptive only after adaptive calibration.
    rng=np.random.default_rng(26);bootstrap={}
    for name,mask,truth_type in [('old_other', (e.scope=='onsite')&(e.truth==j.o),'other'),
                                ('old_positive',(e.scope=='onsite')&(e.truth>=0)&(e.truth!=j.o),'positive'),
                                ('new_known_drink_false',e.new_false,'drink')]:
        groups=defaultdict(list)
        for i in np.flatnonzero(mask):groups[j.rows[i]['group']].append(i)
        data=[]
        for inds in groups.values():
            if truth_type=='other':a=base[inds]!=j.o;b=pick[inds]!=j.o
            elif truth_type=='positive':a=base[inds]==e.truth[inds];b=pick[inds]==e.truth[inds]
            else:a=base[inds]==e.drink;b=pick[inds]==e.drink
            data.append([len(inds),int(a.sum()),int(b.sum())])
        d=np.array(data);ix=rng.integers(0,len(d),size=(5000,len(d)));s=d[ix].sum(1)
        dif=100*(s[:,2]-s[:,1])/s[:,0]
        bootstrap[name]=dict(groups=len(d),windows=int(d[:,0].sum()),before=int(d[:,1].sum()),after=int(d[:,2].sum()),
             delta_percentage_points=float(100*(d[:,2].sum()-d[:,1].sum())/d[:,0].sum()),
             descriptive_cluster_95ci=np.quantile(dif,[.025,.975]).tolist(),resamples=5000)
    save(OUT/'descriptive-bootstrap.json',bootstrap)
    gains={k:int(sum(r[k] for r in rows if r['scope']=='onsite')) for k in ['lost_true','gained_true','removed_other_fp','added_other_fp']}
    save(OUT/'paired-changes.json',gains)
    names=['baseline','previous_two_parameter','recommended']
    display=['原配置','上轮两参数','本轮四参数']
    if 'fixed_action_option' in summary:names.append('fixed_action_option');display.append('动作阈值保留0.85')
    if 'true_drink_option' in summary:names.append('true_drink_option');display.append('保留新真喝水')
    fig,axs=plt.subplots(2,2,figsize=(12,8.5));axs=axs.ravel();x=np.arange(len(names));colors=['#718096','#4c78a8','#e67e22','#59a14f','#ad70b0'][:len(names)]
    values=[[summary[n]['onsite']['all']['positive_correct'] for n in names],
            [summary[n]['onsite']['test']['positive_correct'] for n in names],
            [summary[n]['onsite']['all']['other_fp'] for n in names],
            [summary[n]['new']['known_drink_fp'] for n in names]]
    for ax,ys,title in zip(axs,values,['旧全量真实动作正确 / 555（含训练内）','旧历史 test 真动作正确 / 51（越高越好）','旧全量 other 误报 / 1009（越低越好）','新已知喝水误报 / 526（越低越好）']):
        bars=ax.bar(x,ys,color=colors);ax.bar_label(bars,padding=3);ax.set_xticks(x,display,rotation=25,ha='right');ax.set_title(title,fontsize=11);ax.set_ylim(0,max(ys)*1.2+1);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('固定算法与模型：四参数联合校准的收益和代价\n唯一新真喝水：以上各方案均未保留（0/1）',fontsize=13);fig.tight_layout();(OUT/'figures').mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(OUT/'figures'/f'01-tradeoff.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    grid=list(csv.DictReader((OUT/'coarse-best-by-alpha-gate.csv').open(encoding='utf-8-sig')))
    alphas=json.loads((OUT/'protocol.json').read_text())['grid']['alpha'];gates=j.gates
    matrix=np.full((len(gates),len(alphas)),np.nan)
    for r in grid:matrix[gates.index(int(r['gate'])),alphas.index(float(r['alpha']))]=float(r['objective'])
    fig,ax=plt.subplots(figsize=(10,4.5));im=ax.imshow(matrix,aspect='auto',origin='lower',cmap='viridis_r')
    ax.set_yticks(range(len(gates)),gates);step=max(1,len(alphas)//12);ax.set_xticks(range(0,len(alphas),step),[f'{a:.2f}' for a in alphas[::step]])
    ax.set_xlabel('RGB 融合权重 α');ax.set_ylabel('原图照度门');ax.set_title('各 α／照度门下，动作与物体阈值最优的可行目标值')
    fig.colorbar(im,ax=ax,label='0.5 ×（旧 val other 误报率 + 新已知喝水误报率）');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(OUT/'figures'/f'02-joint-grid.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    # Named-class comparison makes losses visible, including unrelated actions.
    cls=[c for c in BEHAVIORS if c!='other' and summary['baseline']['onsite']['all']['classes'][c]['support']]
    chinese={'drink':'喝水','pick_up':'捡物','wave':'挥手','chase':'跑动','fall':'跌倒','throw':'投掷'}
    fig,ax=plt.subplots(figsize=(10,4));xs=np.arange(len(cls))
    for offset,name,title,color in [(-.2,'baseline','原配置','#718096'),(.2,'recommended','四参数候选','#e67e22')]:
        ys=[summary[name]['onsite']['all']['classes'][c]['tp'] for c in cls]
        ax.bar_label(ax.bar(xs+offset,ys,width=.38,label=title,color=color),padding=2)
    ax.set_xticks(xs,[f'{chinese.get(c,c)} / {summary["baseline"]["onsite"]["all"]["classes"][c]["support"]}' for c in cls]);ax.set_ylabel('真实动作正确窗口数');ax.set_title('旧现场逐类收益／损失（含训练内来源）');ax.legend();fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(OUT/'figures'/f'03-class-retention.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    lock=json.loads((OUT/'selection-lock.json').read_text());cov=json.loads((OUT/'coverage.json').read_text());verify=json.loads((OUT/'verification.json').read_text())
    lines=['# 四参数联合调优：固定算法与模型','',
        f'候选：RGB 权重 **{setting["alpha"]:.3f}**、原图照度门 **{setting["gate"]}**、动作阈值 **{setting["action"]:.3f}**、物体阈值 **{setting["object"]:.3f}**。离线实验候选，未替换生产包、模型、镜像或默认参数。','',
        '| 方案 | α / 门 / 动作 / 物体 | 旧真动作 /555 | 旧other误报 /1009 | 新已知喝水误报 /526 | 新真喝水 /1 |',
        '|---|---|---:|---:|---:|---:|']
    for n,title in zip(names,display):
        s=summary[n];v=s['setting'];lines.append(f'|{title}|{v["alpha"]:.3f} / {v["gate"]} / {v["action"]:.3f} / {v["object"]:.3f}|{s["onsite"]["all"]["positive_correct"]}|{s["onsite"]["all"]["other_fp"]}|{s["new"]["known_drink_fp"]}|{s["new"]["known_drink_tp"]}|')
    lines+=['',f'配对变化：旧真实动作损失 {gains["lost_true"]} 窗、新增正确 {gains["gained_true"]} 窗；旧other纠正 {gains["removed_other_fp"]} 个、新增误报 {gains["added_other_fp"]} 个。窗口不是完整动作事件数或保存片段数。旧全量净增不能掩盖历史test的损失，本候选只适合表述为“误报优先的校准候选”，不是全面胜出。','',
        '在本轮网格与旧val逐类保护等约束下，没有找到同时保留唯一新真喝水的可行组合。该窗兼容瓶/杯证据最高仅0.012022；降低物体阈值可以放过它，但不能在当前约束下同时守住其他指标。这不能推导为所有未来场景均无法兼顾。','',
        '| 旧现场拆分 | 真动作正确（原→候选） | other误报（原→候选） |','|---|---|---|']
    for split in ['train','val','test']:
        a=summary['baseline']['onsite'][split];b=summary['recommended']['onsite'][split]
        lines.append(f'|{split}|{a["positive_correct"]}→{b["positive_correct"]}/{a["positive_count"]}|{a["other_fp"]}→{b["other_fp"]}/{a["other_count"]}|')
    lines+=['','逐类旧现场正确数：'+ '；'.join(f'{chinese.get(c,c)} {summary["baseline"]["onsite"]["all"]["classes"][c]["tp"]}→{summary["recommended"]["onsite"]["all"]["classes"][c]["tp"]}/{summary["baseline"]["onsite"]["all"]["classes"][c]["support"]}' for c in cls)+'。','',
        '新场景其他类别没有逐窗真值，以下仅为触发量：'+ '；'.join(f'{chinese.get(c,c)} {summary["baseline"]["new"]["actions"].get(c,0)}→{summary["recommended"]["new"]["actions"].get(c,0)}' for c in ['wave','chase','pick_up'])+'。不能把减少量全部称为纠错。','',
        f'粗扫 {lock["coarse"]["count"]:,} 组，细扫 {lock["fine"]["count"]:,} 组（有少量重叠）；α粗步长.01，动作/物体粗步长.01；局部细步长.001。主搜索照度门 {j.gates}，每5单位。70/75因新现场存在低于门限的单帧，未放入联合排名；旧现场仍完成这两档的跟踪/骨架诊断。千分位最优只反映当前样本，不是跨场景精确最优。','',
        '门35–65在主候选的校准目标及逐类val指标上同分，因此按预定“更接近原配置”的规则选35；这不证明35是唯一最佳亮度分界，也不把test表现用于打破平局。','',
        '新增正确的22个旧真实动作窗全部位于30≤L<35。门提高后，这些窗转为纯RGB且跳过物体先验，因此获得恢复；这不代表模型本身识别能力提高，也不能抵消亮场test的退化。','',
        '目标：旧val other误报率与新526已知喝水误报率等权。约束从搜索前冻结：旧val每个有标签动作的TP不低于基线、旧val/历史线上val/新已知误报不增加、新未审核喝水不增加，动作阈值不直接压掉唯一新真喝水。选参函数不读取旧train/test指标。','',
        f'544来源、51,834窗。151个受门影响的旧来源按生产跟踪器独立回放；13档轨迹和裁剪全部对齐，908个受影响gate30窗口的RGB/骨架概率复现。新9段80,316原帧逐一计算亮度，与全部47,600窗口亮度均值精确对齐；主搜索各门的逐帧骨架掩码一致。新执行 {cov["stats"]["enhanced_frames"]:,} 帧旧源增强，以及 {cov["stats"]["object_fresh_crops"]} 次新增物体裁剪推理，物体分数保留至.001。其余同像素、同模型缓存按哈希复用。','',
        '算法保持：现场微调YOLO26m人体与物体模型、RTMO-s、VideoMamba、历史现场ST-GCN，以及公开NTIRE增强；720增强图同人裁剪，5帧中任一帧有兼容物体即可。原图窗口/采样照度门共同约束物体先验。没有改成连续证据规则、类别专用阈值或新模型。','',
        '证据关联沿用前轮：旧视频使用精确逐帧轨迹；新已知候选复用冻结ROI，新补候选取同人最近事件（≤0.55秒）并重新推理姿态，框IoU>0.4歧义时弃权。这是离线近似绑定，不等同于已集成生产逐帧物体先验。','',
        '限制：旧全量含训练数据，历史test已多次查看；新9段现在是用户反馈校准集，不能称独立泛化验收。新仅527个既有喝水触发有二元标签：1真526假；not_drink不等于所有动作other。捡物“基本全误报”没有逐窗标注。71个旧来源无动作窗（含漏检/未成窗），不计正确；525个旧现场未知窗不计准确率。历史线上128来源2145窗亮度未知，物体先验弃权，单列统计。','',
        f'校验：{verify["baseline_windows_reproduced"]:,}基线窗决策复现，{verify["brute_force_grid_checks"]}组累积搜索与逐窗暴力结果对照，{verify["deployment_fusion_policy_checks"]:,}次正式融合策略对照，{verify["object_policy_checks"]:,}次正式物体规则对照；9项物体先验CPU回归通过。不是实时速度/推流/保存验收。','',
        '文件：[逐窗结果](window-results.csv)、[已知得失](known-changes.csv)、[全部来源](video-results.csv)、[参数预设](recommended-settings.json)、[选参锁](selection-lock.json)、[验证](verification.json)、[统计附录](stats-appendix.md)。','',
        '![收益与代价](figures/01-tradeoff.png)','','![联合搜索](figures/02-joint-grid.png)','','![逐类结果](figures/03-class-retention.png)','']
    text='\n'.join(lines);(OUT/'README.md').write_text(text);(OUT/'analysis-report.md').write_text(text)
    appendix=['# 统计附录','', '这是单组固定模型的阈值反馈校准，没有多随机种子训练，不能给出种子均值±标准差。全部主表为原始计数和明确分母。', '',
        '相邻人物窗高度重叠，采用原视频家族分组的配对bootstrap（seed26，5000次）；每次整组重采样，以下为“候选−原配置”的百分比点差。仅描述当前样本不确定性；未校正参数选择偏差，不作显著性或因果提升声明。', '',
        '|比较|来源组|窗口|差值 pp|描述性95%区间|','|---|---:|---:|---:|---|']
    for k,r in bootstrap.items():appendix.append(f'|{k}|{r["groups"]}|{r["windows"]}|{r["delta_percentage_points"]:.3f}|[{r["descriptive_cluster_95ci"][0]:.3f}, {r["descriptive_cluster_95ci"][1]:.3f}]|')
    appendix+=['','旧正动作的负差表示损失，误报的负差表示改善。新真喝水只有1例，不给总体召回置信结论。多参数搜索不做大量未校正p值检验；未提供p值。样本规模与比例详见summary.json，所有544来源包括无窗来源均在video-results.csv。','']
    (OUT/'stats-appendix.md').write_text('\n'.join(appendix))
    (OUT/'figure-catalog.md').write_text('''# 图表说明

- 01-tradeoff：同一窗口集比较旧全量真动作、旧历史test真动作、旧other误报、新已知喝水误报。前两项越高越好，后两项越低越好。必须同时阅读新真喝水0/1或1/1以及每个方案的四参数，不能只看误报或含训练内的总TP。
- 02-joint-grid：每个RGB权重/照度门组合，在满足旧val逐类保护后最好的动作/物体阈值目标值。空白表示无可行方案，颜色越低越好。此为校准面，不代表独立测试面。
- 03-class-retention：旧全量已知真实动作逐类TP，展示喝水、跑动等损失是否被别类增益掩盖。全量含训练内来源，分拆结果见报告。

PNG供预览，PDF供矢量导出。数值来源均为summary.json/coarse-best-by-alpha-gate.csv；未生成示意或合成结果。
''')
    table='<table><tr><th>方案</th><th>α/门/动作/物体</th><th>旧真动作</th><th>旧other误报</th><th>新喝水误报</th><th>新真喝水</th></tr>'
    for n,title in zip(names,display):
        s=summary[n];v=s['setting'];table+=f'<tr><td>{title}</td><td>{v["alpha"]}/{v["gate"]}/{v["action"]}/{v["object"]}</td><td>{s["onsite"]["all"]["positive_correct"]}/555</td><td>{s["onsite"]["all"]["other_fp"]}/1009</td><td>{s["new"]["known_drink_fp"]}/526</td><td>{s["new"]["known_drink_tp"]}/1</td></tr>'
    table+='</table>'
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><title>四参数联合调优</title><style>body{font:16px system-ui;max-width:1250px;margin:30px auto;color:#24324a}table{border-collapse:collapse;width:100%}th,td{padding:12px;border-bottom:1px solid #ddd;text-align:left}img{width:100%}a{color:#165ca4}</style><h1>四参数联合调优</h1><p>固定算法与模型；旧现场＋新多人用户反馈校准。候选未上线。</p>'+table+'<p>同时比较误报收益和真实动作损失；新捡物、跑动、挥手没有逐窗真值。全量包含旧训练数据，不是独立泛化成绩。</p><p><a href="README.md">完整报告</a> · <a href="recommended-settings.json">参数</a> · <a href="window-results.csv">逐窗CSV</a> · <a href="known-changes.csv">已知得失</a> · <a href="stats-appendix.md">统计附录</a></p>'+''.join(f'<img src="figures/{f}" alt="结果图">' for f in ['01-tradeoff.png','02-joint-grid.png','03-class-retention.png'])+'</html>'
    (OUT/'index.html').write_text(page)
    manifest={str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.json','.csv','.npz','.md','.py','.png','.pdf','.html'] and p.name!='artifact-manifest.json' and '__pycache__' not in p.parts}
    save(OUT/'artifact-manifest.json',manifest)
    print(json.dumps(dict(settings=doc['settings'],paired_changes=gains),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
