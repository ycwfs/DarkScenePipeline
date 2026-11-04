"""Decision-only ablation: identical tracks, ROIs, object evidence and model outputs.

This does not disable RTMO or replay the alpha=1 production frontend. It isolates
ST-GCN probability fusion from pose-dependent tracking/object crops.
"""
from collections import Counter
from common import *
from tune4 import Joint


def main():
    j=Joint();e=j.ev
    s=json.loads((OUT/'summary.json').read_text())['recommended']['setting'];gi=j.gates.index(s['gate'])
    fused=j.predict(s);rgb=j.predict(dict(s,alpha=1.))
    # Same damping budget, with no class-specific information from ST-GCN.
    p=s['alpha']*j.rgb;p[:,e.o]+=1-s['alpha'];p[j.luma<s['gate']]=j.rgb[j.luma<s['gate']]
    top=p.argmax(1);score=p[np.arange(len(p)),top];sup=np.full(len(p),np.inf)
    for ci,c in enumerate([e.drink,e.pickup]):sup[top==c]=j.support[gi,top==c,ci]
    damp=top.copy();damp[(score<s['action'])|(sup<s['object'])]=e.o
    # Prior candidate union ended at alpha=.99. Check every pure-RGB object
    # candidate in this diagnostic has actual evidence, not a default infinity.
    covered=set()
    for f in (OUT/'evidence').glob('*.json'):
        covered.update((r['window_id'],r['label']) for r in json.loads(f.read_text())['records'])
    count=0
    for r,probs in zip(j.rows,j.rgb):
        ci=int(probs.argmax())
        if BEHAVIORS[ci] in ('drink','pick_up') and probs[ci]>=.85:
            assert (r['id'],BEHAVIORS[ci]) in covered;count+=1
    def metrics(pred):
        old={split:e.metrics(pred,(e.scope=='onsite') & ((e.split==split) if split!='all' else True)) for split in ['all','train','val','test']}
        return dict(onsite=old,online=e.metrics(pred,e.scope=='online_supplement'),
                    new_known_drink_fp=int((pred[e.new_false]==e.drink).sum()),
                    new_true_drink_tp=int((pred[e.new_true]==e.drink).sum()),
                    new_unreviewed_drink=int((pred[e.new_unknown]==e.drink).sum()),
                    new_actions=dict(Counter(BEHAVIORS[i] for i in pred[e.new])))
    ts=[i/1000 for i in range(850,991)]
    grid=j.surfaces(1.,s['gate'],ts,[s['object']]);candidates=[]
    for i in np.flatnonzero(grid['feasible'][:,0]):
        candidates.append(j.calibration(dict(s,alpha=1.,action=ts[i])))
    chosen=min(candidates,key=j.key)
    setting={k:chosen[k] for k in ['alpha','gate','action','object']}
    pr=j.predict(setting)
    data=dict(type='Decision-only ablation on frozen tracks/ROIs/pose-derived object evidence',
        reference=s,retuned_rgb_setting=setting,
        results={k:metrics(v) for k,v in dict(fusion=fused,rgb_same_threshold=rgb,constant_other_damping=damp,rgb_retuned_action=pr).items()},
        all_windows_changed_vs_rgb=int((fused!=rgb).sum()),all_windows_changed_vs_damping=int((fused!=damp).sum()),
        covered_rgb_object_candidates=count,
        retuning='141 action thresholds .850-.990 step.001; gate35 and object.581 fixed; same old-val class guard and new binary feedback objective. Old test diagnostic only.',
        limitation='alpha=1 runtime also disables pose. This experiment retains pose-dependent tracks/crops and does not prove whole-pipeline removal quality or speed.',
        inputs_sha256=digest(OUT/'inputs.npz'))
    save(OUT/'fusion-necessity.json',data)
    lines=['# 骨架动作融合的实际贡献','',
        '固定同一批人物轨迹、RGB裁剪、姿态辅助物体ROI和物体检测证据，仅替换最终动作融合。其他参数门35、动作.925、物体.581。下表是人物窗口计数，不是独立动作事件数。','',
        '| 设置 | 旧全量真动作 /555 | 旧other误报 /1009 | 旧test真动作 /51 | 旧test other误报 /117 | 新已知喝水误报 /526 |',
        '|---|---:|---:|---:|---:|---:|']
    for key,title in [('fusion','RGB .972 + 骨架 .028'),('rgb_same_threshold','纯RGB，同动作阈值.925'),('constant_other_damping','RGB .972 + 固定other .028'),('rgb_retuned_action','纯RGB，动作阈值校准至.928')]:
        r=data['results'][key];a=r['onsite']['all'];t=r['onsite']['test']
        lines.append(f'|{title}|{a["positive_correct"]}|{a["other_fp"]}|{t["positive_correct"]}|{t["other_fp"]}|{r["new_known_drink_fp"]}|')
    lines+=['',
        '唯一新真喝水在四种设置中均为0/1，仍受弱物体证据限制。新挥手、跑动等没有逐窗真值，触发量变化不称误报纠正。旧全量含训练数据，历史test反复查看，新喝水反馈参与调参。','',
        '同阈值下融合相比纯RGB：旧other少6个误报、历史线上负样本少4个误报、新已知喝水少4个误报，同时少识别2个旧真实跑动窗口。全部51,834窗中有101窗输出变化，不能把这些变化全部当作纠错。','',
        '2.8%的融合量最多改变某类2.8个百分点；动作门槛.925附近足以翻转判断。例如RGB=.94、骨架=.10，融合为.91648，低于门槛；若骨架=.90，融合为.93888，仍可通过。','',
        '固定other阻尼对照用于区分“骨架有动作信息”和“整体降低RGB分数”：真实骨架融合保留485个旧真实动作，而恒定阻尼仅465个，同时误报略多。这说明骨架不只是一个固定置信度折扣。纯RGB允许重调动作阈值后，当前小范围比较仍存在误报/召回权衡。','',
        '工程裁决：当前骨架更适合作为临界判断校正，尚无证据证明它对每类都必要。不能仅由α=.972推出可无损删除。是否值得计算成本，需要完整纯RGB流水线与分别校准后的融合流水线对照，并实测速度。取消ST-GCN融合和删除RTMO姿态估计是不同改动：姿态还参与跟踪关联、局部物体裁剪和骨架显示。','',
        '正式入口alpha>=.999还会关闭姿态提取；本轮固定了前端，所以不是实际关闭整个骨架流后的结果。原参数、模型和交付文件未改。','']
    (OUT/'fusion-necessity.md').write_text('\n'.join(lines))
    print(json.dumps({k:dict(positive=v['onsite']['all']['positive_correct'],old_fp=v['onsite']['all']['other_fp'],test_tp=v['onsite']['test']['positive_correct'],new_fp=v['new_known_drink_fp']) for k,v in data['results'].items()},indent=2))


if __name__=='__main__':main()
