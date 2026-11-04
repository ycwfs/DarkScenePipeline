"""Offline review entry with real media and explicit candidate limitations."""
from html import escape
import json

from build_dataset import OUT


NOTES = {
    'old-drink-supported':'旧测试真实喝水；连续检出瓶，两种规则均保留。',
    'old-drink-missed':'旧测试真实喝水；瓶只断续检出，严格规则误杀。',
    'old-pickup-missed':'旧测试真实捡物；有纸张框，但连续抬起证据不足。',
    'new05-phone':'本人手里的亮屏手机未被可靠检出；邻人的手机不能否决本人的动作。',
    'new03-exhibit':'实际是机器人展品，仅一帧约0.40，未达到持续否决条件。',
    'new09-supported':'关键反例：视觉上是相机拍摄，模型却持续预测为瓶，严格规则仍保留喝水。',
    'new03-near':'手靠近头部；零星“瓶”预测缺少可靠物体身份，保守保持原结果。',
    'new07-pickup':'人物站立、手在背后，未见连续捡起动作；原模型仍预测捡物。',
}


def main() -> None:
    demos=json.loads((OUT/'demo-manifest.json').read_text())
    cards=[]
    for r in demos:
        cards.append(f'<article><h3>{escape(r["title"])}</h3><p>{escape(NOTES[r["id"]])}</p>'
            f'<video controls preload="none" poster="{escape(r["sheet"])}" src="{escape(r["video"])}"></video>'
            f'<p><a href="{escape(r["video"])}">打开MP4</a> · <a href="{escape(r["sheet"])}">查看5帧证据大图</a>'
            f' · T{r["person_id"]} / {r["timestamp"]:.2f}s</p></article>')
    figures=''.join(f'<figure><img loading="lazy" src="figures/{name}.png" alt="{escape(caption)}"><figcaption>{escape(caption)}</figcaption></figure>'
        for name,caption in [('01-object-test','旧物体测试：裁剪检出部分瓶、纸、展品，手机仍0/3。原模型不支持的类别省略。'),
            ('02-action-retention','旧动作测试子集：严格规则明显增加漏检。'),
            ('03-selection','只用旧验证集选第17轮；单seed，不是20次独立重复。'),
            ('04-feedback','新9段的相关动作触发数；没有完整真值，不代表准确率。')])
    text='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>现场物体先验 · 微调与反馈验证</title><style>
body{margin:0;background:#f1f4f7;color:#182c3b;font:16px/1.75 system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:28px}
h1{font-size:32px;line-height:1.3}h2{margin-top:38px}a{color:#006e72}strong{color:#9f3525}.note{background:#fff3df;border-left:5px solid #be7532;padding:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:24px}article{background:white;border:1px solid #d5dee6;border-radius:8px;padding:18px}article h3{margin-top:0}
video,img{max-width:100%;height:auto}video{width:100%;background:#101926}figure{margin:24px 0}figcaption{color:#526575}table{border-collapse:collapse;width:100%;background:white}td,th{padding:8px 12px;border-bottom:1px solid #d5dee6;text-align:left}
</style></head><body><main><p>2026-10-26 · 单次训练 seed25 · 独立实验候选</p><h1>物体能检出一些，仍不足以可靠纠正动作误报</h1>
<p class="note"><strong>本候选未启用到正式平台。</strong>正式包已改为“跑动”、动作阈值0.85；原镜像不变。保守规则在新9段否决0窗，严格规则会误杀真实动作，且相机误判为瓶的反例仍会被放行。</p>
<p><a href="analysis-report.md">完整分析</a> · <a href="stats-appendix.md">统计与限制</a> · <a href="annotations.html">407个最终标注视图</a> · <a href="clip-coverage-454.csv">454片段对应CSV</a> · <a href="REPRODUCE.md">复现</a> · <a href="DELIVERY.md">交付说明</a></p>
<table><thead><tr><th>检查</th><th>结论</th></tr></thead><tbody>
<tr><td>旧物体test</td><td>微调裁剪：瓶4/12、纸7/12、展品5/7、手机0/3；杯/相机无留出支持。</td></tr>
<tr><td>旧动作test子集</td><td>正确喝水 原10→保守10→严格5 /11；正确捡物4→4→0 /4。</td></tr>
<tr><td>新9段固定规则回放</td><td>564个喝水/捡物窗；保守否决0，严格否决555。没有完整动作真值。</td></tr>
<tr><td>原454片段覆盖</td><td>0.85后363段覆盖同人同动作窗口，保守仍363段；不是重新保存的数量。</td></tr>
</tbody></table><h2>8段真实对照</h2><p>左侧是原视频连续播放；右侧是明确标记的冻结采样帧。标题是所选窗口的判断，不是新一轮连续实时输出。黄框是物体预测，不能直接当真值。</p><div class="grid">'''
    text+='\n'.join(cards)+'</div><h2>数值对照</h2>'+figures
    text+='<p>535历史来源按帧抽样审核，831条审核记录；不是逐帧密集标注。新9段及454片段未用于训练/选模。所有照片与视频均为真实数据，未生成或补造物体细节。</p></main></body></html>'
    (OUT/'index.html').write_text(text)
    (OUT/'demo-review.json').write_text(json.dumps(dict(review_type='assistant qualitative visual audit, not exhaustive action ground truth',
        selected_examples=[dict(id=r['id'],source_id=r['source_id'],event_index=r['event_index'],note=NOTES[r['id']])for r in demos],
        used_for_training_or_selection=False),ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
