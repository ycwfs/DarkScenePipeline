"""Create evidence figures and a local four-way video review page."""
from html import escape
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/dark-bright-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from replay import OUT, TITLES


def main() -> None:
    audit=json.loads((OUT/'visual-audit-results.json').read_text())['summary']['0.5']
    exhibit=json.loads((OUT/'exhibit-audit.json').read_text())['videos']
    names=['original','detector_only','pose_only','both']
    xt=['Original','Detector only','Pose only','Both tuned']
    fig,axs=plt.subplots(1,3,figsize=(12,4))
    metrics=['matched','true_associated_pose10','unmatched_associated_pose10']
    titles=['Box matches to 47 audit instances','Matched people with >=10 joints','Unmatched displayed skeletons']
    for ax,metric,title in zip(axs,metrics,titles):
        values=[audit[n][metric] for n in names]
        bars=ax.bar(xt,values,color=['#64748b','#df7851','#27857b','#bd5c4f'])
        ax.bar_label(bars,padding=3);ax.set_title(title,fontsize=10);ax.tick_params(axis='x',rotation=25)
        ax.set_ylim(0,52 if metric!='unmatched_associated_pose10' else 11)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('41 fixed bright test frames | assistant visual audit | no temporal filtering')
    fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(OUT/'figures'/f'audit-comparison.{suffix}',dpi=180)
    plt.close(fig)
    labels=[v['source'].split('/')[-1].replace('.mp4','') for v in exhibit]
    # English video IDs keep the scientific plots portable without font dependencies.
    labels=['other106','other95','other97','drink26','wave28','wave29','chase24','chase25','chase29']
    fig,ax=plt.subplots(figsize=(10,4.7));x=np.arange(len(exhibit));width=.35
    for offset,model,color in [(-width/2,'original','#64748b'),(width/2,'candidate','#bd5c4f')]:
        values=[100*v['counts'].get(model+'_detector',0)/v['frames'] for v in exhibit]
        bars=ax.bar(x+offset,values,width,label='Original detector' if model=='original' else 'Dark-tuned detector',color=color)
        ax.bar_label(bars,labels=[f'{v["counts"].get(model+"_detector",0)}/{v["frames"]}' for v in exhibit],fontsize=8,padding=3)
    ax.set_xticks(x,labels);ax.set_ylabel('Frames detecting exhibit as person (%)');ax.set_ylim(0,115)
    ax.set_title('Known humanoid exhibit: matched fixed region, IoU >= 0.5',pad=40)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False)
    ax.spines[['top','right']].set_visible(False);fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(OUT/'figures'/f'exhibit-regression.{suffix}',dpi=180)
    plt.close(fig)
    videos=json.loads((OUT/'replay-summary.json').read_text())
    video_cards=[];gallery=[]
    for v in videos:
        title=escape(v['source'].split('/')[-1]);uid=v['id']
        if v['video']:
            video_cards.append(f'<article><h2>{title}</h2><video controls preload="metadata" poster="posters/{uid}.jpg" src="{v["video"]}"></video><p>{v["frames"]} 帧，{v["fps"]:g} fps，源帧全部保留。<a href="{v["video"]}">下载 MP4</a></p></article>')
        gallery.append(f'<figure><a href="posters/{uid}.jpg"><img loading="lazy" src="posters/{uid}.jpg" alt="{title} 四组合中间帧"></a><figcaption>{title}</figcaption></figure>')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>亮场回归 · 无平滑四组合</title>
<style>body{font:16px/1.7 system-ui,sans-serif;color:#183044;background:#f4f6f8;max-width:1240px;margin:32px auto;padding:0 20px}h1{font-size:30px}h2{font-size:20px}article,section{background:white;border-radius:10px;padding:20px;margin:20px 0}video{width:100%;background:#111}a{color:#0b6477}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}figure{margin:0}img{max-width:100%}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;text-align:left;padding:10px}strong{color:#9a3529}</style>
<h1>暗场微调模型：亮场回归</h1>
<p><strong>检测器亮场退化，当前不建议两模型一起替换。</strong>单独换姿态模型的真人骨架覆盖有所改善；缺少独立人工关节点真值，尚不能证明定位精度完全保持。</p>
<section><p>41 段冻结测试原片、3,163 帧，原图 L≥75。四格统一增强、阈值固定，不做平滑、不做持续时间筛选。左上原检测＋原姿态；右上微调检测＋原姿态；左下原检测＋微调姿态；右下两者都微调。</p>
<p>9 段展品场景共 955 帧，展品被检测为人的帧数 39→944。原/新 RTMO 都能给展品生成骨架，检测器是此次假人显著增加的来源。</p>
<p>P 编号是当前帧的框序号，不是跟踪身份。本页只检查人体/姿态前端，不输出动作类别，不改照度门和平台包。</p><a href="README.md">查看完整报告</a> · <a href="stats-appendix.md">统计口径和限制</a></section>
'''+''.join(video_cards)+'''<details><summary>展开全部 41 段中间帧四组合对照</summary><div class="gallery">'''+''.join(gallery)+'''</div></details></html>'''
    (OUT/'index.html').write_text(html)


if __name__=='__main__':
    main()
