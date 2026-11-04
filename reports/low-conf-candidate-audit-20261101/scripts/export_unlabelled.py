"""Export combined-rule newly recovered unlabelled windows as reviewable clips and HTML."""
import csv,html,json,subprocess
from pathlib import Path
from common import *

def main():
    rows={r['id']:r for r in read(OUT/'replay-rows.json')};raw={r['id']:r for r in read(ROOT/'reports/threshold-tuning-20261026/windows.json')};dec=read(OUT/'decisions.json');ev=read(OUT/'prior-evidence.json')
    ids=[r['id'] for r in rows.values() if r['bin']==2 and r['truth'] is None and dec[r['id']]['combined']['recovered']]
    assert len(ids)==10,ids
    source_info={v['id']:v for v in read(ROOT/'reports/wave-pose-prior-20261027/protocol.json')['old_sources']}
    export=OUT/'unlabelled-combined-085-0925';clips=export/'clips';posters=export/'posters';clips.mkdir(parents=True,exist_ok=True);posters.mkdir(exist_ok=True)
    items=[]
    for n,rid in enumerate(sorted(ids),1):
        r=dict(rows[rid],**{k:raw[rid][k] for k in ('source_frames','bbox')});v=source_info[r['source_id']];src=ROOT/v['source'];assert src.exists()
        # Keep enough context to see the preceding anchor and subsequent window.
        start=max(0.,float(r['timestamp'])-2.0);end=min(float(v['seconds']),float(r['timestamp'])+1.5);duration=max(.1,end-start)
        stem=f'{n:02d}_{rid}'
        clip=clips/(stem+'.mp4');poster=posters/(stem+'.jpg')
        cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-ss',f'{start:.6f}','-i',str(src),'-t',f'{duration:.6f}',
             '-vf','scale=960:-2','-an','-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p',str(clip)]
        subprocess.run(cmd,check=True)
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(clip),'-frames:v','1','-q:v','2',str(poster)],check=True)
        info=dict(review_index=n,id=rid,source_id=r['source_id'],source=str(v['source']),source_video=str(src),
            split=r['split'],track_id=r['track_id'],history_generation=r['history_generation'],
            event_time_s=r['timestamp'],source_frame=r['frame_index'],source_frames=r['source_frames'],
            action=r['raw_label'],score=r['score'],prior_support=r['support'],prior_reason=r['prior_reason'],
            prior_evidence=ev[rid],combined_decision=dec[rid]['combined'],truth=None,review_status='pending',
            clip_start_s=start,clip_end_s=end,clip=f'clips/{clip.name}',poster=f'posters/{poster.name}')
        items.append(info)
    save(export/'review.json',dict(schema_version=1,scope='combined_rule',bin=[.85,.925],items=items,
        note='These are recovered by the combined continuation + running-two-window replay but have no human label. They are not counted as TP or FP.'))
    fields=['review_index','id','source_id','event_time_s','track_id','history_generation','action','score','prior_support','prior_reason','clip_start_s','clip_end_s','clip','source_video','review_status']
    with(export/'review.csv').open('w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:x[k] for k in fields}for x in items])
    cards=[]
    for x in items:
        cards.append(f'''<article><h2>#{x['review_index']} {html.escape(x['id'])}</h2><p><b>Predicted action:</b> {html.escape(x['action'])} &nbsp; <b>score:</b> {x['score']:.4f} &nbsp; <b>track:</b> {x['track_id']} / G{x['history_generation']}<br><b>source time:</b> {x['event_time_s']:.3f}s &nbsp; <b>prior:</b> {html.escape(x['prior_reason'])}<br><b>source:</b> {html.escape(x['source'])}</p><video controls preload="metadata" poster="{html.escape(x['poster'])}" src="{html.escape(x['clip'])}"></video><p class="status">待人工确认：真实动作 / other / 不确定</p></article>''')
    (export/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>Low-confidence combined recovered windows</title><style>body{font:15px system-ui;margin:24px;background:#f3f5f7;color:#17283a}h1{margin-bottom:4px}.note{background:#fff3cd;padding:12px;border-left:4px solid #d49c18}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px}article{background:#fff;padding:14px;border-radius:8px;box-shadow:0 1px 5px #ccd}video{width:100%;max-height:360px;background:#111}.status{color:#885; font-weight:600}</style><h1>联合规则：0.85–0.925 未标注补回窗口</h1><div class="note">共10个窗口。它们由“已确认动作续接＋跑动双窗口复核”回放新增，但没有人工真值，不能直接视为真实动作或误报。请结合前后文审核。</div><p><a href="review.csv">下载CSV</a> · <a href="review.json">下载JSON</a></p><main>'''+''.join(cards)+'''</main>\n''')
    print(json.dumps(dict(export=str(export),items=len(items),clips=sum(x['clip'] is not None for x in items)),ensure_ascii=False))

if __name__=='__main__':main()
