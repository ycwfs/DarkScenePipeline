"""Validate all final archive demos and produce a browsable evidence index."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import html
import json
from pathlib import Path

from analyze_live import video

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]

def read(path):
    return json.loads(path.read_text())

def link(path, title):
    return f'<a href="{html.escape(str(path.relative_to(OUT)), quote=True)}">{html.escape(title)}</a>'

def main():
    items = read(OUT/'inputs.json')
    summaries, checks, cards = [], [], []
    for item in items:
        folder = OUT/'archive-demos'/item['id']
        d = read(folder/'summary.json')
        rows = read(folder/'events.json')
        lumas = [r['raw_luma'] for r in read(folder/'frames.json') if r.get('raw_luma') is not None]
        brightness = sum(lumas)/len(lumas) if lumas else None
        stats = d['stats']
        for key in ('clips_failed','clips_forced','clips_abandoned','frames_dropped','write_errors','clips_unfinished'):
            assert stats['clips'].get(key, 0) == 0, (item['id'], key)
        assert all(e['threshold'] == .925 for e in rows)
        assert all(e['label'] == e['running_prior']['label'] for e in rows)
        for e in rows:
            prior = e['object_prior']
            assert prior['object_min_conf'] == .581 and prior['lum_gate'] == 35
            if prior['applied']:
                assert e['raw_luma'] >= 35
            if prior['reason'] == 'no_compatible_object':
                assert e['label'] == 'other'
        clips = list((folder/'clips').rglob('*.mp4'))
        checks.extend([folder/'output.mp4', *clips])
        summary = dict(id=item['id'],title=item['title'],source=item['input'],
            brightness=brightness,historical_brightness=item.get('historical_luma'),input_processed=item.get('processed', False),
            frames=stats['frames'],fps=stats['fps'],people_max=d['people_max'],
            events=len(rows),labels=d['labels'],
            clips=len(clips),context_complete=sum(read(p.with_suffix('.json')).get('context_complete',False) for p in clips),
            gpu_mib=d['nvidia_peak_mib'],rss_mib=d['rss_peak_bytes']/1024**2,
            torch_peak_mib=d['torch_peak_allocated']/1024**2,
            object_frames=stats['object_overlay']['frames_with_objects'],
            prior_reasons={k:dict(Counter(e[k]['reason']for e in rows))for k in ('object_prior','wave_prior','running_prior')})
        summaries.append(summary)
        title=f"{item['id']} — {item['title']}"
        description=f"原L={brightness:.1f}；最多检测 {d['people_max']} 人；{stats['fps']:.2f} FPS（离线）；{len(clips)} 片段" if brightness is not None else f"原L未知（已增强回传）；最多检测 {d['people_max']} 人；{stats['fps']:.2f} FPS（离线）；{len(clips)} 片段"
        clip_links=' · '.join(link(p,p.name) for p in clips) or '无动作片段'
        cards.append(f'<article><h3>{html.escape(title)}</h3><p>{html.escape(description)}</p>'
            f'<video controls preload="none" src="{folder.relative_to(OUT)}/output.mp4"></video>'
            f'<p>{link(folder/"events.json","逐人事件")}</p><p>{clip_links}</p></article>')
    with ThreadPoolExecutor(max_workers=2) as pool:
        decodes=list(pool.map(video, checks))
    assert all(d['passed'] for d in decodes)
    (OUT/'demos-video-verification.json').write_text(json.dumps(decodes,ensure_ascii=False,indent=2)+'\n')
    result=dict(passed=True,samples=len(items),frames=sum(s['frames']for s in summaries),
        events=sum(s['events']for s in summaries),clips=sum(s['clips']for s in summaries),
        context_complete=sum(s['context_complete']for s in summaries),
        decoded_videos=len(decodes),samples_detail=summaries,
        note='Historical diagnostic inputs, not independent accuracy. Persons are detector/tracker counts, not ground truth. File-boundary context is reported, not hidden.')
    (OUT/'demos-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    live=OUT/'live-final'
    live_clips=list((live/'clips').rglob('*.mp4'))
    clip_list=''.join(f'<li>{link(p,p.name)}</li>'for p in live_clips)
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Box 单卡优化验证</title>
<style>body{max-width:1100px;margin:40px auto;padding:0 18px;font-family:system-ui;line-height:1.6;color:#20252d}video{width:100%;max-height:550px;background:#111}article{margin:36px 0;border-top:1px solid #bbb;padding-top:12px}a{color:#1555aa}li{overflow-wrap:anywhere}.note{background:#fff2cf;padding:16px}</style>
<h1>NVIDIA-Box 单卡优化候选 · 2026-10-28</h1>
<p class="note">实测硬件为一张 RTX 3090，尚无 Orin Nano 4GB 真机结果。全部六个模型、阈值和先验保留；本轮采用 TensorRT 人体/物体 FP16、姿态 FP32，增强仍用 PyTorch。720 长边输出。有限测试不能保证无限流零故障。</p>
<p><a href="analysis-report.md">完整分析</a> · <a href="README.md">报告入口</a> · <a href="live-final-verification.json">流式原始统计</a> · <a href="demos-verification.json">短片原始统计</a></p>
<h2>实际推流的五分钟完整录制</h2><p>外部 FFmpeg 接收 RTMP 推送，包含不同光照、单/双人与拥挤场景，无像素条码。录制帧率不代表算法处理帧率。</p>
<video controls preload="metadata" src="live-final/full-output-5min.mp4"></video>
<h3>对应动作片段</h3><ul>'''+clip_list+'</ul><h2>23 个固定样本的完整结果</h2><p>以下数据包含历史训练/反馈样本，不能作为独立泛化准确率。没有触发也保留完整输出。短源视频不具备前后各 2 秒时，片段元数据标记上下文不足。</p>'+''.join(cards)+'</html>'
    (OUT/'index.html').write_text(document)
    print(json.dumps({k:v for k,v in result.items()if k!='samples_detail'},ensure_ascii=False))

if __name__=='__main__':
    main()
