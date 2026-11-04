"""Decode all delivered videos, align the received stream, and build a review index."""
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import subprocess
import zipfile

import cv2
import numpy as np

OUT=Path(__file__).resolve().parents[1]
LIVE=OUT/'live'
ZH={'wave':'挥手','drink':'喝水','chase':'追逐','pick_up':'捡东西','throw':'抛物','fall':'摔倒','other':'其他',
    'Waving':'挥手','Drinking water':'喝水','Chasing':'追逐','Picking up':'捡东西','Throwing':'抛物','Falling':'摔倒',
    'Picking up object':'捡东西','Throwing object':'抛物'}


def load(path):return json.loads(path.read_text())
def lines(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def stats(values):
    if not values:return None
    a=np.asarray(values,dtype=float)
    return dict(n=len(values),min=float(a.min()),median=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(a.max()))
def stamp(seconds):return f'{int(seconds)//60:02d}:{seconds%60:05.2f}'
def decode(path):
    result=subprocess.run(['ffmpeg','-nostdin','-v','error','-xerror','-threads','2','-i',str(path),'-f','null','-'],capture_output=True,text=True)
    if result.returncode or result.stderr.strip():raise RuntimeError(f'{path}: {result.stderr}')
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
        '-show_entries','stream=width,height,codec_name,avg_frame_rate,nb_read_frames:format=duration', '-of','json',str(path)]))
    return dict(path=str(path.relative_to(OUT)),duration=float(info['format']['duration']),**info['streams'][0],full_decode=True)


def main():
    cv2.setNumThreads(2)
    source=load(OUT/'input-manifest.json');timeline=load(LIVE/'source-timeline.json')
    complete=load(LIVE/'complete.json');session=load(LIVE/'session.json')
    assert abs(complete['wall_playback_seconds']-source['seconds'])<1., 'Sender did not maintain original playback speed'
    config=load(LIVE/'effective-config.json')
    assert config['reco_min_conf']==.75 and config['lum_gate']==30 and config['skel_fusion_alpha']==.95
    assert config['person_mode'] and config['record']=='', 'Production full recorder must remain disabled'
    frames=lines(LIVE/'frames.jsonl');frame_map={r['frame']:r for r in frames}
    actual=[r for r in frames if r['raw_luma'] is not None and r['raw_luma']>1.]
    assert actual, 'No real source image processed'
    anchor=actual[0]['timestamp'];origin=actual[0]['origin']
    capture_min,capture_max=actual[0]['frame'],actual[-1]['frame']
    receiver=load(LIVE/'receiver.json')
    received=[r for r in receiver['frames'] if capture_min<=r['frame']<=capture_max and r['frame'] in frame_map]
    latencies=[r['time']-frame_map[r['frame']]['origin']-frame_map[r['frame']]['timestamp'] for r in received]
    unique=[];seen=set()
    for r in received:
        if r['frame'] not in seen:unique.append(r);seen.add(r['frame'])
    health=[r for r in load(LIVE/'health-history.json') if timeline['source_start_monotonic']<=r['monotonic']<=timeline['source_end_monotonic']]
    good=[r['health'] for r in health if 'health' in r]
    events=lines(next((LIVE/'clips').rglob('events.jsonl')))
    assert all(r['threshold']==.75 for r in events)
    # The barcode exists only in displayed output, after inference. Decode it from the actual network recording.
    cap=cv2.VideoCapture(str(OUT/'full-output-stream.mp4'));assert cap.isOpened()
    fps=cap.get(cv2.CAP_PROP_FPS);output_index=0;first=last=None;marker_rows=[];unmapped=0
    while True:
        ok,image=cap.read()
        if not ok:break
        marker=sum(int(image[4:12,bit*8+2:bit*8+6].mean()>127)<<bit for bit in range(24))
        row=frame_map.get(marker)
        if row is None:unmapped+=1
        elif capture_min<=marker<=capture_max:
            if first is None:first=output_index
            last=output_index
            marker_rows.append(dict(output_frame=output_index,capture_frame=marker,source_time=row['timestamp']-anchor))
        output_index+=1
    cap.release();assert first is not None and last is not None
    assert all(b['capture_frame']>=a['capture_frame'] for a,b in zip(marker_rows,marker_rows[1:])), 'Received frame order regressed'
    start,end=first/fps,(last+1)/fps
    trimmed=OUT/'output-multiaction-5min.mp4'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','2','-i',str(OUT/'full-output-stream.mp4'),
        '-vf',f'trim=start_frame={first}:end_frame={last+1},setpts=PTS-STARTPTS','-an','-c:v','libx264',
        '-preset','veryfast','-crf','18','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart','-y',str(trimmed)],check=True)
    save(OUT/'output-alignment.json',dict(untrimmed_frames=output_index,fps=fps,trim_start_frame=first,
        trim_end_frame=last+1,trim_start_seconds=start,trim_duration_seconds=end-start,
        capture_first=capture_min,capture_last=capture_max,capture_time_anchor=anchor,
        unmatched_barcodes=unmapped,marker_order_monotonic=True,
        first_input_capture_delay=origin+anchor-timeline['source_start_monotonic'],
        note='Trim only blank preroll/tail using output markers; do not remove failed recognition intervals or reconstruct frames offline.',
        received_frames=marker_rows))
    validations=[decode(OUT/source['input']),decode(OUT/'full-output-stream.mp4'),decode(trimmed)]
    clips=[]
    for path in sorted((LIVE/'clips').rglob('*.mp4')):
        validations.append(decode(path));m=load(path.with_suffix('.json'))
        a,b=m['start_source_time']-anchor,m['end_source_time']-anchor
        segments=[s['index'] for s in source['segments'] if s['start']<b and s['end']>a]
        clips.append(dict(file=str(path.relative_to(OUT)),metadata=str(path.with_suffix('.json').relative_to(OUT)),
            person_id=m['person_id'],label=m['label'],action_key=m['action_key'],
            confidence=m['confidence'],source_start=a,source_end=b,duration=validations[-1]['duration'],
            context_complete=m['context_complete'],pre=m['pre_context_available'],post=m['post_context_available'],
            closed_because=m['closed_because'],source_segments=segments,frames_written=m['frames_written']))
    clips.sort(key=lambda r:(r['source_start'],r['person_id']))
    for i,c in enumerate(clips,1):c['index']=i
    for path in sorted(LIVE.glob('viewer-*.mkv')):validations.append(decode(path))
    overlaps=[]
    for i,a in enumerate(clips):
        for b in clips[i+1:]:
            duration=min(a['source_end'],b['source_end'])-max(a['source_start'],b['source_start'])
            if duration>0 and a['person_id']!=b['person_id']:
                overlaps.append(dict(first=a['index'],second=b['index'],seconds=duration))
    summary=dict(source_seconds=source['seconds'],source_frames=source['frames'],source_segments=len(source['segments']),
        unique_sources=source['no_repeated_sources'],complete=complete,configuration=config,
        processed_main_frames=len(actual),main_capture_span_frames=capture_max-capture_min+1,
        capture_frames_not_processed=capture_max-capture_min+1-len(actual),
        output_seconds=end-start,output_main_frames=last-first+1,output_unmapped_markers=unmapped,
        health_samples=len(health),health_non_ok=sum(r.get('health',{}).get('status')!='ok' for r in health),
        health_non_ok_samples=[dict(elapsed=r['elapsed'],http_status=r.get('http_status'),
            error=r.get('error'),details={k:r.get('health',{}).get(k) for k in
            ('status','degraded_reason','frame_age_s','processing_error','push_alive','push_restarts',
             'push_wedged_s','hls_alive','hls_restarts','hls_wedged_s')})
            for r in health if r.get('health',{}).get('status')!='ok'],
        processing_fps=stats([r['fps_proc'] for r in good]),
        processing_latency_ms=stats([r['latency_ms_last'] for r in good]),
        maximum_frame_age_s=max(r['frame_age_s'] for r in good),
        input_reconnects=max(r['reconnects'] for r in good),push_restarts=max(r['push_restarts'] for r in good),
        hls_restarts=max(r['hls_restarts'] for r in good),
        capture_to_receiver_seconds=stats(latencies),
        receiver_unique_main_frames=len(unique),receiver_unique_frame_gap_seconds=stats([b['time']-a['time'] for a,b in zip(unique,unique[1:])]),
        receiver_errors_during_input=[r for r in receiver['errors'] if timeline['source_start_monotonic']<=r['time']<=timeline['source_end_monotonic']],
        model_decision=session['person'],event_counts=dict(Counter(r['label'] for r in events)),
        event_log_stats=session['event_log_stats'],sse=load(LIVE/'sse-status.json'),
        recording={k:v for k,v in session['clips'].items() if k!='clips'},
        clips=len(clips),clip_labels=dict(Counter(r['label'] for r in clips)),
        context_complete=sum(c['context_complete'] for c in clips),cross_person_overlaps=overlaps,
        all_videos_decode=True,decoded_video_files=len(validations),
        caveats=['Local loopback and local storage, not a real camera/NFS/HDFS/platform soak test',
            'Synthetic scene cuts and historically seen sources; no accuracy benchmark or guarantee of all true actions',
            'Streaming latest-frame scheduling drops input frames and can repeat output frames; continuous duration is not every-frame inference',
            'Capture-to-receiver latency excludes camera/encoder-to-capture delay'],clips_index=clips)
    save(OUT/'verification.json',summary);save(OUT/'video-verification.json',validations)
    with (OUT/'clips-index.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(clips[0]) if clips else ['index']);writer.writeheader();writer.writerows(clips)
    make_review(source,summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('configuration','clips_index','recording','cross_person_overlaps')},ensure_ascii=False,indent=2),flush=True)


def make_review(source,summary):
    esc=html.escape
    rows=[]
    for s in source['segments']:
        labels=Counter(c['label'] for c in summary['clips_index'] if s['index'] in c['source_segments'])
        result='、'.join(f'{ZH.get(k,k)}×{v}' for k,v in labels.items()) or '未保存动作片段'
        rows.append(f'<tr><td>{s["index"]}</td><td><button onclick="seek({s["start"]})">{stamp(s["start"])}–{stamp(s["end"])}</button></td><td>{esc(s["source"])}</td><td>{ZH[s["label"]]}</td><td>{s["raw_mean_luma"]:.1f}</td><td>{esc(result)}</td></tr>')
    cliprows=[]
    for c in summary['clips_index']:
        label=ZH.get(c['label'],c['label'])
        cliprows.append(f'<tr><td>{c["index"]}</td><td>T{c["person_id"]:03d}</td><td>{esc(label)} {c["confidence"]:.1%}</td><td><button onclick="seek({max(0,c["source_start"])})">{stamp(max(0,c["source_start"]))}–{stamp(c["source_end"])}</button></td><td>{c["duration"]:.2f}s</td><td>{"完整" if c["context_complete"] else "不足/分段"}（前{c["pre"]:.2f}s／后{c["post"]:.2f}s）</td><td><a href="{esc(c["file"])}" onclick="showClip(this.href);return false">播放片段</a> · <a href="{esc(c["metadata"])}">元数据</a></td></tr>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>五分钟真实流验证</title>
<style>body{font-family:system-ui,sans-serif;background:#f4f6f8;color:#172a3c;margin:24px auto;padding:0 20px;max-width:1500px}h1{font-size:26px}p{line-height:1.7}section{background:white;padding:20px;margin:20px 0;border-radius:10px}.videos{display:grid;grid-template-columns:1fr 1fr;gap:16px}video{width:100%;background:#111}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;border-bottom:1px solid #dce2e8;padding:9px}button,a{color:#086c9a;cursor:pointer}button{border:0;background:#eaf4fa;padding:6px;border-radius:4px}.scroll{overflow-x:auto}#clip{max-width:1000px}@media(max-width:850px){.videos{grid-template-columns:1fr}}</style>
<h1>五分钟现场多动作 · 真实流式输入输出验证</h1>
<p>阈值0.75 · 原图亮度门30 · RGB权重0.95 · 原0.3.0镜像 · GPU 0、1、4。输出视频由外部客户端接收实际推流后录制。左上角细小黑白条是推流核验帧号，位于模型推理之后；动作片段没有该标记。</p>
<section><div class="videos"><div><h2>拼接输入</h2><video id="input" controls preload="metadata" src="input-multiaction-5min.mp4"></video></div><div><h2>完整处理后输出</h2><video id="output" controls preload="metadata" src="output-multiaction-5min.mp4"></video></div></div><p><button onclick="both()">两侧从当前输入时间一起播放</button> · <a href="full-output-stream.mp4">含预热/收尾的原始输出流录制</a> · <a href="verification.json">核验数据</a> · <a href="README.md">测试说明</a></p><p>按源时间近似对齐；真实流有处理与传输延迟。下表的源动作名称来自文件类别，不是对每个人每帧的真值标注。未识别、误识别和切换过程均保留。</p></section>
<section><h2>按原始素材定位</h2><div class="scroll"><table><thead><tr><th>段</th><th>输入时间</th><th>原始文件</th><th>源类别</th><th>原L</th><th>交叠的保存片段</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div></section>
<section id="clip-section"><h2>算子实际保存的全部动作片段</h2><video id="clip" controls preload="none"></video><div class="scroll"><table><thead><tr><th>编号</th><th>人物</th><th>预测动作</th><th>原视频近似区间</th><th>时长</th><th>上下文</th><th>查看</th></tr></thead><tbody>'''+''.join(cliprows)+'''</tbody></table></div><p>每人＋动作独立保存完整画面，允许重叠。完整性指识别触发前后的缓存覆盖，不代表模型已识别到真实动作的全部起止。拼接切镜附近可出现跨镜头缓存和身份重建。</p></section>
<script>const input=document.getElementById('input'),output=document.getElementById('output');function seek(t){input.currentTime=t;output.currentTime=t;document.querySelector('.videos').scrollIntoView({behavior:'smooth'});}function both(){output.currentTime=input.currentTime;input.play();output.play();}function showClip(url){const v=document.getElementById('clip');v.src=url;v.play();document.getElementById('clip-section').scrollIntoView({behavior:'smooth'});}</script></html>'''
    (OUT/'index.html').write_text(page)

if __name__=='__main__':main()
