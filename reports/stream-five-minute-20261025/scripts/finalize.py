"""Write the auditable report and a portable bundle of the actual videos."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]


def main():
    r=json.loads((OUT/'verification.json').read_text())
    source=json.loads((OUT/'input-manifest.json').read_text())
    protocol=json.loads((OUT/'live/protocol.json').read_text())
    align=json.loads((OUT/'output-alignment.json').read_text())
    recording=r['recording']['totals']
    clips=r['clips_index']
    text=f'''# 五分钟多动作真实流验证（2026-10-25）

[打开视频与片段检查页](index.html)。页面可跳转到29段原始素材的对应时间，并查看算子实际保存的全部片段；不删除误识别或无识别区间。

- [拼接输入视频](input-multiaction-5min.mp4)：{source['seconds']:.2f}秒，{source['frames']}帧，1920×1080、25fps，29份不同现场原片，保持原速。
- [完整处理后输出](output-multiaction-5min.mp4)：{r['output_seconds']:.2f}秒，1440×808、15fps；来自外部接收的实际输出流，仅裁除测试空白预热/收尾，原始推流15fps的时间戳取样与固定帧率封装会带来约0.36秒时长差。
- [未经裁剪的输出流录制](full-output-stream.mp4)：保留预热/收尾，供核对上述裁剪。
- [全部动作片段](live/clips/)：{r['clips']}段，完整目录、逐文件JSON和events.jsonl均保留；[片段索引](clips-index.csv)。
- [便携检查包](stream-review-5min-20261025.zip)：输入、输出、全部片段和检查页面，可解压后打开index.html。

## 使用的真实链路

当前`dark-demo-v2_person_tau075_20261024.tar.gz`独立解包，SHA256 `{protocol['package_sha256']}`。原`darkpipe-operator:0.3.0`镜像，ID `sha256:d359716620d7eb757cd7c36733e25642469450f1c801e0367fc99248f0a2f212`。物理GPU 0/1/4（RTX3090），容器可见0/1/2。动作阈值0.75、亮度门30、RGB权重0.95，720处理、双倍输出、默认15fps/3M出流、片段前后各2秒、15秒重叠分段，使用现有五模型。

原始拼接视频 → 按25fps实时发帧 → FFmpeg H.264/RTSP TCP → 输入MediaMTX → 算子实际RTSP解码与逐人推理 → 算子主动RTMP推送 → 输出MediaMTX → 外部FFmpeg通过RTSP接收同一路推送并录制。另有独立低延迟接收端核对实际帧号，三个时间点连接MJPEG、HTTP-FLV、HLS观察端，持续监听SSE，每秒采集health。

测试源以FFmpeg -re直接转发已编码视频，先发送75秒空白预热；确认算子和外部接收端在正式素材开始前已经收到视频，再从第0帧播放正式素材一次；末尾空白8秒让片段收尾，然后SIGTERM正常停止。主素材期间没有人为断流。该次验证正常持续运行；此前断流/输出故障恢复证据见[10-24原镜像测试](../person-release-20261024/README.md)，其阈值为0.91，不能拿来替代本轮阈值的保存负载结果。

首次诊断发现测试端逐帧解码再编码发送仅约21–22fps，放慢了源播放；该轮保留在diagnostic-raw-sender目录，不作为下面的原速成绩。正式轮改为已编码H.264直接按时间戳转发。

没有在平台算法内新增全流录像。`scripts/instrumented_operator.py`在推理后的展示画面左上角增加192×16像素帧号条，记录时间及内存；不改变模型输入、概率、跟踪、保存片段或平台包。该条用于确认接收端拿到的画面在更新，且帧号顺序不倒退。所有归档文件保持原哈希。

## 本次实际结果

| 核验项目 | 结果 |
|---|---|
| 正式输入 | {source['seconds']:.2f}秒／{source['frames']}帧；发布耗时{r['complete']['wall_playback_seconds']:.2f}秒 |
| 正式段健康采样 | {r['health_samples']}次，非ok {r['health_non_ok']}次 |
| 输入重连／主动推流重启／HLS重启 | {r['input_reconnects']}／{r['push_restarts']}／{r['hls_restarts']} |
| 处理FPS（正式段全部健康采样） | 中位{r['processing_fps']['median']:.2f}；最低{r['processing_fps']['min']:.2f} |
| 原图捕获到处理输出 | 中位{r['processing_latency_ms']['median']:.1f}ms；P95 {r['processing_latency_ms']['p95']:.1f}ms |
| 动作结果年龄 | P95 {r['model_decision']['decision_age_p95']:.3f}s；最大{r['model_decision']['max_decision_age']:.3f}s |
| 原图捕获到低延迟观察端 | P95 {r['capture_to_receiver_seconds']['p95']:.3f}s；最大{r['capture_to_receiver_seconds']['max']:.3f}s |
| 观察端新画面最大间隔 | {r['receiver_unique_frame_gap_seconds']['max']:.3f}s；正式输入期间接收断开{len(r['receiver_errors_during_input'])}次 |
| 实际处理正式段 | {r['processed_main_frames']}帧；捕获跨度{r['main_capture_span_frames']}帧，其中{r['capture_frames_not_processed']}帧未进入处理 |
| 动作片段 | {r['clips']}段；逐文件上下文标为完整{r['context_complete']}段 |
| 片段保存异常 | 失败{recording['clips_failed']}、强制释放{recording['clips_forced']}、未完成{recording['clips_unfinished']}、录制队列丢帧{recording['frames_dropped']}、放弃{recording['clips_abandoned']} |
| 事件日志 | {r['event_log_stats']['events_written']}条，丢弃{r['event_log_stats']['events_dropped']}条；SSE接收{r['sse']['events']}条 |
| 输出与片段解码 | {r['decoded_video_files']}个视频全部通过FFmpeg严格完整解码 |
| 正常停止 | 退出{r['complete']['service_returncode']}，会话{r['complete']['session_status']}，子进程已收尾 |

本轮有5次health返回`degraded`，均为HTTP200、采集/处理正常、HLS/推流存活且无重启；原因字段为短暂写入“阻塞”。当前实现对`blocked_s()>0`立即降级，却把显示时长四舍五入到0.1秒，因此瞬时正常写入也可能报`wedged_s=0.0`的降级。它说明健康状态存在误报敏感性，不代表这5次是真实断流；接收端帧顺序、连续接收与最大新画面间隔提供了独立证据。此次保留原包行为并记录该问题，未为得到全绿结果调整健康规则。SSE接收499条，与事件日志一致；SSE状态中连接拒绝均发生在主动停止之后，不计为正式输入期间断线。

处理帧率与输出帧率不同：保持低延迟时会跳过部分输入帧，推流以15fps取最新处理画面，必要时重复上一帧。这里的“完整输出”指完整时间段的真实输出流，不宣称7514个输入帧逐帧全部经过模型，也没有离线补跑漏帧后替换视频。采集到接收延迟不包括摄像头自身编码/送流到采集的延迟；本地源第一张非空白画面进入采集约延迟{align['first_input_capture_delay']:.3f}秒。

## 动作与片段如何核查

素材包含挥手、喝水、追逐、捡东西、抛物、摔倒与other；包含历史训练/验证/测试来源，只用于工程回放和人工检查，未新增训练或据此调参。拼接没有人为压暗、加亮、重复或慢放，帧率统一带来最多约一帧的时间量化。文件名类别不是逐人逐帧真值，旁观者仍可能为other。

实际事件类别：`{json.dumps(r['event_counts'],ensure_ascii=False)}`。
实际保存片段类别：`{json.dumps(r['clip_labels'],ensure_ascii=False)}`。
片段按人物＋动作分别保存完整画面，跨人物时间重叠共{len(r['cross_person_overlaps'])}对，详见verification.json。上下文不足或达到15秒分段的文件在检查页明确列出；“上下文完整”仅指识别触发前后缓存覆盖，不等于真实动作起止均被正确识别。

29段原片之间是人为硬切，不是同一摄像机连续拍摄；跟踪和1秒识别窗口可能在切镜附近跨场景。测试未注入真值重置或人为纠正结果。误识别、漏识别、片段跨切镜都应在视频中检查。

## 证据与复现

- [输入来源与时间索引](input-manifest.json)、[输入概览](input-contact-sheet.jpg)、[实际输出抽帧](output-contact-sheet.jpg)
- [完整核验结果](verification.json)、[逐视频解码记录](video-verification.json)、[输出帧号及裁剪边界](output-alignment.json)
- `live/protocol.json`：完整容器启动命令；`live/effective-config.json`：真实生效配置
- `live/health-history.json`、`live/frames.jsonl`、`live/receiver.json`、`live/source-timeline.json`、`live/source-progress.jsonl`
- `live/operator.log`、`live/session.json`、`live/complete.json`、`live/cleanup.json`、`live/sse-status.json`
- `scripts/build_input.py`、`scripts/run_live.py`、`scripts/analyze.py`、`scripts/finalize.py`

这是约5分钟、本机真实协议和本地存储的验证，不是现场RTSP摄像机、NFS/HDFS、vGPU资源及长时间网络波动验收，不能保证无限时长无故障。整段录制与检查文件仅供本次人工验证，生产算子包、镜像和默认参数未改变。
'''
    (OUT/'README.md').write_text(text)
    (OUT/'STATUS.md').write_text('# 已完成\n\n五分钟真实流输入、输出录制、动作片段及严格解码核验已完成。入口：[README.md](README.md)／[视频检查页](index.html)。\n')
    with (OUT/'source-timeline.csv').open('w',newline='') as f:
        fields=['index','start','end','source','label','raw_mean_luma','source_split']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for s in source['segments']:writer.writerow({k:s[k] for k in fields})
    files=[OUT/p for p in ['README.md','STATUS.md','index.html','input-manifest.json','input-contact-sheet.jpg',
        'output-contact-sheet.jpg','source-timeline.csv','clips-index.csv','verification.json','video-verification.json','output-alignment.json',
        'input-multiaction-5min.mp4','output-multiaction-5min.mp4','full-output-stream.mp4']]
    files+=list((OUT/'scripts').glob('*.py'))
    files+=[p for p in (OUT/'live/clips').rglob('*') if p.is_file()]
    files+=[p for p in (OUT/'live').glob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.log')]
    archive=OUT/'stream-review-5min-20261025.zip'
    records={}
    with zipfile.ZipFile(archive,'w') as z:
        for path in sorted(set(files)):
            name=path.relative_to(OUT).as_posix()
            z.write(path,name,compress_type=zipfile.ZIP_STORED if path.suffix in ('.mp4','.jpg') else zipfile.ZIP_DEFLATED)
            h=hashlib.sha256(path.read_bytes()).hexdigest();records[name]=dict(bytes=path.stat().st_size,sha256=h)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    h=hashlib.sha256(archive.read_bytes()).hexdigest()
    Path(str(archive)+'.sha256').write_text(f'{h}  {archive.name}\n')
    (OUT/'review-manifest.json').write_text(json.dumps(dict(archive=archive.name,sha256=h,files=records),ensure_ascii=False,indent=2)+'\n')
    print('Review bundle:',archive.name,archive.stat().st_size,h,flush=True)

if __name__=='__main__':main()
