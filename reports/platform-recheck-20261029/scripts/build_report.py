"""Build a traceable review index from completed checks, without suppressing imperfect results."""
from collections import Counter
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET

OUT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((OUT / name).read_text())


def main():
    package = read('package-verification.json')
    live, extra = read('live-verification.json'), read('live-extended-verification.json')
    faults, demo = read('fault-verification.json'), read('demo-verification.json')
    clips, samples = read('live-clips-check.json'), read('demo-results.json')
    trim = read('stream-trim.json')
    assert all(r['passed'] for r in (package, live, extra, faults, demo, clips))
    junit = ET.parse(OUT/'cpu-junit.xml').getroot()
    cases = list(junit.iter('testcase'))
    errors = [c for c in cases if c.find('failure') is not None or c.find('error') is not None]
    assert not errors, 'CPU regressions are not all passing'
    cpu = dict(passed=len(cases)-sum(c.find('skipped') is not None for c in cases),
               skipped=sum(c.find('skipped') is not None for c in cases), deselected=1,
               optional_exclusion='SciPy comparison only; production NumPy assignment covered by remaining tests')
    current_sha = package['sha256']
    md = [
        '# 平台完整功能与实时稳定性复核 · 2026-10-29',
        '',
        '**当前平台包确实对应 `reports/full-prior-release-20261027` 的方案和实测结果。** '
        f'本次重新解包默认TAR，69个文件与该报告当时实际运行目录和当前源码逐字节一致，SHA256为 `{current_sha}`。'
        'Box单卡优化没有进入平台包；本轮没有改动生产代码、模型、参数、镜像或提交归档。',
        '',
        '本轮在原 `darkpipe-operator:0.3.0` 镜像中完成15分钟连续真实RTSP→RTMP、独立110秒故障恢复、'
        '23个历史亮暗/多人样本及功能回归。已验证范围内服务正常收尾，输出和动作片段可解码；'
        '高密度多人仍可能低于15 FPS，既有误报/漏报仍存在；另发现小型轨迹调度元数据持续累积的长期隐患。此处“功能完整”指工程链路和既定规则齐备，不表示所有动作均识别正确或全天候稳定已验收。',
        '',
        '- [视频检查页：15分钟完整输出、片段和23个样本](index.html)',
        '- [包来源核对](package-verification.json) / [实际运行配置](live/effective-config.json) / [实测命令](live/protocol.json)',
        '- [实时校验](live-verification.json) / [循环、内存及额外检查](live-extended-verification.json) / [片段与最终事件逐项核对](live-clips-check.json)',
        '- [故障恢复](fault-verification.json) / [样本复跑及与旧结果比较](demo-results.json) / [CPU回归日志](cpu-run.log)',
        '',
        '| 功能 | 本轮验证 |', '|---|---|',
        '| 照度增强、人体检测与跟踪 | 真实六模型推理；23段历史原始/已增强输入，多档亮度、多人人数 |',
        '| 逐人RGB＋骨架融合 | 同人ROI/骨架；低照度门、缺骨架fallback、动作阈值均有回归与真实事件证据 |',
        '| 物体、挥手、跑动先验 | 物体单帧支持/否决、同腕3帧、0.35框高/秒；事件保存每项原因 |',
        '| 可视化 | 人体框、人物动作、条件启用的骨架与青色物体框；无整体动作底栏/测试条码 |',
        '| 实时输入与输出 | RTSP输入；RTMP主动推送及外部完整录制；MJPEG/FLV/HLS各6次接入；SSE、health、config |',
        '| 多人动作片段 | 按人物ID＋动作保存全画面、允许重叠；前后2秒、15秒续段；逐片段匹配最终人物/动作事件 |',
        '| 中断与退出 | 暂停输入、断开输入、推流中继断开、观看端重连；SIGTERM排空及错误路径回归 |',
        '| 文件/HDFS路径 | 历史样本实际离线视频与片段；WebHDFS本地协议替身回归；现场HDFS/NFS未接入 |',
        '',
        '默认仍为RGB0.972、骨架0.028、原图照度门35、动作阈值0.925、物体阈值0.581、挥手3帧、'
        '跑动0.35框高/秒；模型处理长边720，平台默认bicubic×2显示放大至1440。低亮仅RGB并跳过三类先验；'
        '原图亮度未知时先验弃权；亮场挥手缺姿态拒绝，跑动缺可靠速度保留。',
        '',
        '## 15分钟真实流', '',
        '输入是原报告固定300秒视频的三次连续拼接，直接复制编码数据，不重新调亮、不插条码。'
        '跟踪不在场景切换时人工重置。原镜像挂载独立解包的算子，观察包装器只写元数据，不改变像素或决策。'
        '完整视频由外部FFmpeg接收实际RTMP推流并录制，平台无限流模式本身没有启用整段录像。', '',
        '| 指标 | 实测 |', '|---|---|',
        f'| 输入 | 900秒 / 22,500源帧 / 三张RTX3090 |',
        f'| 实际处理帧 | {live["main_frames_processed"]}，平均 {live["main_frames_processed"]/900:.2f} FPS |',
        f'| health处理FPS | 中位 {live["fps_proc"]["median"]:.2f}，最低 {live["fps_proc"]["minimum"]:.2f} |',
        f'| 采集交付后→处理输出 | P95 {live["capture_to_output_ms"]["p95"]:.1f} ms，最大 {live["capture_to_output_ms"]["maximum"]:.1f} ms |',
        f'| 动作结果年龄 | P95 {live["action_result_age_seconds"]["p95"]:.3f} s，最大 {live["action_result_age_seconds"]["maximum"]:.3f} s |',
        f'| 输出间隔最大值 | {live["output_intervals_seconds"]["maximum"]:.3f} s；大于1秒 {live["output_gaps_over_1s"]} 次 |',
        f'| 输入重连 / RTMP重启 / HLS重启 | {live["reconnects"]} / {live["push_restarts"]} / {live["hls_restarts"]} |',
        f'| 动作片段 / 上下文完整 | {clips["clips"]} / {clips["complete"]} |',
        f'| 写入失败 / 强制收尾 / 弃段 / 写队列丢帧 | {live["final_health_clip_counters"]["clips_failed"]} / {live["final_health_clip_counters"]["clips_forced"]} / {live["final_health_clip_counters"]["clips_abandoned"]} / {live["final_health_clip_counters"]["frames_dropped"]} |',
        f'| HTTP观看端 / SSE事件 | {len(extra["http_clients"])}次观看端录像均可解码 / {extra["sse"]["events"]}条 |',
        f'| 完整解码校验 | {live["decoded_videos"]}个视频，另含{trim["seconds"]:.3f}秒裁去黑色预热/尾部的展示版 |',
        '',
        '输入丢旧帧和动作窗口合并用于限制实时延迟；源帧未全部推理不等于片段写盘丢帧。'
        f'动作窗口合并累计{live["coalesced_action_tasks"]}次。固定出流节拍可能重复上一幅图，因此播放器帧率不代表模型处理帧率。',
        '',
        '| 连续5分钟周期 | 处理帧数 / 平均FPS | health中位FPS | 动作年龄P95(s) | 父进程RSS峰值(MiB) |',
        '|---|---|---|---|---|',
    ]
    for c in extra['cycles']:
        md.append(f'| {c["cycle"]} | {c["frames"]} / {c["average_processed_fps"]:.2f} | {c["health_fps"]["median"]:.2f} | {c["action_age"]["p95"]:.3f} | {c["parent_rss_mib"]["maximum"]:.1f} |')
    md += ['', f'物体历史最大{extra["bounded_object_history_bytes"]/1024**2:.1f}MiB，低于128MiB上限；'
           '各GPU显存和父进程RSS记录见live-gpu.csv及扩展JSON。父进程RSS不含全部工作进程，三周期也不能证明长期不存在慢泄漏。', '',
           f'health状态计数为 `{live["health_statuses"]}`。短时degraded的逐次输出/采集状态保留在扩展JSON，'
           '需结合输出是否存活、最近帧和实际接收判断，不能只凭HTTP 200判健康，也不能把每个degraded都当成断流。', '',
           '本轮23次短时degraded均发生在推流/HLS存活、最近帧不超过0.20秒时，显示的写入阻塞时长均四舍五入为0.0秒；'
           '与旧报告相同，健康检查对瞬时写入偏敏感。实际处理输出最大间隔0.453秒，外部完整接收与观看端解码均通过，'
           '没有证据把这些短时状态判作实际断流。该监控行为本轮未改。', '',
           '实时任务独占GPU0/1/2；样本复跑使用GPU4，CPU回归与样本任务共享主机CPU、内存和磁盘。'
           '本轮用于再次验证稳定性，不能与10月27日数字作严格的隔离吞吐提升/下降归因。', '',
           '## 片段与中断', '',
           f'本次{clips["clips"]}个实时片段均找到同一人物/代次/动作的最终触发事件；'
           f'不同人物片段存在{extra["overlapping_pairs"]}对时间重叠，其中{extra["different_action_overlapping_pairs"]}对动作不同。'
           '另有真实H.264回归覆盖8人同时动作、同人动作切换、长动作重叠续段。', '',
           '未满足前后2秒的片段没有隐藏：', '']
    for c in clips['records']:
        if not c['complete']:
            md.append(f'- [T{c["person_id"]} {c["action"]}]({c["path"]})：前{c["pre"]:.3f}s，后{c["post"]:.3f}s；{c["closed_because"]}。')
    if all(c['complete'] for c in clips['records']):
        md.append('全部实时片段满足元数据定义的上下文要求。')
    continuations = clips.get('max_length_continuations', [])
    if continuations:
        md += ['', '达到15秒上限的续段核对：']
        for c in continuations:
            md.append(f'- [前段]({c["first"]}) / [后段]({c["next"]})：重叠{c["overlap_seconds"]:.3f}s。')
    md += ['', '片段上下文完整只说明识别触发前后的已处理视频保存齐全，不证明动作模型从真实动作开始就识别到了。'
           '多个动作靠得很近时，前后文中仍可能出现另一动作；断流无法补回缺失源画面。', '',
           '独立110秒受控测试：输入暂停8秒、推流中继断开8秒、输入断开8秒，并重连观看者。', '',
           '| 恢复场景 | health恢复(s) | 接收端再次有帧(s) |', '|---|---|---|']
    labels = {'source_resume':'输入恢复发送', 'output_up':'推流中继恢复', 'source_up':'输入重连'}
    for name, value in faults['recoveries'].items():
        md.append(f'| {labels[name]} | {value["health_ok_seconds"]:.3f} | {value["receiver_activity_seconds"]:.3f} |')
    md += ['', f'故障测试保存{faults["clips_saved"]}段，其中{faults["clips_context_complete"]}段上下文完整；'
           f'写错/强制收尾/弃段为 `{faults["final_clip_counters"]}`，帧丢弃{faults["clip_frame_drops"]}。'
           '接收活动可能包括输入丢失提示或编码器旧帧，不能当作新鲜源端到端恢复时间；暂停后在途帧仍可能短暂完成。', '',
           '## 样本与功能回归', '',
           f'原报告23个输入均按原文件哈希复用。本轮处理{demo["frames"]}帧、{demo["events"]}人物窗口，'
           f'保存{demo["clips"]}片段，{demo["decoded_videos"]}个完整视频/片段严格解码。'
           f'与旧结果按人物、代次和32个源帧严格匹配{demo["matched_windows"]}窗，其中最终标签改变{demo["changed_matched_labels"]}窗。'
           '无匹配窗口不强行配对；原报告与本次都是历史数据回归，不是新增独立准确率。', '',
           f'原镜像中CPU功能回归{cpu["passed"]}项通过；1项需要镜像未安装的可选SciPy对照被显式排除，'
           '未加入生产依赖。装载模块路径核验为独立归档/app，测试中直接读取的platform源码也挂载同一归档。', '',
           '首次扩大到旧测试集时出现9项失败，原始日志与JUnit已保留。原因是7项FFmpeg替身没有fileno、'
           '1项只检查生命周期包装函数而未检查实际渲染函数、1项未考虑为end收尾保留队列位置。'
           '只修正tests/test_streams.py与tests/test_clips.py：替身使用真实管道fd、定位实际渲染函数、验证仅剩收尾位置时不得开录。'
           '补齐替身后又暴露两项旧假设：异步出流用ring_full记录写端背压，旧测试只看同步出流的skipped；'
           '测试恢复真实队列时丢掉了已经入队的start，现将其交回写线程并验证片段实际保存。'
           '生产文件未改；保留修正前后结果，不把旧假设失败包装成实际服务故障。', '',
           '## 长期运行审查发现', '',
           '[元数据生命周期复现](long-session-metadata.json)直接调用当前包的ActionPool.submit（CPU替身邮箱，不运行模型）：'
           '连续提交50,000个不同人物/代次后，仅重复最后一个身份10,000次，assignment仍保留50,000项。'
           '仅字典与tuple的浅层大小约5.55MiB，未包括所有整数对象。PersonEngine的last_seen/generations/last_submitted也只在输入代次变化时清空，连续输入下不按离开人物回收。', '',
           '这是明确的长期元数据回收缺口；不等同于已发生OOM，也不能据此解释本轮RSS的全部增长。'
           '图像/动作缓存另有上限，15分钟测试中没有因此中断。当前交付包保持原样，以便本轮结论严格对应用户指定版本；'
           '在宣称7×24小时稳定前，应增加过期人物/代次元数据回收并进行更长时间压力验证。', '',
           '## 部署结论和边界', '',
           '当前正式平台包的增强、逐人识别与先验、可视化、实时出流、事件和多人片段保存链路具备，'
           '本轮有限时长测试没有发现新的服务中断或录制故障。'
           '多人高密度吞吐和既有识别误报/漏报仍按实测保留，不能承诺所有人数均15FPS或无限时长不出错。', '',
           '本机三张RTX3090、localhost中继和本地盘无法代替实际平台vGPU调度、现场摄像机编码、网络抖动、NFS容量/权限和HDFS可达性。'
           '本会话未连接线上任务；部署后需核对实际/config及旧任务显式参数，确保导入当前suanzi.json。'
           '原0.3.0镜像沿用，无需因本轮核验重新上传。', '']
    (OUT/'README.md').write_text('\n'.join(md))
    body = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>平台复核 · 2026-10-29</title>',
        '<style>body{font:16px system-ui,sans-serif;max-width:1150px;margin:32px auto;padding:0 20px;color:#22354a}video{width:100%;max-height:640px;background:#111}a{color:#2264ab}article{border-top:1px solid #ccd6df;padding-top:20px;margin-top:28px}p{line-height:1.7}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse;width:100%}</style>',
        '<h1>当前平台包 · 完整功能与实时复核</h1>',
        '<p>对应10月27日完整先验正式包。原镜像、三GPU实时推流，外部接收录像。模型和阈值未改；展示实际误报/漏报。</p>',
        '<p><a href="README.md">完整报告与限制</a> · <a href="live/effective-config.json">实际参数</a> · <a href="../full-prior-release-20261027/index.html">原报告结果</a></p>',
        '<h2>15分钟连续真实流输出</h2><video id="stream" controls preload="metadata" src="live/full-output-15min.mp4"></video>',
        '<p><a href="live/full-output-15min.mp4">下载15分钟展示版</a> · <a href="live/full-output-stream.mp4">保留黑色预热和尾部的完整接收版</a> · <a href="live/clips-index.html">全部实时动作片段</a></p>',
        f'<p>实际处理{live["main_frames_processed"]}帧；保存{clips["clips"]}段，{clips["complete"]}段前后文完整。片段根据最终人物动作保存，完整输出与片段显示时刻可能因异步回绘缓冲不同而略有差异。</p>',
        '<h2>23段同输入复跑</h2>']
    for r in samples:
        name=html.escape(r['id']);title=html.escape(r['title']);base='demos/'+name
        body += [f'<article id="{name}"><h3>{title}</h3>',
                 f'<p>帧数{r["frames"]}；峰值检测人数{r["people_max"]}；人物窗口{r["events"]}；最终标签 {html.escape(str(r["labels"]))}</p>',
                 f'<video controls preload="none" poster="{base}/preview.jpg" src="{base}/output.mp4"></video>',
                 f'<p><a href="{base}/events.json">事件及先验证据</a> · <a href="../../{html.escape(r["input"])}">输入视频</a></p><details><summary>本样本保存片段</summary>']
        for path in r['clips']:
            body.append(f'<p><a href="{html.escape(path)}">{html.escape(Path(path).name)}</a></p>')
        body.append('</details></article>')
    body.append('</html>');(OUT/'index.html').write_text('\n'.join(body))
    clip_page = body[:2] + ['<h1>15分钟流：按人物与动作保存的全部片段</h1><p><a href="../index.html">返回完整视频</a></p>']
    for c in sorted(clips['records'], key=lambda x:x['start']):
        path=html.escape('../'+c['path'])
        clip_page += [f'<article><h3>T{c["person_id"]} · {html.escape(c["action"])}</h3>',
                      f'<p>源时间 {c["start"]:.2f}–{c["end"]:.2f}s；前{c["pre"]:.2f}s / 后{c["post"]:.2f}s；上下文完整：{c["complete"]}；{html.escape(c["closed_because"])}</p>',
                      f'<video controls preload="none" src="{path}"></video><p><a href="{html.escape("../"+c["metadata"])}">人物、动作和时间元数据</a></p></article>']
    clip_page.append('</html>');(OUT/'live/clips-index.html').write_text('\n'.join(clip_page))
    result = dict(passed=True, scope='Finite functional and streaming regression; long-session metadata risk remains open', package=package, cpu=cpu, live=dict(seconds=900, frames=live['main_frames_processed'],
        fps=live['main_frames_processed']/900, clips=clips['clips'], clips_complete=clips['complete'],
        write_errors=live['final_health_clip_counters']['write_errors']), fault_seconds=110,
        demos=demo, production_modified=False, onsite_platform_tested=False,
        long_session_metadata_accumulates=True, unlimited_stream_stability_accepted=False)
    (OUT/'final-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
