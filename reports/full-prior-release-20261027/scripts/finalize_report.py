"""Finalize report and clip links from recorded evidence, without rerunning inference."""
from pathlib import Path
from collections import Counter
import html,json
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
results=read(OUT/'demo-results.json');stream=read(OUT/'live-verification.json');four=read(OUT/'live-4gpu-verification.json')
rows=[]
for p in (OUT/'live/clips').rglob('*.json'):
 if p.with_suffix('.mp4').exists():rows.append((p,read(p)))
rows.sort(key=lambda x:x[1]['start_source_time'])
body=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>实时保存片段</title><h1>5分钟真实流 · 自动保存29段</h1><p><a href="../index.html">演示索引</a> · <a href="full-output-5min.mp4">完整推流接收录像</a>。按人物＋动作独立保存完整画面，前后各2秒，时间可重叠。上下文完整不等于完整真实动作的召回保证。</p>']
for p,m in rows:
 rel=str(p.with_suffix('.mp4').relative_to(OUT/'live'));meta=str(p.relative_to(OUT/'live'))
 body.append(f'<p><a href="{html.escape(rel)}">T{m["person_id"]} · {html.escape(m["action_key"])} · {m["duration_seconds"]:.2f}秒</a>；上下文完整：{m["context_complete"]} · <a href="{html.escape(meta)}">元数据</a></p>')
body.append('</html>');(OUT/'live/clips-index.html').write_text('\n'.join(body))
trim=read(OUT/'stream-trim.json');trim['output']='live/full-output-5min.mp4';save(OUT/'stream-trim.json',trim)
p=OUT/'scripts/trim_stream.py';s=p.read_text().replace('output=out.name','output=str(out.relative_to(OUT))');p.write_text(s)
p=OUT/'README.md';s=p.read_text().replace('共4607帧、2124个事件、19个动作片段','共5207帧、2399个事件、19个动作片段')
s=s.replace('峰值2、3、4、8、13、15、19人','检测人数从单人到峰值19人')
s=s.replace('原因是HLS/RTMP客户端重连，不是算子进程退出；推流自身重启数为0。', '对应健康检查恰逢HLS/RTMP编码器的stdin.write进行中：进程仍存活，阻塞时长显示四舍五入为0.0秒，重启和持续输出空档均为0。这是当前健康检查过于敏感的边界，不能把这些采样称为实际断流，也不能声称健康状态始终为ok。')
s=s.replace('33个健康采样点低于15 FPS', '240个健康采样点中33个低于15 FPS（采样间隔不完全均匀，不等于精确时间占比）')
s=s.replace('片段写入失败、强制释放、未完成、丢帧均为0。', '片段写入失败、强制释放、未完成、写入队列丢帧均为0。输入7500帧中处理5746帧，其余旧帧为维持低延迟而跳过；输出编码上限15 FPS，输出录像帧数不等同于处理吞吐。')
s=s.replace('三卡总体处理FPS', '实际保存29段，其中28段前后上下文完整；另1段不完整有元数据标记。动作结果年龄P95为0.815秒，最大1.240秒，33个待处理动作任务被同人物较新任务替代；不能保证每个动作判断都在1秒内完成。三卡总体处理FPS')
s=s.replace('四卡不是替换三卡正式默认，而是容量诊断。','四卡不是替换三卡正式默认，而是容量诊断。四卡处理1763/3000输入帧，另外帧按低延迟策略跳过；输出编码不会凭空补回未处理帧。两次运行窗口相位/人物轨迹不同，仅作部署容量诊断，不作逐事件精度对照。')
s=s.replace('未发生处理队列coalescing','没有待处理动作任务被更新任务替代')
s=s.replace('推流端断开时标为degraded；','推流端断开时标为degraded（HTTP仍为200，监控需同时读body.status）；')
s=s.replace('完整视频解码检查](demo-video-verification.json)', '完整视频解码检查](demo-video-verification.json)')
s += '''
## 本轮路径覆盖与剩余误报

两个补充挥手诊断片中，累计否决3个缺少抬腕证据的挥手窗口；成人行走片中速度先验否决2个跑动窗口。它们是历史已审核/诊断素材的功能回归，不作独立泛化统计。儿童片保留2个跑动窗口，其中1个速度可靠而通过、1个缺可靠姿态按约定放行。

手机及相机短片各仍有1个喝水触发，低亮片仍可能误跑动，捡物片也可能被识别成跑动；本轮没有隐藏这些结果，也没有降低动作/物体阈值或重新训练。先验降低一部分误报，无法保证零误报；物体证据不足造成的真实喝水漏报也仍存在。历史准确率权衡见 [误过滤诊断](../false-filter-audit-20261027/analysis-report.md)。

## 复现与审计

代码晋升记录 `promotion.json`，上一版归档manifest和替换文件在 `previous-package-manifest.json`、`before/`。使用正式打包器构建TAR/ZIP和Box，`scripts/verify_delivery.py --publish`逐载荷比对并同步默认别名；脚本不修改镜像。`scripts/prepare.py`准备初始21输入和300秒编码流，`add_wave_cases.py`补充2个挥手诊断输入；`demo.py`调用最终归档的真实 `service.run_offline`，观察包装器只记录元数据。`run_live.py`/`run_faults.py`调用最终归档main，`analyze_live.py`/`analyze_faults.py`校验日志及完整视频；`review.py`生成演示页，`finalize_report.py`生成片段索引和更新说明。

记录的采集→输出延迟从OpenCV交付帧之后开始计时，不含相机编码和网络接收缓冲；本轮没有独立源像素时间标记，不能称为相机到播放器端到端延迟。五分钟剪辑仅裁去外部录制中的连续黑场预热/尾部，未删除中间误识别画面、补帧或重新推理；详见 `stream-trim.json`。现场NFS、HDFS上传故障、本机外网络抖动和无限流需要另行部署验收。
'''
p.write_text(s)
summary=dict(demo_inputs=len(results),demo_frames=sum(x['total_frames']for x in results),demo_events=sum(x['events']for x in results),demo_clips=sum(len(x['clips'])for x in results),
 demo_incomplete_context=sum(not c['context_complete']for x in results for c in x['clips']),
 tests=dict(source=114,archive=85,archive_deselected=1),stream_clips=29,fault_clips=10,four_gpu_clips=2,
 all_scenarios_realtime_target_met=False,model_weights_changed=False,image_changed=False,
 notes=['Finite loopback tests passed, high-density throughput below15FPS','health transient write-in-progress may report degraded','same source evaluated historically; demo is not independent accuracy benchmark'])
save(OUT/'final-summary.json',summary)
print(json.dumps(summary,ensure_ascii=False,indent=2))
