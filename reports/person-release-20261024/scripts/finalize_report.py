from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 names=['stream-two-frontends','final-archive-stream','stream-crowded-three','release-archive-stream']
 rows=[json.loads((OUT/n/'verification.json').read_text())for n in names]
 for r in rows:
  assert r['complete']['normal_stop'] and all(v==0 for v in r['complete']['http_clients'].values())
  for key in ['clips_failed','clips_forced','clips_unfinished','write_errors','frames_dropped']:
   assert r['recording_counters'][key]==0,(r['name'],key)
 final=rows[-1];p=final['decision'];submitted=sum(w['queue']['submitted']for w in p['action_workers'])
 assert p['submitted']==submitted==p['events']+p['coalesced']
 archives=[]
 for src,dst in [(ROOT/'platform/dark-demo-v2_person_20261024.tar.gz',ROOT/'platform/dark-demo-v2.tar.gz'),
                 (ROOT/'platform/dark-demo-v2_person_20261024.zip',ROOT/'platform/dark-demo-v2.zip'),
                 (ROOT/'NVIDIA-Box_20261024.zip',ROOT/'NVIDIA-Box.zip')]:
  manifest=json.loads(src.with_name(src.name+'.manifest.json').read_text())
  assert sha(src)==manifest['sha256']
  # Earlier default bytes remain in the existing dated10-22 archives.
  shutil.copyfile(src,dst)
  alias=dict(manifest,archive=dst.name)
  dst.with_name(dst.name+'.manifest.json').write_text(json.dumps(alias,ensure_ascii=False,indent=2)+'\n')
  dst.with_name(dst.name+'.sha256').write_text(f'{manifest["sha256"]}  {dst.name}\n')
  assert sha(dst)==manifest['sha256']
  archives.append(dict(file=str(src.relative_to(ROOT)),alias=str(dst.relative_to(ROOT)),bytes=src.stat().st_size,sha256=manifest['sha256']))
 image_id=json.loads((OUT/'image-id.json').read_text());image_sha=(OUT/'image-sha256.txt').read_text().split()[0]
 previous=json.loads((ROOT/'reports/feedback-finetune-20261019/release.json').read_text())
 assert image_id=='sha256:d359716620d7eb757cd7c36733e25642469450f1c801e0367fc99248f0a2f212'
 assert image_sha=='a562d53d7bc588aded2d146939a634cf3e0a78afd6dfb8c5f53692c6afec2b77'
 release=dict(version='20261024-person-gate30',defaults=dict(person_mode=True,lum_gate=30,skel_fusion_alpha=.95,reco_min_conf=.91,skel_no_person='soft',clip_pre_sec=2,clip_post_sec=2,clip_denoise='off',gpu_ids='0,1,2'),archives=archives,
 image=dict(id=image_id,archive='platform/darkpipe-operator-0.3.0.tar.gz',sha256=image_sha,unchanged=True,reupload_required=False),
 regression=dict(tests=160,passed=True,log='archive-regression.log'),
 verification=dict(final_archive_stream='release-archive-stream/verification.json',crowded='stream-crowded-three/verification.json',
 box='box-offline-verification.json',box_selfcheck='box-final-selfcheck.log',offline='offline-verification.json',archive='archive-verification.json'),
 demo=dict(videos=13,frames=1240,full_decode=True),flow=dict(editable=True,pages=2,pictures=0),
 caveats=['Finite local loopback tests; not real platform camera/NFS/WebHDFS','Native Box tested on development host, not physical Box','Accuracy still has errors; historical source exposure; incomplete class/luma coverage','Single/dual GPU real-time15FPS not established','Initial temporal history1s is separate from decision latency'])
 (OUT/'release.json').write_text(json.dumps(release,ensure_ascii=False,indent=2)+'\n')
 table='| 验证 | 时长 | 处理FPS中位数 | 处理延迟P95 | 动作延迟P95/最大 | 片段 |\n|---|---:|---:|---:|---:|---:|\n'
 labels=['三卡并行源树故障测试','同算法归档故障测试（统计修正前）','约13人回传负样本三卡','最终归档三卡故障测试']
 for name,r in zip(labels,rows):
  d=r['decision'];table+=f"| {name} | {r['duration']:g}s | {r['processing_fps']['median']:.2f} | {r['processing_latency_ms']['p95']:.0f}ms | {d['decision_age_p95']:.3f}/{d['max_decision_age']:.3f}s | {r['clips_total']} |\n"
 receivers='\n'.join(f"- 最终归档 {r['url'].split(':')[0].upper()} 接收端：匹配帧时延P95={r['capture_to_receiver_seconds']['p95']:.3f}s，实际收到{r['frames']}帧。"for r in final['receivers'])
 contexts='\n'.join(f"- {r['name']}：{r['clips_context_complete']}/{r['clips_total']}单文件前后上下文达标；收尾原因{r['clip_close_reasons']}。"for r in rows if r['clips_total'])
 text=f'''# 实时与保存验收

最终归档在未修改的darkpipe-operator:0.3.0镜像内，挂载独立解包目录运行真实模型。物理3张RTX3090，25FPS真实监控片段循环RTSP输入；前两卡交替前端，第三卡动作。当前未对CPU/内存施加平台配额，所以不能将本机数据保证为平台任意vGPU配置的速度。

{table}

时延均为实测，画面处理延迟是解码收到帧至输出JPEG，动作延迟是识别窗末帧捕获至动作计算完成；首次识别需另外积累约1秒历史。接收端延迟用JPEG顶部帧ID条码匹配，未改变模型输入与保存视频。流默认15FPS，处理FPS高于15不代表每个25FPS源帧都输出；最新帧覆盖是设计行为。稳态汇总排除开始10秒及预设输入故障/恢复5秒，完整最小值、最大值与逐帧数据保留在verification.json；恢复瞬间滚动帧率可能包含断流时间而降低。

{receivers}

故障注入：暂停输入8秒、停止输出中继10秒、杀输入源10秒后重新发布；300秒测试另断开/重连观看端。均正常恢复并SIGTERM退出0、会话stopped；MJPEG/FLV/HLS均由FFmpeg真实接收成功。接收器错误列表包含启动前和关服后的重试，以及人为中继故障期，不能直接把该计数视为正常期中断。输入失联显示同尺寸重连提示，健康检查显式降级；真实写盘/模型异常仍必须失败退出。

拥挤样本原图亮度未知，input_processed=true，跳过重复增强：候选数中位13、峰值16，90秒保存0个动作片段，1817个人物窗口均未触发保存；动作队列合并15个过期同人窗口，保持最近窗口。它验证拥挤识别与推流，不等同于“13人原始暗视频＋增强＋13路同时录像”的综合压力。单/双卡可运行，未据此宣称15FPS达标。

## 片段完整性

各测试clips_failed/clips_forced/clips_unfinished/write_errors/录制frames_dropped均0，所有已保存视频通过FFmpeg完整解码。前后2秒是相对模型触发事件：

{contexts}

不完整原因包括起始缓存不足、输入故障、提前停止及15秒长动作分段；分段文件本身尾部可能不足2秒，但下一重叠段承接。不能把“全部可解码”解释为“每个文件都有完整真实动作起止”。同名JSON逐条记录源时间覆盖、context_complete、收尾原因及掉帧；真实动作漏识不会靠保存器补出来。

原镜像挥手3离线输入160帧，整段160帧1440×808 H.264，动作片段1个159帧，已修复首次实现按离线墙钟误判断流导致短片的问题。Box最终ZIP原生挥手11输入228帧，完整228帧输出并保存4段，不同人物/动作片段重叠3.16秒和1.64秒；全部可解码。这里的喝水/挥手是模型实际事件，只验保存协作，不把它们当正确动作真值。

160项回归通过；包含8人同时保存、同人动作切换保留旧尾、前后2秒、重叠续段、迟到事件、断流、写盘失败、线程清理和前端顺序/队列上限。实时帧队列、动作待办、结果缓存、保存事件队列和近期统计均有边界；现场多人存储过载仍可能丢帧/丢事件，并有计数。源树230秒测试主进程RSS约在后半程2.2GiB附近稳定，不能据有限测试承诺长期无内存增长。

## 交付一致性

平台tar/zip的63份文件载荷一致，Box83份文件（manifest含82条）完整哈希通过；共享runtime/12份平台权重一致。5份活动模型SHA匹配models.json。最终包路径与哈希见release.json，默认别名同步，历史10-22归档保留。Box原生五模型前向和门30自检通过。

镜像ID和镜像压缩包SHA均与10-19交付一致，无新运行依赖，无需重传镜像；平台必须重新导入清单。没有连接真实NFS/HDFS、远端摄像头或实体Box。网络、存储和驱动故障仍可能出现；本次只说明所测恢复流程和保存协议通过，不承诺永不中断。
'''
 (OUT/'stability-report.md').write_text(text)
 print(json.dumps(release,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
