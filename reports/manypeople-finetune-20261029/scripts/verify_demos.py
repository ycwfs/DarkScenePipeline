"""Strict video decode, event-rule consistency and clip writer checks."""
import csv
import subprocess
from common import *


def main():
    jobs=read(OUT/'demo-jobs.json');videos=[];summaries=[]
    for job in jobs:
        for which in ('baseline','selected'):
            dest=OUT/'demos'/job['id']/which;summary=read(dest/'summary.json');events=read(dest/'events.json')
            config=summary['configuration'];clips=summary['stats']['clips']
            for key in ['clips_failed','write_errors','clips_forced','clips_unfinished','frames_dropped']:
                assert clips.get(key,0)==0,(job['id'],which,key,clips)
            for e in events:
                if e['label']=='other':continue
                assert e['confidence']+1e-6>=config['reco_min_conf']
                for prior in ['object_prior','wave_prior','running_prior']:
                    p=e[prior]
                    if p.get('applied'):
                        assert p.get('label',e['label'])==e['label'] and p.get('supported',True)
            for path in dest.rglob('*.mp4'):
                subprocess.run(['ffmpeg','-v','error','-xerror','-threads','2','-i',str(path),'-f','null','-'],check=True)
                probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=codec_name,nb_read_frames,duration,width,height','-of','json',str(path)],text=True))['streams'][0]
                assert int(probe['nb_read_frames'])>0 and probe['codec_name']=='h264'
                if path.name=='output.mp4':assert int(probe['nb_read_frames'])==summary['stats']['frames']
                videos.append(dict(path=str(path.relative_to(OUT)),**probe))
            summaries.append(dict(id=job['id'],which=which,frames=summary['stats']['frames'],events=len(events),clips=clips,
                labels=summary['labels']))
    save(OUT/'demo-verification.json',dict(videos=videos,runs=summaries,strict_full_decode=True,
        finite_offline_service=True,live_stream_retested=False,whole_frame_accuracy_claim=False,
        scope='Models, fusion, all priors, per-person overlays, event publication and per-person clip writes exercised through shared production service. Short clip boundary context can be incomplete.'))
    with (OUT/'DEMO_VIDEOS.md').open('w')as f:
        f.write('# 原方案与本轮候选的实际管线输出\n\n每个样本目录提供baseline/output.mp4和selected/output.mp4，直接下载用本机播放器查看。未开启HTTP服务。新片为原始帧FFV1无损摘录，人物ID在新输入起点重新计数，不能与长片的T编号直接相等。\n\n')
        for job in jobs:
            f.write(f'- {job["id"]}：原方案 [MP4](demos/{job["id"]}/baseline/output.mp4)，候选 [MP4](demos/{job["id"]}/selected/output.mp4)；片段分别在对应clips目录。\n')
        f.write('\n这是有限离线文件的完整算法功能检查，不是新的实时推流/平台NFS压力测试，也没有把整幅画面所有人补成完整动作真值。\n')
    print('VERIFIED',len(videos),'videos',len(summaries),'runs',flush=True)


if __name__=='__main__':main()
