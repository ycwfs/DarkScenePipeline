"""Verify paired model demos and encode before/after comparisons with editable source labels."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT/'reports/person-action-20261023/scripts'))
from person_core import BEHAVIORS


def main() -> None:
    base=OUT/'demo-baseline';new=OUT/'demo';dest=OUT/'comparisons';dest.mkdir(exist_ok=True)
    rows=json.loads((new/'sample-results.json').read_text());reports=[]
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',28)
    header=Image.new('RGB',(1920,54),(24,30,40));draw=ImageDraw.Draw(header)
    draw.text((20,5),'微调前：整帧训练 pn2 → 逐人输入',font=font,fill=(235,235,235))
    draw.text((980,5),'微调后：逐人 ROI 训练 → 逐人输入',font=font,fill=(100,230,190))
    selected=json.loads((OUT/'selection.json').read_text())
    if selected['epoch']==0:
        draw.rectangle((960,0,1919,53),fill=(24,30,40))
        draw.text((980,5),'本轮选择：保留原 pn2（未获得更优候选）',font=font,fill=(100,230,190))
    header.save(dest/'comparison-header.png')
    for row in rows:
        name=row['name'];a=json.loads((base/'events'/f'{name}.json').read_text());b=json.loads((new/'events'/f'{name}.json').read_text())
        assert [(r['track_id'],r['frame_index'],r['crop_sha256'],r['source_frames'])for r in a]==[(r['track_id'],r['frame_index'],r['crop_sha256'],r['source_frames'])for r in b]
        # Actual pose probabilities are also checked: only the RGB checkpoint changes.
        skeleton_max_delta=0.
        for x,y in zip(a,b):
            pa,pb=x['skeleton_probs'],y['skeleton_probs']
            assert (pa is None)==(pb is None)
            if pa is not None:
                assert np.allclose(pa,pb,atol=1e-6,rtol=1e-5)
                skeleton_max_delta=max(skeleton_max_delta,float(np.max(np.abs(np.array(pa)-np.array(pb)))))
        changed=[dict(track_id=x['track_id'],timestamp=x['timestamp'],before=x['label'],after=y['label'])for x,y in zip(a,b)if x['label']!=y['label']]
        target=dest/f'{name}_微调前后.mp4'
        video_a=base/row['video'];video_b=new/row['video']
        filters='[0:v]scale=960:-2,setsar=1[a];[1:v]scale=960:-2,setsar=1[b];[a][b]hstack=inputs=2[pair];[2:v][pair]vstack=inputs=2:shortest=1[out]'
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(video_a),'-i',str(video_b),
            '-loop','1','-framerate',str(row['fps']),'-i',str(dest/'comparison-header.png'),
            '-filter_complex_threads','1','-filter_complex',filters,'-map','[out]','-an','-c:v','libx264','-threads','4',
            '-preset','fast','-crf','20','-pix_fmt','yuv420p','-frames:v',str(row['frames']),
            '-movflags','+faststart',str(target)],check=True)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                '-show_entries','stream=codec_name,width,height,nb_read_frames','-of','json',str(target)]))['streams'][0]
        assert probe['codec_name']=='h264' and int(probe['nb_read_frames'])==row['frames'],probe
        # Decode both standalone outputs fully as well, not just their container headers.
        for path in [video_a,video_b]:
            cap=cv2.VideoCapture(str(path));n=0
            while cap.read()[0]:n+=1
            cap.release();assert n==row['frames'],(path,n)
        report=dict(name=name,predictions=len(b),matched_person_windows=True,skeleton_unchanged=True,
            skeleton_max_absolute_difference=skeleton_max_delta,
            changed_predictions=changed,baseline_labels={k:sum(r['label']==k for r in a)for k in BEHAVIORS},
            selected_labels={k:sum(r['label']==k for r in b)for k in BEHAVIORS},
            comparison=target.relative_to(OUT).as_posix(),output=probe,sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        reports.append(report);print(f'[compare] {name}: {len(changed)}/{len(a)} predictions changed; {probe}',flush=True)
    (OUT/'demo-comparison.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
