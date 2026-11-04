"""Repaint report captions using a full CJK font; preserves inference records."""
import json
import subprocess

import cv2
import numpy as np
from PIL import Image,ImageDraw

from demos import _cjk_font
from prepare import OUT,save


def main():
    summaries=json.loads((OUT/'demo-summary.json').read_text())['videos']
    checked=[]
    for s in summaries:
        uid=s['id'];dest=OUT/s['video'];marker=OUT/'data'/uid/'title-verification.json'
        if marker.exists():checked.append(json.loads(marker.read_text()));continue
        data=json.loads((OUT/'data'/uid/'demo.json').read_text());temp=dest.with_name(uid+'-titles.mp4')
        cap=cv2.VideoCapture(str(dest));fps=s['fps']
        encoder=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1440x940',
            '-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',
            '-threads','2','-movflags','+faststart',str(temp)],stdin=subprocess.PIPE)
        index=0
        try:
            while True:
                ok,frame=cap.read()
                if not ok:break
                f=data['frames'][index];pil=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(pil)
                for x,y,text in [(0,0,'原模型 · 未平滑'),(720,0,'原模型 · 连续确认＋因果平滑'),
                                 (0,446,'微调候选 · 未平滑'),(720,446,'微调候选 · 连续确认＋因果平滑')]:
                    d.rectangle((x,y,x+720,y+39),fill=(20,25,32));d.text((x+12,y+8),text,font=_cjk_font(20),fill=(235,241,248))
                d.rectangle((0,892,1440,940),fill=(20,25,32))
                text=f'{s["source"].split("/")[-1]} | {s["split"]} | F{index:04d} L={f["luma"]:.1f} | 姿态诊断展示：暗场也显示骨架；运行照度门仍为75'
                d.text((12,900),text,font=_cjk_font(19),fill=(235,241,248))
                frame=cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR);encoder.stdin.write(frame.tobytes())
                if index==s['frames']//2:cv2.imwrite(str(OUT/'posters'/f'{uid}.jpg'),frame)
                index+=1
        finally:
            cap.release();encoder.stdin.close();code=encoder.wait(timeout=60)
        assert code==0 and index==s['frames'],(uid,code,index)
        check=subprocess.run(['ffmpeg','-v','error','-threads','2','-i',str(temp),'-f','null','-'],capture_output=True,check=True)
        assert not check.stderr,check.stderr
        temp.replace(dest)
        row=dict(id=uid,frames=index,full_decode=True,font='NotoSansCJK-Regular.ttc',inference_unchanged=True)
        save(marker,row);checked.append(row);print('[titles]',uid,index,flush=True)
    save(OUT/'video-verification.json',dict(videos=checked,total_frames=sum(r['frames'] for r in checked)))


if __name__=='__main__':main()
