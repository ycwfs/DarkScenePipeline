"""Concatenate real, unprocessed sources at original speed; freeze before inference."""
import hashlib
import json
from pathlib import Path
import subprocess
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
SOURCES = [
    'real-actiondata-all/other/other114.mp4', 'real-actiondata/挥手.mp4',
    'real-actiondata-all/喝水16.mp4', 'real-actiondata-all/追逐16.mp4',
    'real-actiondata/捡东西.mp4', 'real-actiondata-all/挥手3.mp4',
    'real-actiondata-all/other/other237.mp4', 'real-actiondata/喝水.mp4',
    'real-actiondata-all/挥手32.mp4', 'real-actiondata-all/捡东西3.mp4',
    'real-actiondata-all/追逐25.mp4', 'real-actiondata/抛物.mp4',
    'real-actiondata-all/喝水35.mp4', 'real-actiondata-all/挥手11.mp4',
    'real-actiondata-all/other/other109.mp4', 'real-actiondata/摔倒.mp4',
    'real-actiondata-all/追逐13.mp4', 'real-actiondata-all/捡东西6.mp4',
    'real-actiondata-all/喝水24.mp4', 'real-actiondata-all/挥手22.mp4',
    'real-actiondata-all/other/other108.mp4', 'real-actiondata-all/追逐33.mp4',
    'real-actiondata-all/捡东西10.mp4', 'real-actiondata-all/喝水41.mp4',
    'real-actiondata-all/挥手45.mp4', 'real-actiondata-all/追逐20.mp4',
    'real-actiondata-all/捡东西26.mp4', 'real-actiondata-all/other/other236.mp4',
    'real-actiondata-all/other/other232.mp4',
]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def main():
    videos = json.loads((ROOT/'reports/person-release-20261024/protocol.json').read_text())['videos']
    lookup = {a:v for v in videos for a in [v['source'], *v['aliases']]}
    parts = OUT/'input-parts'; parts.mkdir(exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 18)
    sheet = Image.new('RGB', (5*384, 6*258), '#edf0f4')
    records = []; frames = 0
    with (OUT/'build-input.log').open('w') as log:
        for i, name in enumerate(SOURCES):
            v = lookup[name]
            assert not v['processed']
            source = ROOT/name
            assert digest(source) == v['sha256']
            target = parts/f'{i:02d}.mp4'
            command = ['ffmpeg','-nostdin','-v','warning','-threads','2','-i',str(source),
                '-an','-vf','scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25',
                '-c:v','libx264','-preset','veryfast','-crf','18','-threads','2',
                '-pix_fmt','yuv420p','-g','25','-bf','0','-video_track_timescale','12800','-y',str(target)]
            subprocess.run(command, check=True, stdout=log, stderr=log)
            cap = cv2.VideoCapture(str(target)); count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, count//2); ok, image = cap.read(); cap.release(); assert ok
            start = frames/25; frames += count
            records.append(dict(index=i+1,source=name,sha256=v['sha256'],label=v['video_label'],
                label_scope='source filename category, not per-person frame truth',
                original_seconds=v['seconds'],original_fps=v['fps'],source_split=v['split'],
                raw_mean_luma=v['raw_mean_luma'],frames=count,start=start,end=frames/25,
                transformed='1920x1080,25fps,original speed,no artificial brightening,no repeats'))
            x,y = (i%5)*384,(i//5)*258
            sheet.paste(Image.fromarray(cv2.cvtColor(cv2.resize(image,(384,216)),cv2.COLOR_BGR2RGB)),(x,y))
            draw=ImageDraw.Draw(sheet)
            draw.text((x+4,y+216),f'{i+1:02d} {Path(name).name} | {start:.2f}s',font=font,fill='#10253f')
            print(i+1,name,count,round(frames/25,2),flush=True)
        (parts/'concat.txt').write_text(''.join(f"file '{i:02d}.mp4'\n" for i in range(len(SOURCES))))
        target=OUT/'input-multiaction-5min.mp4'
        subprocess.run(['ffmpeg','-nostdin','-v','warning','-f','concat','-safe','1','-i',str(parts/'concat.txt'),
            '-c','copy','-movflags','+faststart','-y',str(target)],check=True,stdout=log,stderr=log)
        subprocess.run(['ffmpeg','-nostdin','-v','error','-xerror','-threads','2','-i',str(target),'-f','null','-'],
            check=True,stdout=log,stderr=log)
    cap=cv2.VideoCapture(str(target)); assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==frames;cap.release()
    sheet.save(OUT/'input-contact-sheet.jpg',quality=90)
    manifest=dict(input=target.name,sha256=digest(target),fps=25,frames=frames,seconds=frames/25,
        selection='Frozen before inference; real onsite raw clips for varied actions, brightness and people; not an accuracy benchmark',
        no_repeated_sources=len(set(r['sha256'] for r in records))==len(records),
        caveat='Hard scene cuts are synthetic; all model history/tracking follows the same continuous stream without oracle resets.',
        segments=records)
    (OUT/'input-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('COMPLETE',frames,frames/25,manifest['sha256'],flush=True)

if __name__=='__main__':main()
