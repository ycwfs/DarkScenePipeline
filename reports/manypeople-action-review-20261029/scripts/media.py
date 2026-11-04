"""Make browser-compatible full sources with stream copy, without inference."""
import subprocess
from concurrent.futures import ThreadPoolExecutor

from collect import OUT, read, save, sha


def remux(s):
    target=OUT/'sources'/f'{s["number"]:02d}.mp4'
    if not target.exists():
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',s['absolute_path'],
            '-map','0:v:0','-an','-c:v','copy','-movflags','+faststart',str(target)],check=True)
    probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
        'stream=width,height,nb_frames,start_time,duration,avg_frame_rate','-of','json',str(target)],capture_output=True,text=True,check=True)
    data=__import__('json').loads(probe.stdout)['streams'][0]
    assert int(data['nb_frames'])==s['frames'],(s['number'],data)
    assert abs(float(data.get('start_time',0)))<.05,(s['number'],data)
    assert sha(__import__('pathlib').Path(s['absolute_path']))==s['sha256']
    return dict(number=s['number'],source_sha256=s['sha256'],video=str(target.relative_to(OUT)),sha256=sha(target),probe=data)


if __name__=='__main__':
    (OUT/'sources').mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows=list(pool.map(remux,read(OUT/'sources.json')))
    save('source-media.json',rows)
    print(f'{len(rows)} remuxed sources, frame counts and original hashes verified')
