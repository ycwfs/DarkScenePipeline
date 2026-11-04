import json, subprocess
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
ROOT=Path(__file__).resolve().parents[1]
rows=json.loads((ROOT/'demo-results.json').read_text())
out=Image.new('RGB',(1200,((len(rows)+2)//3)*280),'white')
for i,row in enumerate(rows):
    p=ROOT/row['video']
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(p),'-f','null','-'],check=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(p)]))['streams'][0]
    assert int(probe['nb_read_frames'])==row['frames'], row
    assert probe['codec_name']=='h264'
    im=Image.open(ROOT/row['poster']);im.thumbnail((398,245))
    x,y=(i%3)*400,(i//3)*280
    out.paste(im,(x,y));ImageDraw.Draw(out).text((x+5,y+246),f"{i+1} L={row['raw_mean_luma']}  candidates={row['people_mode']}",fill='black')
out.save(ROOT/'demo-contact-sheet.jpg')
(ROOT/'demo-validation.json').write_text(json.dumps(dict(videos=len(rows),frames=sum(r['frames'] for r in rows),h264_full_decode=True),indent=2))
