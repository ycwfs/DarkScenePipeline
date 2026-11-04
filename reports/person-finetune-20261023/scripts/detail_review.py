"""Dense temporal contact sheets for long actions and uncertain transitions."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from build_review import OUT, fit


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('video_id')
    parser.add_argument('--track',type=int,default=1)
    args=parser.parse_args()
    matches=list((OUT/'data').glob(args.video_id+'*/manifest.json'))
    assert len(matches)==1
    data=json.loads(matches[0].read_text())
    samples=data['review'][str(args.track)][::2]
    font=ImageFont.truetype('DejaVuSans.ttf',17)
    for page,start in enumerate(range(0,len(samples),48)):
        part=samples[start:start+48]
        canvas=Image.new('RGB',(1200,180*((len(part)+11)//12)),(245,245,245))
        draw=ImageDraw.Draw(canvas)
        for index,sample in enumerate(part):
            x,y=(index%12)*100,(index//12)*180
            tile=fit(Image.open(OUT/sample['image']),98,150)
            canvas.paste(tile,(x,y))
            draw.text((x,y+153),f'{sample["timestamp"]:.2f}s',font=font,fill='black')
        target=OUT/'review'/f'detail-{args.video_id}-T{args.track:02d}-{page:02d}.jpg'
        canvas.save(target,quality=94)
        print(target)


if __name__=='__main__':
    main()
