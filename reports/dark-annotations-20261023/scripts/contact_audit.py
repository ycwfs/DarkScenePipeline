"""Compact coverage sheets for visual audit; one actual mid-dark frame per source."""
import json
import argparse
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

from scan import OUT
from build_review import gamma, overlay, FONT


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new-only',action='store_true')
    args=parser.parse_args()
    directory=OUT/'audit';directory.mkdir(exist_ok=True)
    paths=list((OUT/'data').glob('*/annotations.json'))
    sources=[json.loads(p.read_text()) for p in paths]
    sources.sort(key=lambda d:d['video']['source'])
    if args.new_only:
        seen={r['video_id'] for r in json.loads((OUT/'assistant-visual-audit.json').read_text())['frames']}
        sources=[s for s in sources if s['video']['id'] not in seen]
        directory=directory/'new-sources';directory.mkdir(exist_ok=True)
    mapping=[];font=ImageFont.truetype(FONT,14)
    for offset in range(0,len(sources),16):
        canvas=Image.new('RGB',(1600,1032),(244,243,238));painter=ImageDraw.Draw(canvas)
        for slot,data in enumerate(sources[offset:offset+16]):
            f=data['frames'][len(data['frames'])//2];v=data['video'];raw=cv2.imread(str(OUT/f['image']))
            im=Image.fromarray(cv2.cvtColor(overlay(gamma(raw),f,v),cv2.COLOR_BGR2RGB));im.thumbnail((400,225))
            x=slot%4*400;y=slot//4*258
            painter.text((x+5,y+3),f'{offset+slot+1:03d} {Path(v["source"]).name} f{f["frame"]} L={f["luma"]:.1f}',font=font,fill=(25,44,48))
            canvas.paste(im,(x,y+29))
            mapping.append(dict(number=offset+slot+1,video_id=v['id'],source=v['source'],frame=f['frame'],
                image=f['image'],sheet=f'audit/overview-{offset//16+1:02d}.jpg'))
        canvas.save(directory/f'overview-{offset//16+1:02d}.jpg',quality=93)
    (directory/'frames-viewed-index.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2)+'\n')
    print(f'{len(sources)} source midpoints, {(len(sources)+15)//16} sheets')


if __name__=='__main__':main()
