"""Render exact-context object comparisons, including unchanged and worsened cases."""
import json
from PIL import Image,ImageDraw
from build_dataset import OUT,NAMES


def main():
    data=json.loads((OUT/'object-context-evaluation.json').read_text())
    target=OUT/'context-comparisons';target.mkdir(exist_ok=True)
    for r in data['records']:
        raw=Image.open(r['image']).convert('RGB');w,h=raw.size;scale=min(600/w,650/h)
        size=(round(w*scale),round(h*scale));canvas=Image.new('RGB',(1200,720),(24,24,24));draw=ImageDraw.Draw(canvas)
        for j,variant in enumerate(('baseline','candidate')):
            im=raw.resize(size);d=ImageDraw.Draw(im)
            for b in r['boxes']:d.rectangle([v*scale for v in b['bbox']],outline='white',width=1)
            for b in r['variants'][variant]['predictions']:
                if b[4]<.581:continue
                rect=[v*scale for v in b[:4]];c=int(b[5]);color='#ff6655'if c in (0,1)else '#40dfff'
                d.rectangle(rect,outline=color,width=2)
                d.text((rect[0],max(0,rect[1]-13)),f'{NAMES[c]} {b[4]:.3f}',fill=color,stroke_width=1,stroke_fill='black')
            canvas.paste(im,(j*600+(600-im.width)//2,45))
            draw.text((j*600+10,8),f'{variant} | src{r["number"]:02} #{r["index"]:03} {r["frame"]/25:.2f}s | {r["split"]}',fill='white')
        draw.text((10,703),'White: reviewed GT. Color: detection >= .581. Original enhanced720 ROI, same pixels in both columns.',fill='white')
        canvas.save(target/f'{r["index"]:03}.jpg',quality=95)


if __name__=='__main__':main()
