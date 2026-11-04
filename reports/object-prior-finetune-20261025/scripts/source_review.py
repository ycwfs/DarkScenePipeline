"""Freeze all-source overview and prioritize object-rich native ROI review.

Proposals only rank images for inspection; they never become training labels.
The all-source overview also exposes missed/ambiguous objects and missing people.
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1]


def main() -> None:
    drafts = [json.loads(p.read_text()) for p in (OUT / 'drafts').glob('*/proposals.json')]
    assert len(drafts) == 535, 'Wait for complete historical source coverage'
    drafts.sort(key=lambda x: x['video']['source'])
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 14)
    folder = OUT / 'source-overview'
    folder.mkdir(exist_ok=True)
    sources, selected = [], []
    for number, draft in enumerate(drafts):
        source = draft['video']
        full = next(r for r in draft['records'] if r['view'] == 'full')
        sources.append(dict(index=number, video_id=source['id'], source=source['source'],
                            split=draft['split'], frame=full['frame'],
                            image_path=str(OUT / 'drafts' / source['id'] / full['image'])))
        if source['video_label'] in ('drink', 'pick_up'):
            continue
        per_track = {}
        for row in draft['records']:
            if row['view'] == 'full':
                continue
            score = max((p['confidence'] for p in row['proposals']), default=0.)
            if score < .4:
                continue
            old = per_track.get(row['track_id'])
            if old is None or score > old[0]:
                per_track[row['track_id']] = (score, row)
        for score, row in per_track.values():
            selected.append(dict(row, proposal_rank=score, video_id=source['id'],
                source=source['source'], split=draft['split'], group=draft['group'],
                image_path=str(OUT / 'drafts' / source['id'] / row['image'])))
    for start in range(0, len(sources), 16):
        sheet = Image.new('RGB', (2000, 1296), (24, 24, 24))
        draw = ImageDraw.Draw(sheet)
        for j, row in enumerate(sources[start:start+16]):
            im = Image.open(row['image_path']).convert('RGB')
            im.thumbnail((496, 279))
            x, y = j % 4 * 500, j // 4 * 324
            sheet.paste(im, (x + (500-im.width)//2, y+44))
            draw.text((x+4, y+2), f'{row["index"]:03d} {Path(row["source"]).name}', font=font, fill='white')
            draw.text((x+4, y+23), f'{row["split"]} f{row["frame"]}', font=font, fill='#aaaaaa')
        sheet.save(folder / f'page-{start//16:03d}.jpg', quality=94)
    (folder / 'manifest.json').write_text(json.dumps(sources, ensure_ascii=False, indent=2)+'\n')
    selected.sort(key=lambda r: (r['source'], r['frame'], r['view']))
    target = OUT / 'tiles-others'
    target.mkdir(exist_ok=True)
    manifest = target / 'manifest.json'
    if manifest.exists():
        assert [r['id'] for r in json.loads(manifest.read_text())] == [r['id'] for r in selected]
    for start in range(0, len(selected), 12):
        sheet = Image.new('RGB', (1600, 1260), (25, 25, 25))
        draw = ImageDraw.Draw(sheet)
        for j, row in enumerate(selected[start:start+12]):
            im = Image.open(row['image_path']).convert('RGB')
            scale = min(390/im.width, 360/im.height)
            nw, nh = round(im.width*scale), round(im.height*scale)
            left, top = (400-nw)//2, (360-nh)//2
            row['tile_index'] = start+j
            row['tile_transform'] = dict(scale_x=nw/im.width, scale_y=nh/im.height, left=left, top=top)
            x, y = j % 4 * 400, j // 4 * 420
            sheet.paste(im.resize((nw, nh)), (x+left, y+60+top))
            draw.text((x+5, y+2), f'{start+j:04d} '+Path(row['source']).name, font=font, fill='white')
            draw.text((x+5, y+23), f'f{row["frame"]} {row["view"]} {row["split"]}', font=font, fill='white')
            for n in range(0, 400, 50):
                draw.text((x+n+1, y+43), str(n), font=font, fill='#aaaaaa')
            for n in range(0, 360, 50):
                draw.text((x+1, y+60+n), str(n), font=font, fill='#aaaaaa')
        sheet.save(target / f'page-{start//12:03d}.jpg', quality=95)
    manifest.write_text(json.dumps(selected, ensure_ascii=False, indent=2)+'\n')
    print(len(sources), 'source overviews;', len(selected), 'object proposal ranked ROI views')


if __name__ == '__main__':
    main()
