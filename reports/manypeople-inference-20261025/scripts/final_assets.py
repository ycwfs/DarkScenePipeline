"""Make an output contact sheet and check all review-page links locally."""
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

from PIL import Image, ImageDraw, ImageFont

REPORT=Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths=[]

    def handle_starttag(self, tag, attrs):
        for key,value in attrs:
            if key in ('href','src','poster') and value:
                self.paths.append(value)


def main() -> None:
    result=json.loads((REPORT/'verification.json').read_text())
    assert result['passed'] and len(result['items'])==9
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    sheet=Image.new('RGB',(1440,906),(240,243,248));draw=ImageDraw.Draw(sheet)
    for i,item in enumerate(result['items']):
        tile=Image.open(item['preview']).convert('RGB').resize((480,270))
        x=(i%3)*480;y=(i//3)*302
        sheet.paste(tile,(x,y))
        draw.text((x+8,y+274),f'{item["number"]:02d} | {item["duration"]/60:.2f} min | inference result',
            font=font,fill=(20,40,60))
    sheet.save(REPORT/'output-contact-sheet.jpg',quality=92)
    parser=Links();parser.feed((REPORT/'index.html').read_text())
    missing=[]
    for value in parser.paths:
        parsed=urlsplit(value)
        if parsed.scheme or not parsed.path:
            continue
        if not (REPORT/unquote(parsed.path)).exists():missing.append(value)
    assert not missing,missing
    (REPORT/'page-verification.json').write_text(json.dumps(dict(local_links_checked=len(parser.paths),
        missing=missing,contact_sheet='output-contact-sheet.jpg'),indent=2)+'\n')
    print('Checked',len(parser.paths),'local links; output sheet ready.')


if __name__=='__main__':
    main()
