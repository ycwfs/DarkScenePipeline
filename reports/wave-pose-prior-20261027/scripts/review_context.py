"""Raw temporal context for review, spanning every window in each wave group."""
import argparse
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw

from common import OUT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    cases = json.loads((OUT / ('partial-wave-cases.json' if args.partial else 'wave-cases.json')).read_text())
    for case in cases:
        if case['scope'] not in ('long-videos', 'wave-shorts'):
            continue
        source = case['source']
        cap = cv2.VideoCapture(source['absolute_path'])
        assert cap.isOpened()
        width, height = int(cap.get(3)), int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        ratio = 720 / max(width, height)
        mw, mh = int(round(width*ratio))//2*2, int(round(height*ratio))//2*2
        rows = [json.loads(line) for line in (OUT/case['directory']/'baseline-events.jsonl').read_text().splitlines()]
        rows = [r for r in rows if r['track_id'] == case['track_id'] and r['history_generation'] == case['history_generation']
                and case['start']-2 <= r['timestamp'] <= case['end']+1]
        boxes = np.asarray([r['bbox'] for r in rows]) * [width/mw, height/mh, width/mw, height/mh]
        lower, upper = boxes[:, :2].min(axis=0), boxes[:, 2:].max(axis=0)
        size = upper-lower
        lower = np.maximum(0, np.floor(lower-.25*size)).astype(int)
        upper = np.minimum([width, height], np.ceil(upper+.25*size)).astype(int)
        stamps = np.linspace(max(0, case['start']-1.5), min(duration-1/fps, case['end']+1), 16)
        sheet = Image.new('RGB', (1440, 1460), '#18212a')
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 6), case['id']+' | RAW CONTEXT, fixed union ROI; no inferred joints', fill='white')
        for i, stamp in enumerate(stamps):
            cap.set(cv2.CAP_PROP_POS_FRAMES, round(stamp*fps))
            ok, frame = cap.read()
            assert ok
            crop = frame[lower[1]:upper[1], lower[0]:upper[0]]
            image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            image.thumbnail((355, 330))
            x, y = i%4*360, i//4*355+30
            sheet.paste(image, (x, y+20))
            draw.text((x+4, y), f'{stamp:.2f}s', fill='white')
        cap.release()
        sheet.save(OUT/'wave-review'/f'{case["id"]}-context.jpg', quality=95)


if __name__ == '__main__':
    main()
