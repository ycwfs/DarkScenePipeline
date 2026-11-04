"""Review images of final labels, separate from immutable training pixels."""
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw
from prepare_bright import OUT
from demos import _cjk_font


def main() -> None:
    rows = json.loads((OUT/'bright-reviewed-labels.json').read_text())['records']
    panels = []
    for row in rows:
        im = cv2.imread(str(OUT/'data/bright'/f'{row["index"]:03d}'/'enhanced.png'))
        for box in row['person_boxes']:
            b = np.rint(box).astype(int)
            cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), (50, 220, 70), 2)
        # Orange crossed outlines are explicit negative-region annotations for this review image only.
        for box in row['known_nonperson_regions']:
            b = np.rint(box).astype(int)
            cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), (0, 120, 240), 1)
            cv2.line(im, tuple(b[:2]), tuple(b[2:]), (0, 120, 240), 2)
        panel = np.full((438, 720, 3), 25, np.uint8)
        panel[34:] = im
        pil = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(pil).text((5, 3), f'{row["index"]:02d} {row["split"]} · 真人{len(row["person_boxes"])} · 橙叉仅背景',
                                 font=_cjk_font(18), fill='white')
        panels.append(np.array(pil))
    for start in range(0, len(panels), 4):
        canvas = np.vstack([np.hstack(panels[start:start+2]), np.hstack(panels[start+2:start+4])])
        Image.fromarray(canvas).save(OUT/'audit'/f'cleaned-{start//4:02d}.jpg')
    print('24 final-label contact sheets written')


if __name__ == '__main__':
    main()
