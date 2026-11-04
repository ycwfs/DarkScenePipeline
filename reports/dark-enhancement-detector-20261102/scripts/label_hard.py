"""AI visual review, before new detector inference; NOT independent human GT."""
import json
from run import OUT, save

protocol = json.loads((OUT / 'hard/protocol.json').read_text())
# Native 1920x1080 xyxy. Original/gamma/cached enhanced views were inspected.
# Fixed foreground robot and the fixed rear wall exhibit are NOT real people.
boxes = [
 [474,370,634,703], [434,352,616,662], [508,313,688,630],
 [1190,595,1380,915], [1060,365,1240,790], [1020,350,1160,770],
 [700,305,880,830], [625,295,880,870], [690,300,875,835],
 [1170,285,1245,445], [1115,235,1210,460], [1115,245,1220,465],
 [940,265,1035,515], [970,260,1065,510], [1045,240,1140,440],
 [475,275,610,680], [618,332,780,715], [753,363,940,838],
 [876,270,963,645], [887,235,1033,580], [911,216,1015,541],
 [970,185,1080,450], [975,210,1080,460], [975,205,1080,465],
]
assert len(boxes) == len(protocol['samples'])
labels = []
for sample, box in zip(protocol['samples'], boxes):
    sample = dict(sample)
    sample['gt_boxes_native'] = [box]
    sample['annotation_status'] = 'AI visually reviewed diagnostic reference; user review pending'
    sample['exhaustive'] = False
    sample['native_image_sha256'] = __import__('hashlib').sha256(
        (OUT / f'hard/native/{sample["id"]}.png').read_bytes()).hexdigest()
    if sample['id'] in [f'H{x:03d}' for x in list(range(4,16))+list(range(22,25))]:
        sample['exhibit_region_native'] = [630,480,890,990]
    else:
        sample['exhibit_region_native'] = None
    labels.append(sample)
save(OUT / 'hard/labels.json', {
    'provenance': 'AI visual review of raw, gamma for visibility, historical enhanced stills; frozen before new inference',
    'limitations': 'approximate visible-person boxes, no independent human confirmation, not exhaustive, no automatic proposals copied to all frames',
    'samples': labels})
print('Frozen', len(labels), 'frames;', len(boxes), 'real-person reference boxes')
