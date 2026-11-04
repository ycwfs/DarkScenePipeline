"""Materialize visually reviewed, source-grouped hard negatives; never use predictions as GT."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
OLD = ROOT / 'reports/object-prior-finetune-20261025'
CACHE = ROOT / 'reports/object-model-ablation-20261026'
NAMES = ['bottle', 'cup', 'phone', 'camera', 'paper', 'portable_object', 'robot_or_exhibit', 'microphone']


def save(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


# Coordinates refer to the explicitly inspected 400x400 review tiles. The optional
# valid rectangle removes uncertain/unreviewed edge objects, not model false positives.
# Only visible extents are boxed; an occluded segment between visible ends is included.
LABELS = {
    1: ([120, 75, 280, 355], [(0, 219, 118, 233, 146)]),
    4: ([170, 90, 270, 355], [(2, 228, 155, 256, 216)]),
    5: ([180, 65, 326, 370], [(5, 227, 97, 241, 119)]),
    6: (None, [(5, 172, 114, 185, 144)]),
    11: ([205, 110, 399, 321], [(5, 303, 169, 352, 226)]),
    13: ([185, 135, 315, 383], [(2, 233, 201, 270, 245)]),
    14: ([185, 135, 315, 383], [(2, 242, 201, 289, 239)]),
    15: ([185, 135, 315, 383], [(2, 250, 195, 290, 237)]),
    16: ([170, 110, 320, 385], [(2, 232, 153, 277, 212)]),
    17: (None, [(2, 52, 110, 189, 224)]),
    18: ([165, 110, 366, 390], [(2, 264, 186, 353, 261)]),
    19: ([155, 115, 320, 390], [(2, 270, 210, 305, 255)]),
    22: (None, [(5, 197, 256, 231, 306)]),
    25: ([120, 45, 258, 326], [(6, 140, 109, 251, 320)]),
    28: ([143, 100, 246, 287], []),
    29: ([95, 112, 303, 390], [(2, 113, 317, 129, 351)]),
    30: ([120, 45, 258, 326], [(6, 142, 110, 251, 317)]),
    31: ([135, 90, 276, 300], []),
    32: ([133, 90, 269, 293], []),
    36: ([120, 45, 258, 326], [(6, 140, 112, 252, 320)]),
    39: ([140, 105, 265, 320], []),
    41: (None, []),
    42: (None, []),
    44: ([108, 90, 166, 280], [(5, 126, 171, 145, 194)]),
    45: ([185, 100, 255, 350], [(5, 218, 163, 233, 184), (5, 238, 170, 249, 183)]),
    46: (None, [(5, 284, 193, 328, 231)]),
    49: ([140, 82, 266, 327], []),
    51: ([145, 113, 322, 388], []),
    56: ([135, 87, 240, 345], [(2, 166, 128, 180, 149)]),
    57: (None, []),
    58: (None, [(7, 142, 174, 188, 357)]),
    59: ([145, 132, 300, 391], [(7, 192, 202, 220, 273)]),
    60: ([122, 126, 279, 390], [(7, 146, 180, 170, 247)]),
    61: ([210, 151, 355, 316], [(2, 294, 256, 325, 286)]),
    62: ([120, 106, 277, 380], []),
    63: ([120, 110, 277, 380], []),
    64: (None, []),
    65: ([123, 108, 275, 380], []),
    66: ([123, 108, 275, 380], []),
    67: ([109, 52, 333, 399], [(7, 172, 243, 197, 317)]),
    68: ([152, 125, 251, 376], [(7, 175, 191, 190, 235)]),
    69: ([153, 126, 251, 381], [(7, 175, 193, 191, 235)]),
    70: ([154, 126, 251, 379], [(7, 170, 191, 185, 230)]),
    71: ([155, 81, 338, 391], [(7, 199, 231, 226, 338)]),
    72: ([154, 126, 251, 378], [(7, 172, 184, 187, 223)]),
    73: ([154, 126, 251, 377], [(7, 174, 188, 188, 227)]),
    78: ([73, 101, 335, 303], [(2, 151, 269, 195, 299)]),
    79: ([200, 40, 279, 103], [(2, 235, 83, 258, 93)]),
    80: (None, []),
    81: ([197, 40, 277, 102], [(2, 225, 83, 246, 93)]),
    82: ([152, 108, 260, 375], [(2, 216, 141, 226, 168)]),
    83: ([130, 231, 193, 364], [(2, 145, 280, 160, 313)]),
    84: ([160, 130, 244, 337], []),
    85: ([162, 102, 234, 344], []),
    86: ([144, 115, 261, 399], []),
    88: (None, []),
    89: ([201, 183, 271, 386], [(0, 222, 278, 250, 307)]),
    91: ([150, 117, 280, 345], [(0, 257, 227, 271, 251)]),
    94: ([147, 102, 251, 341], []),
    95: ([117, 69, 303, 366], []),
    96: ([151, 108, 237, 337], [(2, 189, 166, 201, 184)]),
    97: (None, []),
    98: ([145, 92, 250, 334], []),
    99: ([157, 112, 256, 340], []),
    100: ([169, 111, 247, 339], [(2, 218, 177, 232, 196)]),
    101: ([142, 115, 253, 235], [(2, 215, 192, 235, 208)]),
    102: ([160, 113, 288, 357], []),
    104: ([171, 90, 265, 241], []),
    105: ([126, 113, 274, 353], [(2, 136, 145, 151, 167)]),
    106: ([128, 115, 276, 350], [(2, 149, 150, 164, 173)]),
    107: ([130, 110, 252, 390], []),
    110: (None, [(2, 184, 139, 207, 158)]),
    111: ([1, 98, 90, 340], [(7, 6, 210, 66, 294)]),
    112: ([127, 115, 286, 376], [(0, 174, 162, 207, 191)]),
    114: (None, []),
    115: (None, [(3, 159, 189, 239, 279)]),
    116: ([155, 106, 247, 273], [(3, 190, 149, 216, 172)]),
    117: ([145, 110, 253, 330], [(3, 194, 150, 218, 171)]),
    118: ([111, 315, 190, 399], [(2, 119, 348, 149, 380)]),
    120: ([131, 41, 309, 181], []),
    121: ([145, 99, 298, 229], []),
    122: ([145, 110, 297, 246], []),
    123: ([136, 121, 281, 265], []),
    124: ([115, 108, 299, 231], []),
    125: ([141, 114, 296, 254], []),
    126: ([164, 117, 268, 353], [(2, 208, 150, 220, 169)]),
    127: ([75, 65, 324, 250], [(3, 168, 157, 214, 198)]),
    130: ([75, 65, 324, 250], [(3, 168, 158, 214, 199)]),
    133: ([75, 65, 324, 250], [(3, 166, 158, 215, 199)]),
    135: ([68, 76, 291, 236], [(3, 149, 147, 184, 177)]),
    138: ([68, 76, 291, 236], [(3, 159, 146, 191, 173)]),
    141: ([68, 76, 291, 236], [(3, 132, 143, 163, 170)]),
}


def main() -> None:
    target = OUT / 'dataset'
    if target.exists():
        raise FileExistsError('Frozen dataset already exists')
    views = json.loads((OUT/'label-views.json').read_text()) + json.loads((OUT/'extra-views.json').read_text())
    annotations, manifest = [], []
    old = json.loads((OLD/'dataset-manifest.json').read_text())
    for r in old['records']:
        stem = 'old_' + Path(r['dataset_image']).stem
        ip = target/'images'/r['split']/(stem+'.png'); lp = target/'labels'/r['split']/(stem+'.txt')
        ip.parent.mkdir(parents=True, exist_ok=True); lp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(r['dataset_image'], ip); shutil.copy2(r['dataset_label'], lp)
        manifest.append(dict(r, cohort='old', dataset_image=str(ip), dataset_label=str(lp)))
    for r in views:
        idx = r['index']
        if idx not in LABELS:
            annotations.append(dict(index=idx, included=False, reason='uncertain or redundant object extent/category'))
            continue
        region, boxes = LABELS[idx]
        t = r['tile_transform']
        offset = np.array([t['left'], t['top']]*2)
        scale = np.array([t['scale_x'], t['scale_y']]*2)
        native_region = np.array([0, 0, t['width'], t['height']], float) if region is None else (np.array(region)-offset)/scale
        native_region = np.clip(native_region, [0]*4, [t['width'],t['height']]*2)
        global_region = native_region + np.tile(r['rect'][:2], 2)
        global_boxes = [(c, (np.array(b)-offset)/scale + np.tile(r['rect'][:2],2)) for c,*b in boxes]
        annotation = dict(r, included=True, tile_valid_region=region, tile_boxes=boxes,
            reviewed_global_region=global_region.tolist(), reviewed_global_boxes=[dict(class_id=c,bbox=b.tolist())for c,b in global_boxes],
            review='Assistant visual annotation from native pixels and temporal context; auxiliary GT, not expert adjudication')
        annotations.append(annotation)
        variants = ['native_raw', 'small_enhanced'] if r['split']=='train' else ['small_enhanced']
        if 'small_enhanced' not in r['variants']:
            variants = ['native_raw']
        for variant in variants:
            v = r['variants'][variant]; image = cv2.imread(str(CACHE/v['path'])); assert image is not None
            origin, transform = np.array(v['origin']*2), np.array(v['scale']*2)
            bounds = np.array([image.shape[1],image.shape[0]]*2)
            roi = np.clip((global_region-origin)/transform, 0, bounds)
            lo = np.floor(roi[:2]).astype(int); hi = np.ceil(roi[2:]).astype(int)
            crop = image[lo[1]:hi[1],lo[0]:hi[0]]; assert crop.size
            h,w = crop.shape[:2]; final_boxes=[]; lines=[]
            for c, b in global_boxes:
                box = np.clip((b-origin)/transform-np.tile(lo,2), 0, [w,h]*2)
                assert np.all(box[2:]-box[:2] >= 1), (idx, variant, box)
                final_boxes.append(dict(class_id=c,bbox=box.tolist()))
                xywh = np.r_[(box[:2]+box[2:])/2, box[2:]-box[:2]]/np.array([w,h]*2)
                lines.append(f'{c} '+ ' '.join(f'{n:.8f}'for n in xywh))
            stem=f'new_{idx:03}_{variant}'; ip=target/'images'/r['split']/(stem+'.png'); lp=target/'labels'/r['split']/(stem+'.txt')
            ip.parent.mkdir(parents=True,exist_ok=True); lp.parent.mkdir(parents=True,exist_ok=True)
            assert cv2.imwrite(str(ip),crop)
            lp.write_text('\n'.join(lines)+('\n' if lines else ''))
            manifest.append(dict(id=f'new_{idx:03}', index=idx, cohort='new', split=r['split'],
                source=r['source'],group=r.get('group',f'new-{r["number"]:02}'),frame=r['frame'],variant=variant,
                boxes=final_boxes,dataset_image=str(ip),dataset_label=str(lp),
                image_sha256=hashlib.sha256(ip.read_bytes()).hexdigest()))
    groups,hashes,sources = {},{},{}
    for r in manifest:
        assert groups.setdefault(r['group'],r['split'])==r['split']
        assert hashes.setdefault(r['image_sha256'],r['split'])==r['split']
        assert sources.setdefault(str(Path(r['source']).resolve()),r['split'])==r['split']
    stats={s:dict(images=len(rs), unique_views=len({r['id']for r in rs}),groups=len({r['group']for r in rs}),
        boxes=dict(Counter(NAMES[b['class_id']] for r in rs for b in r['boxes'])),backgrounds=sum(not r['boxes']for r in rs))
        for s in ('train','val','test') for rs in [[r for r in manifest if r['split']==s]]}
    save(OUT/'reviewed-labels.json',dict(records=annotations,classes=NAMES))
    save(OUT/'dataset-manifest.json',dict(records=manifest,stats=stats,group_and_pixel_checks=True))
    (OUT/'objects.yaml').write_text(yaml.safe_dump(dict(path=str(target),train='images/train',val='images/val',test='images/test',names=dict(enumerate(NAMES))),sort_keys=False))
    for start in range(0,len(annotations),24):
        included=[r for r in annotations[start:start+24]if r.get('included')]
        sheet=Image.new('RGB',(1600,2400),(24,24,24))
        for j,r in enumerate(included):
            idx=r['index']; path=OUT/'review/tiles'/f'{idx:03}.png' if idx<127 else OUT/'review'/f'extra-tile-{idx-127:02}.png'
            im=Image.open(path).convert('RGB');d=ImageDraw.Draw(im)
            if r['tile_valid_region']:d.rectangle(r['tile_valid_region'],outline='white',width=1)
            for c,*b in r['tile_boxes']:
                d.rectangle(b,outline='#00ff88',width=2);d.text((b[0],max(30,b[1]-12)),NAMES[c],fill='#00ff88',stroke_width=1,stroke_fill='black')
            sheet.paste(im,(j%4*400,j//4*400))
        sheet.save(OUT/'review'/f'verified-{start//24:02}.jpg',quality=95)
    print(json.dumps(stats,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
