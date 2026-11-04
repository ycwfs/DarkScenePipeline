"""Inventory existing onsite sources and expose original-resolution interaction crops.

Person boxes and action names are model predictions. This script creates NO object
ground truth and NO training labels; dataset/policy choices are pending discussion.
"""
import hashlib
import html
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'reports/manypeople-inference-20261025'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def fit(image, shape):
    target_w, target_h = shape
    scale = min(target_w / image.shape[1], target_h / image.shape[0])
    resized = cv2.resize(image, (round(image.shape[1]*scale), round(image.shape[0]*scale)))
    canvas = np.full((target_h, target_w, 3), 24, np.uint8)
    h, w = resized.shape[:2]
    canvas[(target_h-h)//2:(target_h-h)//2+h, (target_w-w)//2:(target_w-w)//2+w] = resized
    return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def read_frame(cap, index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, image = cap.read()
    if not ok:
        raise RuntimeError(f'Unable to decode source frame {index}')
    assert abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-index-1) < .51
    return image


def inventory():
    old = json.loads((ROOT / 'reports/person-finetune-20261023/source-inventory.json').read_text())
    splits = json.loads((ROOT / 'reports/person-finetune-20261023/splits.json').read_text())
    assignments = {x['video_id']: x for x in splits['records']}
    records = []
    for source in old['videos']:
        row = {key: source[key] for key in ('id','source','sha256','processed','frames','fps','width','height')}
        row.update(historical_split=assignments[source['id']]['split'],
                   source_group=assignments[source['id']]['group'],
                   aliases=source['aliases'], file_exists=(ROOT / source['source']).is_file(),
                   object_annotations='not started', person_action_truth='use reviewed person intervals only')
        records.append(row)
    for source in json.loads((BASE / 'input-manifest.json').read_text()):
        records.append(dict(id=source['sha256'][:16], source=source['absolute_path'],
                            sha256=source['sha256'], processed=False, frames=source['frames'],
                            fps=source['fps'], width=source['width'], height=source['height'],
                            historical_split=None, proposed_split='feedback_validation_pending_user',
                            source_group=f'onsite-20261025-{source["number"]:02d}',
                            file_exists=Path(source['absolute_path']).is_file(),
                            object_annotations='not started', person_action_truth='unlabeled'))
    assert len({row['sha256'] for row in records}) == len(records)
    assert all(row['file_exists'] for row in records)
    result = dict(status='inventory only; split policy not finalized',
                  unique_sources=len(records), historical_sources=len(old['videos']),
                  new_sources=9, saved_feedback_clips=454,
                  rule='Saved clips are derived from the nine originals, not 454 independent sources. '
                       'Preserve original grouping, prohibit overlapping raw/crop/frame split leakage. '
                       'Historical processed clips have no recoverable raw pixels.',
                  records=records)
    (OUT / 'source-inventory.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')


def main():
    inventory()
    font = ImageFont.truetype(FONT, 20)
    candidates = json.loads((OUT / 'review-candidates.json').read_text())
    visual = OUT / 'visual'
    visual.mkdir(exist_ok=True)
    cards = []
    decoded = 0
    for index, row in enumerate(candidates, 1):
        event = row['event']
        frames = sorted(set((event['source_frames'][0], event['source_frames'][16], event['frame_index'])))
        cap = cv2.VideoCapture(row['source'])
        images = [read_frame(cap, frame_index) for frame_index in frames]
        decoded += len(images)
        cap.release()
        h, w = images[-1].shape[:2]
        # Production inference bbox coordinates belong to the 720 x 404 input.
        scale = np.array([w/720, h/404, w/720, h/404])
        box = np.asarray(event['bbox']) * scale
        margin = np.tile((box[2:]-box[:2])*.3, 2)
        expanded = box + margin * [-1,-1,1,1]
        lo = np.floor(np.maximum([0,0], expanded[:2])).astype(int)
        hi = np.ceil(np.minimum([w,h], expanded[2:])).astype(int)
        assert np.all(hi > lo)
        sheet = Image.new('RGB', (1260, 510), (24,24,24))
        thumb = images[-1].copy()
        b = np.rint(box).astype(int)
        cv2.rectangle(thumb, tuple(b[:2]), tuple(b[2:]), (255,220,0), 3)
        sheet.paste(fit(thumb, (600,338)), (0,85))
        draw = ImageDraw.Draw(sheet)
        title = (f'{index:02d}  原片{row["number"]:02d} T{event["track_id"]}  '
                 f'{event["timestamp"]:.2f}s  模型预测 {event["label"]} {event["confidence"]:.4f}')
        draw.text((12,10), title, font=font, fill='white')
        draw.text((12,42), '原始1080p像素裁剪，未做物体检测/标注；蓝框为待核对的人体预测', font=font, fill='white')
        for j, (frame_index, image) in enumerate(zip(frames, images)):
            crop = image[lo[1]:hi[1],lo[0]:hi[0]]
            cv2.imwrite(str(visual / f'{index:02d}_f{frame_index:06d}_roi.png'), crop)
            sheet.paste(fit(crop, (220,338)), (600+j*220,85))
            draw.text((602+j*220,430), f'帧 {frame_index}', font=font, fill='white')
        filename = f'{index:02d}_source{row["number"]:02d}_{event["label"]}.jpg'
        sheet.save(visual / filename, quality=95)
        row['review_image'] = f'visual/{filename}'
        row['crop_xyxy_raw'] = [*map(int, lo), *map(int, hi)]
        row['review_source_frames'] = frames
        row['roi_box_note'] = 'end-of-window person bbox held fixed for context; not per-frame object GT'
        cards.append(f'<article><h2>{html.escape(title)}</h2><a href="visual/{filename}">'
                     f'<img loading="lazy" src="visual/{filename}" alt="{html.escape(title)}"></a></article>')
    (OUT / 'review-candidates.json').write_text(json.dumps(candidates, ensure_ascii=False, indent=2)+'\n')
    (OUT / 'review.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<title>现场物体先验：待核对原始ROI</title><style>body{font-family:sans-serif;max-width:1280px;'
        'margin:24px auto;background:#eee}article{background:white;padding:12px;margin:24px 0}'
        'img{width:100%}h2{font-size:18px}</style><h1>现场物体先验：25组待核对原始ROI</h1>'
        '<p>从9段录像的喝水/捡物预测中取高分且位置或时间分散的候选。选择是诊断性抽样，'
        '不代表总体误报率；人体框与动作均为模型预测，不是物体真值。原片保留完整像素，'
        '右侧为窗口首/中/末帧固定区域，未用动作模型400×224缓存放大。</p>'+''.join(cards)+'</html>')
    (OUT / 'preparation-verification.json').write_text(json.dumps(dict(
        sources_found=544, reviewed_candidates_prepared=len(candidates), original_frames_decoded=decoded,
        object_ground_truth_created=0, training_started=False, original_media_modified=False), indent=2)+'\n')
    print(f'Inventoried 544 sources; prepared {len(candidates)} ROI sequences ({decoded} source frames).')


if __name__ == '__main__':
    main()
