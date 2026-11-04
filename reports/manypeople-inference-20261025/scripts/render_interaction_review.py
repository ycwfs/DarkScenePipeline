"""Render compact source-frame contact sheets for interaction predictions."""
import csv
import html
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / 'reports/manypeople-inference-20261025'
AUDIT = ROOT / 'reports/manypeople-interaction-audit-20261026'
OUT = AUDIT / 'review'


def draw_box(image, box, color, label, scale_x, scale_y):
    x1, y1, x2, y2 = [float(v) for v in box]
    p1 = (int(x1 * scale_x), int(y1 * scale_y))
    p2 = (int(x2 * scale_x), int(y2 * scale_y))
    cv2.rectangle(image, p1, p2, color, 2)
    cv2.putText(image, label, (p1[0], max(18, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                .55, color, 2, cv2.LINE_AA)


def frame_at(cap, frame_index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, image = cap.read()
    if not ok:
        raise RuntimeError(f'Cannot read source frame {frame_index}')
    return image


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((REPORT / 'input-manifest.json').read_text())
    samples = json.loads((AUDIT / 'selected-samples.json').read_text())
    # Keep one strongest segment representative per person/action/time segment.
    unique = {}
    for item in samples:
        seg, event = item['segment'], item['event']
        key = (seg['source_number'], seg['track_id'], seg['history_generation'],
               seg['action'], round(seg['start'], 2), round(seg['end'], 2))
        if key not in unique or event['confidence'] > unique[key]['event']['confidence']:
            unique[key] = item
    chosen = {}
    for key, item in unique.items():
        seg = item['segment']
        bucket = (key[0], key[3], bool(seg['roi_supported_windows']))
        score = (seg['windows'], item['event']['confidence'])
        if bucket not in chosen or score > chosen[bucket][0]:
            chosen[bucket] = (score, (key, item))
    selected = [value[1] for _, value in sorted(chosen.items())]
    rows = []
    for i, (key, item) in enumerate(selected, 1):
        seg, event = item['segment'], item['event']
        source = manifest[key[0] - 1]
        source_path = Path(source['absolute_path'])
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open {source_path}')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(source['fps'].split('/')[0]) / float(source['fps'].split('/')[1])
        sx, sy = width / 720.0, height / 404.0
        indices = sorted(set([int(round(t * fps)) for t in np.linspace(seg['start'], seg['end'], min(3, max(1, int(seg['windows']))))]))
        # Include the representative event's actual source frame.
        indices = sorted(set(indices + [int(event['frame_index'])]))
        tiles = []
        det_by_frame = {}
        detections = event.get('compatible_boxes', [])
        if isinstance(detections, str):
            detections = json.loads(detections)
        for det in detections:
            det_by_frame.setdefault(int(det['frame']), []).append(det)
        for frame_index in indices:
            image = frame_at(cap, frame_index)
            image = cv2.resize(image, (720, 404), interpolation=cv2.INTER_AREA)
            # Event person bbox uses coordinates in the 720x404 inference frame.
            person = event['bbox']
            if isinstance(person, str):
                person = json.loads(person)
            draw_box(image, person, (0, 220, 255),
                     f"T{seg['track_id']} {seg['action']} {event['confidence']:.2f}", 1, 1)
            exact = det_by_frame.get(frame_index, [])
            for det in exact:
                box = det['bbox']
                if isinstance(box, str):
                    box = json.loads(box)
                draw_box(image, box, (255, 80, 30),
                         f"{det['class_name']} {det['confidence']:.2f}", 720 / width, 404 / height)
            timestamp = frame_index / fps
            cv2.putText(image, f"source {key[0]:02d}  {timestamp:.2f}s  frame {frame_index}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(image)
        cap.release()
        sheet = np.vstack(tiles)
        filename = f'{i:03d}_src{key[0]:02d}_{seg["action"]}_T{seg["track_id"]}_{seg["start"]:.2f}.jpg'
        out_path = OUT / filename
        cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        rows.append(dict(source=key[0], action=seg['action'], track_id=seg['track_id'],
                         start=seg['start'], end=seg['end'], windows=seg['windows'],
                         confidence=event['confidence'], roi_supported=bool(seg['roi_supported_windows']),
                         object_classes=json.dumps(seg['object_classes'], ensure_ascii=False),
                         sample_frame_count=len(tiles), image=filename,
                         output_video=f'../../manypeople-inference-20261025/videos/{key[0]:02d}_处理后.mp4',
                         source_video=f'../../../data-10-25-manypeople/{source["path"]}'))
    with (OUT / 'samples.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cards = []
    for row in rows:
        cards.append(f'''<article><h2>{row['source']:02d} · {html.escape(row['action'])} · T{row['track_id']} · {row['start']:.2f}–{row['end']:.2f}s</h2>
<p>窗口 {row['windows']} · 最高分 {row['confidence']:.3f} · 单帧ROI支持 {'有' if row['roi_supported'] else '无'} · 物体类 {html.escape(row['object_classes'])}</p>
<img loading="lazy" src="{html.escape(row['image'])}" alt="source frames with event and sampled object detections">
<p><a href="{html.escape(row['output_video'])}">播放对应完整处理视频</a> · <a href="{html.escape(row['source_video'])}">原始视频</a></p></article>''')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Manypeople 喝水与捡物候选人工复核</title><style>
body{font:15px/1.55 system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 18px;background:#f4f6f8;color:#202a33}h1{font-size:24px}article{border-top:1px solid #cbd2d8;padding:14px 0 24px}h2{font-size:18px;margin-bottom:4px}img{width:min(100%,720px);height:auto;display:block;background:#161a1d}p{margin:6px 0}a{color:#075ca8} .note{background:#e7edf2;padding:10px}</style>
<h1>喝水／捡物候选复核</h1><p class="note">模型阈值 0.85。此页从每源、每类别和物体支持状态抽取代表段，不是全量标注。黄框是事件人物框；蓝框仅在物体缓存实际检测的采样帧出现。候选由模型筛出，不是真值。人物框坐标源自720×404推理画面并映射到源画面。逐段查看时应结合前后完整视频，物体存在不等于发生喝水或捡物。</p>
''' + ''.join(cards) + '</html>'
    (OUT / 'index.html').write_text(page)
    print(f'rendered {len(rows)} unique person/action segments to {OUT}')


if __name__ == '__main__':
    main()
