"""Real field-video brightness atlas: raw pixels and fixed enhancement, no synthetic dimming."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
SCAN = ROOT / 'reports/dark-annotations-20261023'
sys.path.insert(0, str(ROOT / 'platform/dark-demo-v2'))
from darkpipe.stages.enhance_retinexformer import RetinexformerStage
from darkpipe.stages.recognize import frame_luma

BINS = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 256)]
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def save(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def downscale(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(720 / max(h, w), 1)
    return cv2.resize(image, (round(w * scale) // 2 * 2, round(h * scale) // 2 * 2), interpolation=cv2.INTER_AREA)


def panel(raw: np.ndarray, enhanced: np.ndarray, sample_id: str, source: str,
          frame: int, timestamp: float, luma: float, mean_luma: float) -> np.ndarray:
    canvas = np.full((498, 1440, 3), 22, np.uint8)
    for i, im in enumerate((raw, enhanced)):
        h, w = im.shape[:2]
        scale = min(720/w, 404/h)
        im = cv2.resize(im, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_AREA)
        x = i*720 + (720-im.shape[1])//2
        y = 42 + (404-im.shape[0])//2
        canvas[y:y+im.shape[0], x:x+im.shape[1]] = im
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    font = ImageFont.truetype(FONT, 22)
    d.text((14, 6), f'原图 · 当前帧 L={luma:.2f}', font=font, fill='#F2EDE5')
    d.text((734, 6), 'Retinexformer增强 · 与左侧同一帧', font=font, fill='#F2EDE5')
    d.text((14, 456), f'{sample_id}  {Path(source).name}  |  F{frame:04d}  {timestamp:.2f}s  |  片段原图均值 {mean_luma:.2f}',
           font=font, fill='#E6DCCB')
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def main() -> None:
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    inventory = json.loads((SCAN/'inventory.json').read_text())
    overview = []
    for lo, hi in BINS:
        overview.append(dict(lo=lo, hi=hi, label=f'{lo}≤L<{hi}' if hi<256 else 'L≥75', frames=0, sources=0, train_sources=0))
    ranges = []
    for v in inventory['raw_videos']:
        scan = json.loads((SCAN/'data'/v['id']/'scan.json').read_text())
        values = np.array([f['luma'] for f in scan['frames']])
        ranges.extend(values.tolist())
        for b in overview:
            count = int(((values >= b['lo']) & (values < b['hi'])).sum())
            b['frames'] += count
            b['sources'] += int(count > 0)
            b['train_sources'] += int(count > 0 and v['frozen_split'].get('split') == 'train')
    values = np.array(ranges)
    histogram = [dict(lo=i, hi=i+5, frames=int(((values>=i)&(values<i+5)).sum())) for i in range(0, 160, 5)]
    selected = json.loads((OUT/'selected-sources.json').read_text())['records']
    model_path = ROOT/'platform/dark-demo-v2/ckpts/NTIRE.pth'
    protocol = dict(bins=overview, histogram_5=histogram, source_count=len(inventory['raw_videos']),
        all_source_frames=len(ranges), luma_range=[float(values.min()), float(values.max())],
        selection='18 visually checked real-actiondata-all train videos; 3 per bin, chosen for brightness spread and varied scenes; no test sample selection',
        grouping='Poster raw-frame L determines bin; clip mean/min/max and crossing are displayed separately',
        preprocessing='Decode original -> longest side720, even dimensions -> unenhanced BT.601 BGR mean; no crop, gamma, denoise or synthetic exposure',
        brightness='L=0.114*mean(B)+0.587*mean(G)+0.299*mean(R), 0..255 image intensity, not lux',
        enhancer=dict(path=str(model_path.relative_to(ROOT)), sha256=sha(model_path)),
        clips='Up to4s around source midpoint, original fps and consecutive frames; lossless PNG references, H264 preview crf18',
        no_detector_or_pose=True, purpose='Visual brightness selection only; not action accuracy evaluation',
        production_lum_gate=75, production_modified=False,
        excluded_processed=len(inventory['excluded_processed']))
    save(OUT/'protocol.json', protocol)
    enhancer = RetinexformerStage(str(model_path.parent), chunk=1)
    enhancer.load('cuda:0')
    completed = []
    for n, v in enumerate(selected):
        assert v['frozen_split']['split'] == 'train' and not v['processed']
        source = ROOT/v['source']
        assert sha(source) == v['sha256']
        scan = json.loads((SCAN/'data'/v['id']/'scan.json').read_text())
        count = len(scan['frames'])
        middle = count//2
        length = min(count, round(v['fps']*4))
        first = max(0, min(count-length, middle-length//2))
        last = first+length
        lumas = [r['luma'] for r in scan['frames'][first:last]]
        poster_luma = scan['frames'][middle]['luma']
        lo, hi = BINS[v['bin_index']]
        assert lo <= poster_luma < hi, (v['source'], poster_luma)
        dest = OUT/'samples'/v['sample_id']
        dest.mkdir(parents=True, exist_ok=True)
        record_path = dest/'record.json'
        if record_path.exists():
            record = json.loads(record_path.read_text())
            assert record['source_sha256'] == v['sha256'] and record['enhancer_sha256'] == protocol['enhancer']['sha256']
            completed.append(record)
            continue
        cap = cv2.VideoCapture(str(source))
        dest_video = dest/'comparison.mp4'
        encoder = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', '1440x498', '-r', str(v['fps']), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
            '-pix_fmt', 'yuv420p', '-threads', '2', '-movflags', '+faststart', str(dest_video)], stdin=subprocess.PIPE)
        records = []
        frame = 0
        try:
            while frame < last:
                ok, native = cap.read()
                assert ok, (v['source'], frame)
                if frame >= first:
                    raw = downscale(native)
                    value = frame_luma(raw)
                    assert abs(value-scan['frames'][frame]['luma']) < 1e-8
                    enhanced = enhancer([raw])[0]
                    display = panel(raw, enhanced, v['sample_id'], v['source'], frame, frame/v['fps'], value, float(np.mean(lumas)))
                    encoder.stdin.write(display.tobytes())
                    records.append(dict(frame=frame, source_time=frame/v['fps'], video_time=(frame-first)/v['fps'], luma=value))
                    if frame == middle:
                        cv2.imwrite(str(dest/'raw.png'), raw)
                        cv2.imwrite(str(dest/'enhanced.png'), enhanced)
                        cv2.imwrite(str(dest/'comparison.jpg'), display, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frame += 1
        finally:
            cap.release()
            encoder.stdin.close()
            if encoder.wait(timeout=60):
                raise RuntimeError(f'Encoding failed: {dest_video}')
        decoded = cv2.VideoCapture(str(dest_video))
        decoded_count = 0
        while decoded.read()[0]:
            decoded_count += 1
        assert abs(decoded.get(cv2.CAP_PROP_FPS)-v['fps']) < .01
        decoded.release()
        assert decoded_count == length == len(records)
        check = subprocess.run(['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(dest_video), '-f', 'null', '-'], capture_output=True, check=True)
        assert not check.stderr, check.stderr
        crossings = [threshold for threshold in (15,30,45,60,75) if min(lumas) < threshold <= max(lumas)]
        notes = []
        if v['sample_id'] == '1-1':
            notes.append('原片自带左右黑边；黑边计入全画面平均值。没有人为压暗或额外裁剪。')
        if v['sample_id'] == '2-3':
            notes.append('Other场景，主要用于比较暗画面背景亮度。')
        if v['sample_id'] == '5-3':
            notes.append('人物被展台遮挡；较高的全画面L不等于人体清晰。')
        record = dict(sample_id=v['sample_id'], bin_index=v['bin_index'], source=v['source'], video_id=v['id'],
            source_sha256=v['sha256'], enhancer_sha256=protocol['enhancer']['sha256'], split='train',
            group=v['frozen_split']['group'], original_frames=count, frame_start=first, frame_end_exclusive=last,
            fps=v['fps'], clip_frames=length, seconds=length/v['fps'], poster_frame=middle, poster_luma=poster_luma,
            mean_luma=float(np.mean(lumas)), min_luma=min(lumas), max_luma=max(lumas), crossing_thresholds=crossings,
            source_start_seconds=first/v['fps'], source_end_seconds=last/v['fps'], notes=notes,
            frames=records, full_decode=True,
            files={key:str((dest/name).relative_to(OUT)) for key,name in [('raw','raw.png'),('enhanced','enhanced.png'),
                ('poster','comparison.jpg'),('video','comparison.mp4')]},
            file_sha256={name:sha(dest/name) for name in ['raw.png','enhanced.png','comparison.jpg','comparison.mp4']})
        save(record_path, record)
        completed.append(record)
        save(OUT/'samples.json', dict(samples=completed))
        print(f'[{n+1}/18] {v["sample_id"]} {v["source"]}: L={poster_luma:.2f}, mean={np.mean(lumas):.2f}, {length}frames', flush=True)
    enhancer.close()
    save(OUT/'samples.json', dict(samples=completed))
    print('All samples completed:', len(completed), sum(r['clip_frames'] for r in completed), 'frames')


if __name__ == '__main__':
    main()
