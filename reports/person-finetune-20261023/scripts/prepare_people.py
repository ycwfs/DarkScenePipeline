"""Build person windows with the reviewed demo's exact detection and crop rules."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
DEMO = ROOT/'reports/person-action-20261023/scripts'
sys.path.insert(0, str(DEMO))
from person_core import OP, PersonTracker, associate_poses, person_boxes
from demo import content_height
from darkpipe.media import VideoWriter
from darkpipe.stages.downscale import DownscaleStage
from darkpipe.stages.enhance_retinexformer import RetinexformerStage
from darkpipe.stages.object_prior import YoloOnnxDetector
from darkpipe.stages.skeleton_aux import SkeletonAux, normalize_window, completeness
from darkpipe.stages.recognize import frame_luma, gate_luma


def write_json(path: Path, value: object) -> None:
    pending = path.with_suffix(path.suffix+'.tmp')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    pending.replace(path)


@torch.inference_mode()
def skeleton_probability(window: list, skeleton: SkeletonAux) -> tuple:
    points = np.stack([sample.pose for sample in window])[:, None]
    normalized, presence = normalize_window(points)
    visible = (points[:, 0, :, 2] > .3).sum(1) >= 10
    valid = bool(presence[0] >= .5 and completeness(points)[0] >= .5
                 and visible.mean() >= .5 and points[-1, 0, :, 2].mean() >= .5
                 and visible[-1] and gate_luma([sample.luma for sample in window], 75) is None)
    probs = None
    if valid:
        value = torch.from_numpy(normalized[:1]).to('cuda:0')
        probs = skeleton.net(value).softmax(1).float().cpu().numpy()[0].tolist()
    return probs, valid


def prepare(item: dict, detector: YoloOnnxDetector, skeleton: SkeletonAux,
            enhancer: RetinexformerStage) -> dict:
    dest = OUT/'data'/item['id']
    complete = dest/'manifest.json'
    if complete.exists():
        return json.loads(complete.read_text())
    dest.mkdir(parents=True, exist_ok=True)
    (dest/'windows').mkdir(exist_ok=True)
    (dest/'review').mkdir(exist_ok=True)
    source = ROOT/item['source']
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {source}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    tracker, downscale = PersonTracker(), DownscaleStage(720)
    records, frames, examples = [], [], defaultdict(list)
    times, frame_index, started = {}, 0, time.monotonic()
    review_writer = VideoWriter(str(dest/'review'/'tracks.mp4'), fps=fps)
    try:
        while True:
            ok, raw = cap.read()
            if not ok:
                break
            timestamp = frame_index/fps
            if item['processed']:
                frame = downscale([raw[:content_height(raw.shape[0])]])[0]
                luma = None
            else:
                raw = downscale([raw])[0]
                luma = frame_luma(raw)
                frame = enhancer([raw])[0]
            result = detector([frame])
            if result is None:
                raise RuntimeError(f'Detector failed: {source}@{frame_index}')
            boxes = person_boxes(result[0])
            poses = [None]*len(boxes)
            if len(boxes) and (luma is None or luma >= 75):
                xy, scores = skeleton.pose(frame)
                poses = associate_poses(boxes, xy, scores)
            tracks = tracker.update(boxes, poses, frame, timestamp)
            shown = frame.copy()
            for track in tracks:
                track.append(frame, timestamp, frame_index, luma)
                x1, y1, x2, y2 = np.rint(track.box).astype(int)
                cv2.rectangle(shown, (x1,y1), (x2,y2), (50,220,100), 2)
                cv2.putText(shown, f'T{track.tid:02d}', (x1,max(y1-5,15)),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (50,220,100), 2)
                if timestamp-times.get(track.tid, -1e9) >= .24:
                    # Narrow review crops avoid spending contact-sheet pixels on letterbox padding.
                    crop = cv2.cvtColor(track.history[-1].crop, cv2.COLOR_RGB2BGR)
                    nonpad = np.any(crop != 114, axis=(0,2))
                    columns = np.flatnonzero(nonpad)
                    if len(columns):
                        crop = crop[:,columns[0]:columns[-1]+1]
                    path = dest/'review'/f'T{track.tid:03d}_{frame_index:06d}.jpg'
                    cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    examples[track.tid].append(dict(frame=frame_index,timestamp=timestamp,
                                                    image=path.relative_to(OUT).as_posix()))
                    times[track.tid] = timestamp
                window = track.window(fps)
                if window is None or timestamp-track.last_inference < .5-1e-6:
                    continue
                track.last_inference = timestamp
                uid = f'T{track.tid:03d}_{frame_index:06d}'
                pixels = np.stack([sample.crop for sample in window])
                destination = dest/'windows'/f'{uid}.npz'
                np.savez_compressed(destination, rgb=pixels,
                                    pose=np.stack([sample.pose for sample in window]))
                probs, valid = skeleton_probability(window, skeleton)
                records.append(dict(id=item['id']+'_'+uid, video_id=item['id'],
                    path=destination.relative_to(OUT).as_posix(), track_id=track.tid,
                    start=window[0].timestamp, end=timestamp,
                    source_frames=[sample.frame_index for sample in window],
                    timestamps=[sample.timestamp for sample in window],
                    crop_boxes=[sample.crop_box.tolist() for sample in window],
                    crop_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                    lumas=[sample.luma for sample in window], skeleton_probs=probs,
                    pose_valid=valid, person_label=None))
            cv2.putText(shown, f'{timestamp:.2f}s', (8,22),
                        cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
            review_writer.write(shown)
            frames.append(dict(frame=frame_index, timestamp=timestamp, luma=luma,
                tracks=[dict(track_id=t.tid,bbox=t.box.tolist()) for t in tracks]))
            if frame_index == 0:
                cv2.imwrite(str(dest/'review'/'scene.jpg'), shown)
            frame_index += 1
    finally:
        cap.release()
        review_writer.close()
    if frame_index != item['frames'] or frame_index != review_writer.count:
        raise RuntimeError(f'Incomplete video decode: {source}: {frame_index}/{item["frames"]}')
    manifest = dict(video=item, windows=records, frames=frames, review=examples,
                    elapsed=time.monotonic()-started, full_decode=True,
                    person_labels_reviewed=False)
    write_json(complete, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard', default='0/1')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--ids', default='')
    args = parser.parse_args()
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    videos = json.loads((OUT/'source-inventory.json').read_text())['videos']
    # Positives first so visual annotation can proceed while negatives are processed.
    videos.sort(key=lambda item:(item['video_label']=='other', item['video_label'], item['source']))
    if args.ids:
        wanted=set(args.ids.split(','))
        videos=[item for item in videos if item['id'] in wanted]
    shard, total=map(int,args.shard.split('/'))
    videos=videos[shard::total]
    if args.limit:
        videos=videos[:args.limit]
    detector=YoloOnnxDetector(str(OP/'ckpts'),'cuda:0')
    skeleton=SkeletonAux(str(OP/'ckpts'),'cuda:0',.7)
    enhancer=RetinexformerStage(str(OP/'ckpts'),chunk=1)
    for name, model in [('detector',detector),('skeleton',skeleton)]:
        reason=model.load()
        if reason:
            raise RuntimeError(f'{name}: {reason}')
    enhancer.load('cuda:0')
    try:
        for index,item in enumerate(videos,1):
            result=prepare(item,detector,skeleton,enhancer)
            print(f'[prepare] {index}/{len(videos)} {item["source"]} '
                  f'frames={len(result["frames"])} windows={len(result["windows"])} '
                  f'tracks={len(result["review"])} seconds={result["elapsed"]:.1f}',flush=True)
    finally:
        enhancer.close()
        skeleton.net=skeleton.pose=detector.sess=None


if __name__=='__main__':
    main()
