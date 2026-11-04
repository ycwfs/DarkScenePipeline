"""Real per-person RGB + skeleton demo; independent of the platform service contract."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from person_core import (ROOT, OP, BEHAVIORS, PersonTracker, associate_poses, person_boxes,
                         fuse_person, render_people)
from darkpipe.media import VideoWriter
from darkpipe.stages.recognize import BehaviorRecognizer, frame_luma, gate_luma
from darkpipe.stages.skeleton_aux import SkeletonAux, normalize_window, completeness
from darkpipe.stages.object_prior import YoloOnnxDetector
from darkpipe.stages.downscale import DownscaleStage
from darkpipe.stages.enhance_retinexformer import RetinexformerStage

OUT = Path(__file__).resolve().parents[1]


def content_height(height: int) -> int:
    values = [h for h in range(height-180, height) if h % 2 == 0 and
              h+max(48, round(.08*h))+(max(48, round(.08*h)) & 1) == height]
    if len(values) != 1:
        raise ValueError(f'Ambiguous old label bar for {height}')
    return values[0]


class PersonModels:
    def __init__(self, device: str):
        self.device = device
        self.rgb = BehaviorRecognizer(str(OP/'ckpts'), skel_fusion_alpha=1.)
        self.rgb.load(device)
        assert self.rgb.cfg.get('fov') == (224, 400)
        self.rgb.warmup()
        self.skeleton = SkeletonAux(str(OP/'ckpts'), device, .7)
        reason = self.skeleton.load()
        if reason:
            raise RuntimeError(reason)
        self.skeleton.warmup()
        self.detector = YoloOnnxDetector(str(OP/'ckpts'), device)
        reason = self.detector.load()
        if reason:
            raise RuntimeError(reason)

    @torch.inference_mode()
    def predict(self, windows: list) -> list[dict]:
        """Same-track RGB and pose sequences share exactly the same 32 timestamps."""
        results = []
        for start in range(0, len(windows), 2):
            part = windows[start:start+2]  # one shared RGB model, bounded memory batch
            pixels = np.stack([np.stack([s.crop for s in window]) for window in part])
            array = (pixels.astype(np.float32)/255.-self.rgb.cfg['mean'])/self.rgb.cfg['std']
            x = torch.from_numpy(array).to(self.device).permute(0, 4, 1, 2, 3).contiguous()
            with torch.autocast('cuda', dtype=torch.float16):
                rgb_probs = self.rgb.net(x).softmax(1).float().cpu().numpy()
            pose_inputs, eligible = [], []
            for window in part:
                sk = np.stack([s.pose for s in window])[:, None]
                normalized, presence = normalize_window(sk)
                valid_frames = ((sk[:, 0, :, 2] > .3).sum(1) >= 10)
                valid = bool(presence[0] >= .5 and completeness(sk)[0] >= .5
                             and valid_frames.mean() >= .5
                             and sk[-1, 0, :, 2].mean() >= .5 and valid_frames[-1])
                valid &= gate_luma([s.luma for s in window], 75) is None
                pose_inputs.append(normalized[0])
                eligible.append(valid)
            probabilities = {}
            indices = np.flatnonzero(eligible)
            if len(indices):
                skx = torch.from_numpy(np.stack([pose_inputs[i] for i in indices])).to(self.device)
                skp = self.skeleton.net(skx).softmax(1).float().cpu().numpy()
                probabilities = dict(zip(indices.tolist(), skp))
            for i, (window, pr) in enumerate(zip(part, rgb_probs)):
                row = fuse_person(pr, probabilities.get(i), [s.luma for s in window])
                row.update(pose_valid=eligible[i], source_frames=[s.frame_index for s in window],
                           source_timestamps=[s.timestamp for s in window],
                           crop_boxes=[s.crop_box.tolist() for s in window],
                           crop_sha256=hashlib.sha256(pixels[i].tobytes()).hexdigest())
                results.append(row)
        return results

    def close(self) -> None:
        self.rgb.close()
        self.skeleton.net = self.skeleton.pose = self.detector.sess = None


def run_sample(item: dict, models: PersonModels, enhancer: RetinexformerStage, output: Path) -> dict:
    source = ROOT/item['source']
    name = item['name']
    cap = cv2.VideoCapture(str(source))
    assert cap.isOpened(), source
    fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed = item.get('processed', False)
    downscale, tracker = DownscaleStage(720), PersonTracker()
    writer = VideoWriter(str(output/'videos'/f'{name}_逐人融合.mp4'), fps=fps)
    events, frame_records, labels, modes = [], [], Counter(), Counter()
    best = None
    best_rank = (-1, -1, -1)
    index, started = 0, time.monotonic()
    timings = Counter()
    crop_examples = set()
    try:
        while True:
            ok, source_frame = cap.read()
            if not ok:
                break
            timestamp = index/fps
            t0 = time.monotonic()
            if processed:
                content = source_frame[:content_height(source_frame.shape[0])]
                frame = downscale([content])[0]
                luma = None
                shown = content
            else:
                raw = downscale([source_frame])[0]
                luma = frame_luma(raw)
                frame = enhancer([raw])[0]
                shown = cv2.resize(frame, (frame.shape[1]*2, frame.shape[0]*2), interpolation=cv2.INTER_CUBIC)
            timings['preprocess'] += time.monotonic()-t0
            t0 = time.monotonic()
            detections = models.detector([frame])
            if detections is None:
                raise RuntimeError('Person detector failed during demo')
            boxes = person_boxes(detections[0])
            poses = [None]*len(boxes)
            if len(boxes) and (luma is None or luma >= 75):
                points, scores = models.skeleton.pose(frame)
                poses = associate_poses(boxes, points, scores)
            timings['detection_pose'] += time.monotonic()-t0
            current = tracker.update(boxes, poses, frame, timestamp)
            ready, windows = [], []
            for track in current:
                track.append(frame, timestamp, index, luma)
                if timestamp-track.last_inference > 1.:
                    track.prediction = None
                window = track.window(fps)
                if window is not None and timestamp-track.last_inference >= .5-1e-6:
                    ready.append(track)
                    windows.append(window)
            t0 = time.monotonic()
            predictions = models.predict(windows)
            timings['person_actions'] += time.monotonic()-t0
            for track, window, prediction in zip(ready, windows, predictions):
                prediction.update(track_id=track.tid, timestamp=timestamp, frame_index=index,
                                  bbox=track.box.tolist(), detector_score=track.score)
                track.prediction, track.last_inference = prediction, timestamp
                events.append(prediction)
                labels.update([prediction['label']])
                modes.update([prediction['mode']])
                if track.tid not in crop_examples:
                    crops = [cv2.cvtColor(window[i].crop, cv2.COLOR_RGB2BGR) for i in (0, 10, 21, 31)]
                    contact = np.concatenate(crops, axis=1)
                    cv2.imwrite(str(output/'crops'/f'{name}_T{track.tid:02d}.jpg'), contact)
                    crop_examples.add(track.tid)
            t0 = time.monotonic()
            rendered = render_people(shown, current, frame.shape, timestamp)
            writer.write(rendered)
            timings['render_encode'] += time.monotonic()-t0
            record = dict(frame=index, timestamp=timestamp, raw_luma=luma, people=len(current),
                          tracks=[dict(track_id=t.tid, bbox=t.box.tolist(),
                                       label=t.prediction['label'] if t.prediction else 'warming_up',
                                       mode=t.prediction['mode'] if t.prediction else None,
                                       prediction_timestamp=t.last_inference if t.prediction else None)
                                  for t in current])
            frame_records.append(record)
            classified = [t for t in current if t.prediction is not None]
            rank = (len(classified), len({t.prediction['label'] for t in classified}), len(current))
            if rank > best_rank:
                best, best_rank = rendered.copy(), rank
            index += 1
            if index % 60 == 0:
                print(f'[progress] {name}: {index}/{count} frames, {len(current)} people, {len(events)} predictions', flush=True)
    finally:
        cap.release()
        writer.close()
    assert count == index == writer.count
    cv2.imwrite(str(output/'stills'/f'{name}.jpg'), best)
    (output/'events'/f'{name}.json').write_text(json.dumps(events, ensure_ascii=False, indent=2)+'\n')
    (output/'events'/f'{name}_frames.json').write_text(json.dumps(frame_records, ensure_ascii=False, indent=2)+'\n')
    result = dict(name=name, source=item['source'], source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  processed=processed, frames=count, fps=fps, duration=count/fps,
                  video=f'videos/{name}_逐人融合.mp4', still=f'stills/{name}.jpg',
                  output_size=list(rendered.shape[1::-1]), global_footer=False,
                  predictions=len(events), labels=dict(labels), modes=dict(modes),
                  max_detected_people=max(r['people'] for r in frame_records),
                  distinct_track_ids=tracker.next_id-1, classified_track_ids=sorted(crop_examples),
                  frames_with_at_least_two_classified=sum(sum(t['label']!='warming_up' for t in r['tracks'])>=2 for r in frame_records),
                  elapsed=time.monotonic()-started, timings=dict(timings))
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--selection', type=Path, default=OUT/'selection.json')
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    for subdir in ('videos', 'stills', 'events', 'crops'):
        (args.output/subdir).mkdir(parents=True, exist_ok=True)
    models = PersonModels(args.device)
    enhancer = RetinexformerStage(str(OP/'ckpts'), chunk=1)
    enhancer.load(args.device)
    results = []
    try:
        for item in json.loads(args.selection.read_text()):
            results.append(run_sample(item, models, enhancer, args.output))
            (args.output/'sample-results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2)+'\n')
    finally:
        models.close()
        enhancer.close()


if __name__ == '__main__':
    main()
