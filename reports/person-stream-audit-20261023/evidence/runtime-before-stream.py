"""Independent optimized per-person runtime: shared algorithm, one to many GPUs."""
from collections import Counter, deque
import copy
import json
from pathlib import Path
import queue
import sys
import threading
import time

import cv2
import torch

from action_engine import ROOT, OP, ActionTask
from scheduling import ActionPool, ResultStore
from person_core import PersonTracker, person_boxes, associate_poses, render_people
from demo import content_height
from darkpipe.media import VideoWriter
from darkpipe.stages.downscale import DownscaleStage
from darkpipe.stages.enhance_retinexformer import RetinexformerStage
from darkpipe.stages.object_prior import YoloOnnxDetector
from darkpipe.stages.skeleton_aux import SkeletonAux
from darkpipe.stages.recognize import frame_luma


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


class Decoder:
    def __init__(self, source: Path, processed: bool = False):
        self.cap = cv2.VideoCapture(str(source))
        if not self.cap.isOpened():
            raise RuntimeError(f'Cannot read {source}')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.queue = queue.Queue(maxsize=2)
        self.stop = threading.Event()
        self.error = None
        self.processed = processed
        self.thread = threading.Thread(target=self._run, daemon=True, name='person-decode')
        self.thread.start()

    def _put(self, value) -> None:
        while not self.stop.is_set():
            try:
                self.queue.put(value, timeout=.1)
                return
            except queue.Full:
                pass

    def _run(self) -> None:
        try:
            downscale = DownscaleStage(720)
            while not self.stop.is_set():
                ok, frame = self.cap.read()
                if not ok:
                    self._put(None)
                    break
                shown = frame[:content_height(frame.shape[0])] if self.processed else None
                raw = downscale([shown if self.processed else frame])[0]
                luma = None if self.processed else frame_luma(raw)
                self._put((raw, shown, luma))
        except Exception as exc:
            self.error = exc
        finally:
            self.cap.release()

    def read(self):
        while True:
            if self.error:
                raise RuntimeError('Decoder failed') from self.error
            try:
                return self.queue.get(timeout=.1)
            except queue.Empty:
                if not self.thread.is_alive():
                    raise RuntimeError('Decoder exited without EOF')

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError('Decoder failed to stop')


class OutputWorker:
    def __init__(self, path: Path, fps: float, origin: float, paced: bool):
        self.writer = VideoWriter(str(path), fps=fps)
        self.origin, self.paced = origin, paced
        self.queue = queue.Queue(maxsize=2)
        self.error = None
        self.stop = threading.Event()
        self.frames = []
        self.thread = threading.Thread(target=self._run, daemon=True, name='person-output')
        self.thread.start()

    def put(self, item) -> None:
        while not self.stop.is_set():
            if self.error:
                raise RuntimeError('Output failed') from self.error
            try:
                self.queue.put(item, timeout=.1)
                return
            except queue.Full:
                pass
        raise RuntimeError('Output stopped')

    def _run(self) -> None:
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    break
                frame, shown, tracks, timestamp, record = item
                if shown is None:
                    shown = cv2.resize(frame, (frame.shape[1]*2, frame.shape[0]*2), interpolation=cv2.INTER_CUBIC)
                now = max(timestamp, time.monotonic()-self.origin) if self.paced else timestamp
                for track in tracks:
                    if track.prediction and now-track.last_inference > 1.:
                        track.prediction = None
                rendered = render_people(shown, tracks, frame.shape, now)
                self.writer.write(rendered)
                record.update(output_age=time.monotonic()-self.origin-timestamp,
                    output_time=time.monotonic()-self.origin,
                    classified=sum(t.prediction is not None for t in tracks),
                    output_size=list(rendered.shape[1::-1]))
                record['classified_ready'] = sum(t.prediction is not None and t.tid in record['ready_ids'] for t in tracks)
                self.frames.append(record)
        except Exception as exc:
            self.error = exc
        finally:
            try:
                self.writer.close()
            except Exception as exc:
                self.error = self.error or exc
            self.stop.set()

    def close(self) -> None:
        if not self.stop.is_set():
            self.put(None)
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError('Output failed to stop')
        if self.error:
            raise RuntimeError('Output failed') from self.error


class PersonRuntime:
    def __init__(self, devices: list[str], max_batch: int | None = None,
                 diagnostics: bool = False, coalesce: bool = False, process_workers: bool = False):
        if not devices or len(set(devices)) != len(devices):
            raise ValueError('Provide distinct devices')
        self.devices = devices
        self.front_device = devices[0]
        self.session = 0
        self.process_workers = process_workers and len(devices)>1
        if self.process_workers:
            from process_pool import ProcessActionPool
            self.pool = ProcessActionPool(devices[1:], max_batch=max_batch or 4,
                                          diagnostics=diagnostics, coalesce=coalesce)
        else:
            self.pool = ActionPool(devices if len(devices)==1 else devices[1:],
                inline=len(devices)==1, max_batch=max_batch or (2 if len(devices)==1 else 4),
                diagnostics=diagnostics, coalesce=coalesce)
        with torch.cuda.device(self.front_device):
            self.enhancer = RetinexformerStage(str(OP/'ckpts'), chunk=1)
            self.enhancer.load(self.front_device)
            self.detector = YoloOnnxDetector(str(OP/'ckpts'), self.front_device)
            reason = self.detector.load()
            if reason:
                raise RuntimeError(reason)
            self.skeleton = SkeletonAux(str(OP/'ckpts'), self.front_device, .7)
            reason = self.skeleton.load()
            if reason:
                raise RuntimeError(reason)

    def run(self, item: dict, output: Path, paced: bool = False, source_factory=None) -> dict:
        output.mkdir(parents=True, exist_ok=True)
        if (output/'result.json').exists():
            raise RuntimeError(f'Refusing to overwrite {output}')
        assert all(box.idle() for box in self.pool.mailboxes)
        self.session += 1
        self.pool.assignment.clear()
        self.pool.assignment_next = 0
        decoder = (source_factory or Decoder)(ROOT/item['source'], item.get('processed', False))
        fps = decoder.fps
        tracker, downscale, store = PersonTracker(), DownscaleStage(720), ResultStore()
        generations, last_seen, last_submitted = {}, {}, {}
        events, stage_times = [], Counter()
        started = getattr(decoder, 'origin', time.monotonic())
        sink = OutputWorker(output/'video.mp4', fps, started, paced)
        index, submitted = 0, 0
        cycle = 0
        source_indices = []
        engine_before = [Counter(e.stats) for e in self.pool.engines]
        queue_before = [Counter(box.stats) for box in self.pool.mailboxes]
        call_start = [len(e.calls) for e in self.pool.engines]

        def collect(rows: list) -> None:
            for row in rows:
                if row['session'] != self.session:
                    raise RuntimeError('Result crossed a session boundary')
                events.append(row)
                store.update(row)

        try:
            while True:
                decoded = decoder.read()
                if decoded is None:
                    break
                if source_factory:
                    source_index, raw, shown, luma = decoded
                    current_cycle = source_index//decoder.source_count
                    if current_cycle != cycle:
                        cycle = current_cycle
                        tracker = PersonTracker()
                        generations.clear()
                        last_seen.clear()
                        last_submitted.clear()
                else:
                    raw, shown, luma = decoded
                    source_index = index
                source_indices.append(source_index)
                timestamp = source_index/fps
                if paced and not source_factory:
                    delay = started+timestamp-time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                frame_start = time.monotonic()
                with torch.cuda.device(self.front_device):
                    if item.get('processed', False):
                        frame = raw
                    else:
                        frame = self.enhancer([raw])[0]
                    enhanced = time.monotonic()
                    result = self.detector([frame])
                    if result is None:
                        raise RuntimeError('Person detector failed')
                    boxes = person_boxes(result[0])
                    poses = [None]*len(boxes)
                    if len(boxes) and (luma is None or luma >= 75):
                        xy, scores = self.skeleton.pose(frame)
                        poses = associate_poses(boxes, xy, scores)
                    detected = time.monotonic()
                    tracks = tracker.update(boxes, poses, frame, timestamp)
                    ready_ids = []
                    for track in tracks:
                        previous_seen = last_seen.get(track.tid, timestamp)
                        if timestamp-previous_seen > .2:
                            generations[track.tid] = generations.get(track.tid, 0)+1
                        last_seen[track.tid] = timestamp
                        key = (self.session, track.tid, (cycle << 32)+generations.get(track.tid, 0))
                        track.append(frame, timestamp, source_index, luma)
                        window = track.window(fps)
                        if window is not None:
                            ready_ids.append(track.tid)
                        if window is not None and timestamp-last_submitted.get(key, -1e9) >= .5-1e-6:
                            if self.pool.inline:
                                while len(self.pool.mailboxes[0].pending) >= self.pool.mailboxes[0].capacity:
                                    self.pool.step()
                                    collect(self.pool.collect())
                            task = ActionTask(key, tuple(window), tuple(track.box), track.score,
                                              time.monotonic(), started)
                            self.pool.submit(task)
                            last_submitted[key] = timestamp
                            submitted += 1
                    tracked = time.monotonic()
                    self.pool.step()  # inline: bounded small batch; multi-GPU workers run independently
                    collect(self.pool.collect())
                ready = time.monotonic()
                copied = []
                for track in tracks:
                    key = (self.session, track.tid, (cycle << 32)+generations.get(track.tid, 0))
                    t = copy.copy(track)
                    t.history = deque()
                    t.prediction = store.for_track(key, timestamp)
                    t.last_inference = t.prediction['timestamp'] if t.prediction else -1e9
                    copied.append(t)
                record = dict(frame=source_index, timestamp=timestamp, people=len(tracks),
                    raw_luma=luma, source_backlog=max(0., frame_start-started-timestamp),
                    frontend_seconds=detected-frame_start, tracking_seconds=tracked-detected,
                    inline_action_seconds=ready-tracked,
                    track_ids=[t.tid for t in tracks], ready_ids=ready_ids)
                sink.put((frame, shown, copied, timestamp, record))
                stage_times.update(enhance_resize=enhanced-frame_start, detect_pose=detected-enhanced,
                                   tracking_submit=tracked-detected, action_collect=ready-tracked)
                index += 1
                if source_factory and index % 300 == 0:
                    health = dict(frames=index, source_time=timestamp, wall_time=time.monotonic()-started,
                        source_dropped=decoder.dropped, submitted=submitted, completed=len(events),
                        latest_decision_age=events[-1]['decision_age'] if events else None,
                        pending=[len(box.pending) for box in self.pool.mailboxes])
                    save_json(output/'progress.json', health)
                    print('[live-progress]', item['name'], health, flush=True)
            collect(self.pool.drain())
        finally:
            decoder.close()
            sink.close()
        elapsed = time.monotonic()-started
        assert index == decoder.count == sink.writer.count == len(sink.frames)
        coalesced = sum(box.stats['coalesced']-old['coalesced'] for box, old in zip(self.pool.mailboxes, queue_before))
        assert submitted == len(events)+coalesced
        # Sorting is for comparison only; completion times retain actual publication order.
        events.sort(key=lambda r: (r['frame_index'], r['track_id']))
        result = dict(name=item['name'], source=item['source'], processed=item.get('processed', False),
            devices=self.devices, process_workers=self.process_workers,
            paced=paced, frames=index, source_fps=fps, duration=index/fps,
            elapsed=elapsed, processing_fps=index/elapsed, submitted=submitted, completed=len(events),
            coalesced=coalesced, dropped_source_frames=getattr(decoder, 'dropped', 0),
            source_frames_offered=getattr(decoder, 'total', index),
            source_duration=getattr(decoder, 'total', index)/fps,
            source_indices=source_indices, clip_boundary_resets=cycle,
            stale_overwrites_rejected=store.rejected_old,
            labels=dict(Counter(r['label'] for r in events)), modes=dict(Counter(r['mode'] for r in events)),
            max_people=max(f['people'] for f in sink.frames), timings=dict(stage_times),
            action_workers=[dict(device=device, stats=dict(e.stats-before), calls=e.calls[start:])
                for device, e, before, start in zip(self.pool.devices, self.pool.engines, engine_before, call_start)])
        save_json(output/'events.json', events)
        save_json(output/'frames.json', sink.frames)
        save_json(output/'result.json', result)
        print('[optimized]', item['name'], self.devices, f'{index/elapsed:.3f} fps',
              f'{len(events)}/{submitted} windows', flush=True)
        return result

    def close(self) -> None:
        self.pool.close()
        with torch.cuda.device(self.front_device):
            self.enhancer.close()
            self.detector.sess = self.skeleton.pose = self.skeleton.net = None
