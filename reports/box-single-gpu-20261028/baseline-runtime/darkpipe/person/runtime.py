"""Shared bounded per-person engine for offline files and live cameras."""
from collections import Counter, deque
import copy
from dataclasses import replace
from pathlib import Path
import time
import threading

import cv2
import numpy as np
import onnxruntime as ort
import torch

from .action_engine import ActionTask, MODEL_FILES
from .core import PersonTracker, associate_poses, person_boxes, iou
from .object_prior import allows_object_prior
from .object_runtime import FrameHistory
from .scheduling import ActionPool, ResultStore
from darkpipe.stages import build_stages
from darkpipe.stages.object_prior import YoloOnnxDetector
from darkpipe.stages.recognize import frame_luma


def content_height(height: int) -> int:
    values = [h for h in range(max(2, height-180), height) if h % 2 == 0 and
              h+max(48, round(.08*h))+(max(48, round(.08*h)) & 1) == height]
    if len(values) != 1:
        raise ValueError(f'Ambiguous old label bar for {height}')
    return values[0]


def prepare_stages(stages, cfg, raw, input_cropped=False):
    if cfg.input_processed and not input_cropped:
        raw = raw[:content_height(raw.shape[0])]
    frame, luma = raw, None
    for stage in stages:
        if stage.post_recognition:
            continue
        if stage.name != 'downscale' and luma is None and not cfg.input_processed:
            luma = frame_luma(frame)
        if cfg.input_processed and stage.name != 'downscale':
            continue
        frame = stage([frame])[0]
    if luma is None and not cfg.input_processed:
        luma = frame_luma(frame)
    return frame, luma


class Frontend:
    def __init__(self, cfg):
        self.detector = YoloOnnxDetector(cfg.ckpt_dir, cfg.device)
        sessions = []
        for name in ('detector', 'pose'):
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            device_id = int(cfg.device.split(':')[-1])
            session = ort.InferenceSession(str(Path(cfg.ckpt_dir)/MODEL_FILES[name]),
                sess_options=options, providers=[('CUDAExecutionProvider', {'device_id': device_id})])
            if session.get_providers()[0] != 'CUDAExecutionProvider':
                raise RuntimeError(f'{name} unexpectedly fell back to CPU')
            sessions.append(session)
        self.detector.sess, self.pose = sessions

    def __call__(self, frame, pose_enabled):
        boxes = person_boxes(self.detector._frames_to_dets([frame])[0])
        poses = [None] * len(boxes)
        if pose_enabled and len(boxes):
            h, w = frame.shape[:2]
            ratio = 640/max(h, w)
            canvas = np.full((640, 640, 3), 114, np.uint8)
            resized = cv2.resize(frame, (int(w*ratio), int(h*ratio)), interpolation=cv2.INTER_LINEAR)
            canvas[:resized.shape[0], :resized.shape[1]] = resized
            x = canvas.transpose(2, 0, 1)[None].astype(np.float32)
            dets, points = self.pose.run(None, {self.pose.get_inputs()[0].name: x})
            dets, points = dets[0], points[0]
            xywh = dets[:, :4].copy()
            xywh[:, 2:] -= xywh[:, :2]
            keep = np.asarray(cv2.dnn.NMSBoxes(xywh.tolist(), dets[:, 4].tolist(), .45, .45), int).reshape(-1)
            poses = associate_poses(boxes, points[keep, :, :2]/ratio, points[keep, :, 2])
        return boxes, poses


class PersonEngine:
    def __init__(self, cfg, origin: float, on_event, realtime: bool):
        self.cfg, self.origin, self.on_event, self.realtime = cfg, origin, on_event, realtime
        self.pool = None
        self.front = None
        self.front_process = None
        self.stages = []
        self.tracker = PersonTracker()
        self.prior_history = FrameHistory()
        self.object_rows = deque(maxlen=256)
        self.object_rows_lock = threading.Lock()
        self.store = ResultStore()
        self.last_seen, self.generations, self.last_submitted = {}, {}, {}
        self.epoch, self.frames, self.events = 0, 0, 0
        self.stats = Counter()
        self.decision_ages = deque(maxlen=2048)
        self.max_decision_age = 0.
        try:
            devices = [f'cuda:{x.strip()}' for x in cfg.gpus.split(',') if x.strip()] or [cfg.device]
            options = dict(ckpt_dir=cfg.ckpt_dir, alpha=cfg.skel_fusion_alpha,
                threshold=cfg.reco_min_conf, lum_gate=cfg.lum_gate, no_pose=cfg.skel_no_person,
                person_object_prior=cfg.person_object_prior, object_min_conf=cfg.object_min_conf,
                wave_pose_min_frames=cfg.wave_pose_min_frames, running_pose_min_speed=cfg.running_pose_min_speed)
            front_count = min(2, len(devices)-1) if realtime else 1
            if len(devices) > 1:
                from .process_pool import ProcessActionPool
                self.pool = ProcessActionPool(devices[front_count:], max_batch=4, coalesce=realtime, options=options)
            else:
                self.pool = ActionPool(devices, inline=True, max_batch=2, coalesce=False, options=options)
            with torch.cuda.device(cfg.device):
                stages, _ = build_stages(replace(cfg, recognize='off', enhance_chunk=1))
                if realtime and len(devices) > 1:
                    from .frontend_process import FrontendPool
                    self.front_process = FrontendPool(cfg, devices[:front_count])
                    stages = [s for s in stages if s.post_recognition]
                self.stages = stages
                for stage in stages:
                    stage.load(cfg.device)
                if self.front_process is None:
                    self.front = Frontend(cfg)
                # Initialize shape-specific kernels before exposing the first live frame.
                dummy = np.full((452 if cfg.input_processed else 404, 720, 3), 48, np.uint8)
                for _ in range(2 if self.front_process is None else 0):
                    frame, _ = self.prepare(dummy)
                    self.front(frame, True)
                torch.cuda.synchronize(cfg.device)
            print(f'[person] 逐人模式 ready，GPU={devices} 门限={cfg.lum_gate:g} '
                  f'alpha={cfg.skel_fusion_alpha:g} 动作阈值={cfg.reco_min_conf:g} '
                  f'物体先验={cfg.person_object_prior} 物体阈值={cfg.object_min_conf:g} '
                  f'挥手骨架最少帧={cfg.wave_pose_min_frames} 跑动速度阈值={cfg.running_pose_min_speed:g}', flush=True)
        except BaseException:
            self.close()
            raise

    def prepare(self, raw):
        return prepare_stages(self.stages, self.cfg, raw)

    def collect(self, drain=False):
        rows = self.pool.drain() if drain else self.pool.collect()
        for row in rows:
            if row.get('object_prior', {}).get('samples'):
                with self.object_rows_lock:
                    self.object_rows.append(row)
            self.store.update(row)
            self.events += 1
            age = max(0., row['decision_age'])
            self.decision_ages.append(age)
            self.max_decision_age = max(self.max_decision_age, age)
            self.on_event(row)
        return rows

    def overlay_evidence(self):
        with self.object_rows_lock:
            return list(self.object_rows)

    def process(self, raw, index: int, stamp: float, epoch: int, fps: float, prepared=None):
        started = time.monotonic()
        if epoch != self.epoch:
            self.tracker = PersonTracker()
            self.prior_history.clear()
            with self.object_rows_lock:
                self.object_rows.clear()
            self.store = ResultStore()
            self.last_seen.clear()
            self.generations.clear()
            self.last_submitted.clear()
            self.epoch = epoch
        with torch.cuda.device(self.cfg.device):
            with self.object_rows_lock:
                while self.object_rows and stamp-self.object_rows[0]['timestamp'] > 2.5:
                    self.object_rows.popleft()
            if self.front_process is None:
                source_shape = raw.shape
                if self.cfg.input_processed:
                    source_shape = (content_height(raw.shape[0]), *raw.shape[1:])
                frame, luma = self.prepare(raw)
                prepared_at = time.monotonic()
                boxes, poses = self.front(frame, self.cfg.skel_fusion_alpha < .999 and
                                          (luma is None or luma >= self.cfg.lum_gate))
            else:
                if prepared is None:
                    raise ValueError('Live frontend must submit and collect before tracking')
                frame, luma, boxes, poses, prepare_time, vision_time, source_shape = prepared
                prepared_at = started+prepare_time
            vision_at = time.monotonic()
            tracks = self.tracker.update(boxes, poses, frame, stamp)
            if self.cfg.person_object_prior:
                self.prior_history.put(index, stamp,
                    frame if allows_object_prior(luma, self.cfg.lum_gate) else None, source_shape)
            keys = {}
            for track in tracks:
                if stamp-self.last_seen.get(track.tid, stamp) > .2:
                    self.generations[track.tid] = self.generations.get(track.tid, 0)+1
                self.last_seen[track.tid] = stamp
                key = (1, track.tid, (epoch << 32)+self.generations.get(track.tid, 0))
                keys[track.tid] = key
                binding_valid = not any(iou(track.box, other.box) > .4 for other in tracks if other is not track)
                track.append(frame, stamp, index, luma, binding_valid=binding_valid)
                window = track.window(fps)
                if window is not None and stamp-self.last_submitted.get(key, -1e9) >= .5-1e-6:
                    if self.pool.inline:
                        while len(self.pool.mailboxes[0].pending) >= self.pool.mailboxes[0].capacity:
                            self.pool.step()
                            self.collect()
                    self.pool.submit(ActionTask(key, tuple(window), tuple(track.box), track.score,
                        time.monotonic(), self.origin,
                        self.prior_history.select(window) if self.cfg.person_object_prior else ()))
                    self.last_submitted[key] = stamp
                    self.stats['submitted'] += 1
            tracked_at = time.monotonic()
            self.pool.step()
            self.collect(drain=not self.realtime)
        copied = []
        for track in tracks:
            t = copy.copy(track)
            t.history = deque()
            t.prediction = self.store.for_track(keys[t.tid], stamp)
            t.last_inference = t.prediction['timestamp'] if t.prediction else -1e9
            copied.append(t)
        self.frames += 1
        if self.frames % 30 == 0:
            alive = {tid for tid, seen in self.last_seen.items() if stamp-seen < 3.}
            self.last_seen = {k:v for k,v in self.last_seen.items() if k in alive}
            self.generations = {k:v for k,v in self.generations.items() if k in alive}
            self.last_submitted = {k:v for k,v in self.last_submitted.items() if k[1] in alive and k[2] >> 32 == epoch}
            self.pool.assignment = {k:v for k,v in self.pool.assignment.items() if k[1] in alive and k[2] >> 32 == epoch}
        record = dict(frame=index, timestamp=stamp, capture_epoch=epoch, people=len(tracks),
            raw_luma=luma, model_shape=frame.shape, tracks=copied, keys=keys,
            prepare_seconds=prepared_at-started, vision_seconds=vision_at-prepared_at,
            tracks_seconds=tracked_at-vision_at,
            frontend_seconds=time.monotonic()-started)
        if prepared is not None:
            record.update(prepare_seconds=prepare_time, vision_seconds=vision_time,
                          frontend_seconds=prepare_time+vision_time+time.monotonic()-started)
        return frame, copied, record

    def picture(self, frame):
        for stage in self.stages:
            if stage.post_recognition:
                frame = stage([frame])[0]
        return frame

    def snapshot(self):
        return dict(frames=self.frames, events=self.events, **dict(self.stats),
            object_history_frames=len(self.prior_history.frames), object_history_bytes=self.prior_history.bytes,
            max_decision_age=self.max_decision_age,
            decision_age_p95=float(np.percentile(self.decision_ages, 95)) if self.decision_ages else None,
            coalesced=sum(b.stats['coalesced'] for b in self.pool.mailboxes) if self.pool else 0,
            action_workers=[dict(device=d, stats=dict(e.stats), queue=dict(b.stats))
                for d,e,b in zip(self.pool.devices, self.pool.engines, self.pool.mailboxes)] if self.pool else [])

    def close(self):
        try:
            if self.pool is not None:
                self.pool.close()
        finally:
            if self.front_process is not None:
                self.front_process.close()
            self.front = None
            self.prior_history.clear()
            for stage in self.stages:
                stage.close()
