"""Existing ROI VideoMamba/ST-GCN with bounded GPU input cache and small batches."""
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
import time

import numpy as np
import torch

from .core import fuse_person
from .wave_prior import apply_wave_prior
from .running_prior import apply_running_prior
from darkpipe.stages.recognize import BehaviorRecognizer, gate_luma
from darkpipe.stages.skeleton_aux import STGCN, normalize_window, completeness

MODEL_FILES = dict(rgb='videomamba_t_behavior_32f_person_roi_20261023.pth',
                   skeleton='stgcn_person_20261024.pth',
                   detector='yolo26m_person_20261024.onnx', pose='rtmo-s_person_20261024.onnx')


@dataclass(frozen=True)
class ActionTask:
    key: tuple  # session, track ID, history generation
    window: tuple
    bbox: tuple
    score: float
    submitted: float
    origin: float
    prior_frames: tuple = ()

    @property
    def timestamp(self) -> float:
        return self.window[-1].timestamp


class SlotCache:
    """Bounded LRU index; never evict an input used by the current batch."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.entries = OrderedDict()
        self.free = list(range(capacity-1, -1, -1))

    def reserve(self, keys: list) -> tuple:
        unique = set(keys)
        if len(unique) > self.capacity:
            raise ValueError('Batch exceeds input cache capacity')
        missing = []
        for key in keys:
            if key in self.entries:
                self.entries.move_to_end(key)
                continue
            if self.free:
                slot = self.free.pop()
            else:
                victim = next(k for k in self.entries if k not in unique)
                slot = self.entries.pop(victim)
            self.entries[key] = slot
            missing.append((key, slot))
        return [self.entries[k] for k in keys], missing


class ActionEngine:
    def __init__(self, device: str, max_batch: int = 4, cache_frames: int = 512,
                 diagnostics: bool = False, options: dict | None = None):
        if max_batch < 1 or cache_frames < 32*max_batch:
            raise ValueError('Cache must hold at least one complete batch')
        self.options = options or {}
        ckpt_dir = Path(self.options['ckpt_dir'])
        self.device, self.max_batch = device, max_batch
        self.diagnostics = diagnostics
        self.stats = Counter()
        self.object_prior = None
        self.calls = deque(maxlen=512)
        self.rgb = BehaviorRecognizer(str(ckpt_dir), skel_fusion_alpha=1.)
        self.rgb.ckpt = str(ckpt_dir/MODEL_FILES['rgb'])
        with torch.cuda.device(device):
            self.rgb.load(device)
            self.skeleton = STGCN(num_classes=10)
            weights = torch.load(ckpt_dir/MODEL_FILES['skeleton'], map_location='cpu')
            self.skeleton.load_state_dict(weights.get('model', weights))
            self.skeleton = self.skeleton.to(device).eval()
            self.mean = torch.as_tensor(self.rgb.cfg['mean'], device=device)
            self.std = torch.as_tensor(self.rgb.cfg['std'], device=device)
            self.cache = SlotCache(cache_frames)
            self.values = torch.empty((cache_frames, 224, 400, 3), dtype=torch.float32, device=device)
            self.host = torch.empty((32*max_batch, 224, 400, 3), dtype=torch.uint8, pin_memory=True)
            self.host_array = self.host.numpy()
            self.upload = torch.empty_like(self.host, device=device)
            self.transfer_done = torch.cuda.Event()
            self.transfer_pending = False
            self.warmup()
            if self.options.get('person_object_prior', False):
                from .object_runtime import PersonObjectPrior
                self.object_prior = PersonObjectPrior(str(ckpt_dir), device,
                    self.options['object_min_conf'], self.options['lum_gate'])

    @torch.inference_mode()
    def warmup(self) -> None:
        with torch.cuda.device(self.device):
            for batch in sorted({1, 2 if self.max_batch >= 2 else 1, self.max_batch}):
                x = torch.zeros((batch, 3, 32, 224, 400), device=self.device)
                for _ in range(2):
                    with torch.autocast('cuda', dtype=torch.float16):
                        self.rgb.net(x)
                    self.skeleton(torch.zeros((batch, 3, 32, 17), device=self.device))
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def predict(self, tasks: list[ActionTask]) -> list[dict]:
        if not tasks:
            return []
        if len(tasks) > self.max_batch:
            raise ValueError('Task batch exceeds warmed capacity')
        started = time.monotonic()
        with torch.cuda.device(self.device):
            keys, samples = [], {}
            for task in tasks:
                if len(task.window) != 32:
                    raise ValueError('Expected 32 sampled frames')
                for sample in task.window:
                    key = (*task.key, sample.frame_index)
                    keys.append(key)
                    samples[key] = sample
            indices, missing = self.cache.reserve(keys)
            if self.transfer_pending:
                self.transfer_done.synchronize()  # host staging memory must be free before reuse
            for i, (key, _) in enumerate(missing):
                np.copyto(self.host_array[i], samples[key].crop)
            packed = time.monotonic()
            if missing:
                n = len(missing)
                self.upload[:n].copy_(self.host[:n], non_blocking=True)
                self.transfer_done.record()
                self.transfer_pending = True
                normalized = self.upload[:n].float().div_(255.).sub_(self.mean).div_(self.std)
                slots = torch.tensor([slot for _, slot in missing], device=self.device)
                self.values.index_copy_(0, slots, normalized)
            index = torch.tensor(indices, device=self.device)
            pixels = self.values.index_select(0, index).reshape(len(tasks), 32, 224, 400, 3)
            x = pixels.permute(0, 4, 1, 2, 3).contiguous()
            prepared = time.monotonic()
            with torch.autocast('cuda', dtype=torch.float16):
                rgb = self.rgb.net(x).softmax(1).float().cpu().numpy()
            rgb_done = time.monotonic()
            pose_inputs, valid = [], []
            for task in tasks:
                sk = np.stack([sample.pose for sample in task.window])[:, None]
                normalized, presence = normalize_window(sk)
                visible = (sk[:, 0, :, 2] > .3).sum(1) >= 10
                eligible = bool(presence[0] >= .5 and completeness(sk)[0] >= .5
                    and visible.mean() >= .5 and sk[-1, 0, :, 2].mean() >= .5 and visible[-1])
                eligible &= gate_luma([sample.luma for sample in task.window], self.options['lum_gate']) is None
                valid.append(eligible)
                pose_inputs.append(normalized[0])
            sk_probs = {}
            eligible = np.flatnonzero(valid)
            if len(eligible):
                xsk = torch.from_numpy(np.stack([pose_inputs[i] for i in eligible])).to(self.device)
                probs = self.skeleton(xsk).softmax(1).float().cpu().numpy()
                sk_probs = dict(zip(eligible.tolist(), probs))
            rows = []
            for i, task in enumerate(tasks):
                row = fuse_person(rgb[i], sk_probs.get(i), [sample.luma for sample in task.window],
                    alpha=self.options['alpha'], threshold=self.options['threshold'],
                    lum_gate=self.options['lum_gate'], no_pose=self.options['no_pose'])
                row.update(track_id=task.key[1], history_generation=task.key[2],
                    session=task.key[0], timestamp=task.timestamp, frame_index=task.window[-1].frame_index,
                    bbox=[float(v) for v in task.bbox], detector_score=float(task.score), pose_valid=valid[i],
                    source_frames=[sample.frame_index for sample in task.window])
                if self.diagnostics:
                    row['crop_sha256'] = hashlib.sha256(b''.join(s.crop.tobytes() for s in task.window)).hexdigest()
                rows.append(row)
            fused_at = time.monotonic()
            for row, task in zip(rows, tasks):
                if self.object_prior is not None:
                    self.object_prior.apply(task, row)
                else:
                    row['object_prior'] = dict(applied=False, reason='disabled', label=row['label'])
                apply_wave_prior(task, row, min_frames=self.options.get('wave_pose_min_frames', 3),
                    lum_gate=self.options['lum_gate'], pose_enabled=self.options['alpha'] < .999)
                self.stats['wave_'+row['wave_prior']['reason']] += 1
                apply_running_prior(task, row, self.options.get('running_pose_min_speed', 0.35),
                    self.options['lum_gate'], pose_enabled=self.options['alpha'] < .999, missing_policy='bypass')
                self.stats['running_'+row['running_prior']['reason']] += 1
            if self.object_prior is not None:
                for key, value in self.object_prior.stats.items():
                    self.stats['object_'+key] = value
            ended = time.monotonic()
            for row, task in zip(rows, tasks):
                row.update(completed=ended, decision_age=ended-task.origin-task.timestamp,
                           queue_seconds=started-task.submitted, batch_size=len(tasks))
            self.stats.update(windows=len(tasks), uploaded_frames=len(missing),
                              reused_frames=len(keys)-len(missing), calls=1)
            self.calls.append(dict(batch=len(tasks), total_seconds=ended-started,
                pack_seconds=packed-started, gpu_prepare_submission_seconds=prepared-packed,
                rgb_and_synchronization_seconds=rgb_done-prepared,
                pose_fusion_seconds=fused_at-rgb_done, object_seconds=ended-fused_at,
                uploaded_frames=len(missing)))
            return rows

    def close(self) -> None:
        with torch.cuda.device(self.device):
            torch.cuda.synchronize(self.device)
            self.rgb.close()
            self.skeleton = self.values = self.host = self.host_array = self.upload = None
            if self.object_prior is not None:
                self.object_prior.close()
