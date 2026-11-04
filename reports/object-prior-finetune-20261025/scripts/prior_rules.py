"""Candidate-only temporal veto: object evidence can never create an action."""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PriorConfig:
    support_conf: float = .35
    device_conf: float = .65
    exhibit_conf: float = .85
    keypoint_conf: float = .5
    consecutive: int = 3
    min_span: float = .35
    wrist_radius_height: float = .12
    head_radius_height: float = .18
    exhibit_iou: float = .6
    pickup_rise_height: float = .12


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    area = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
    union = np.maximum(0, a[2:]-a[:2]).prod()+np.maximum(0, b[2:]-b[:2]).prod()-area
    return float(area/max(union, 1e-6))


def observations(sample: dict, cfg: PriorConfig) -> list:
    if not sample.get('binding_valid', False):
        return []
    actor = np.asarray(sample['person_box'], float)
    height = max(actor[3]-actor[1], 1.)
    pose = np.asarray(sample['pose'], float) if sample.get('pose') is not None else None
    result = []
    for row in sample['detections']:
        box, conf, cls = np.asarray(row[:4]), float(row[4]), int(row[5])
        center = (box[:2]+box[2:])/2
        obs = dict(cls=cls, center=center.tolist(), height=height, timestamp=sample['timestamp'],
                   robot=False, device=False, drink=False, pickup=False, low=False, y=float(center[1]))
        if cls == 6 and conf >= cfg.exhibit_conf and overlap(box, actor) >= cfg.exhibit_iou:
            obs['robot'] = True
        if pose is not None and pose.shape == (17, 3):
            wrists = [p[:2] for p in pose[[9,10]] if p[2] >= cfg.keypoint_conf]
            heads = [p[:2] for p in pose[:5] if p[2] >= cfg.keypoint_conf]
            knees = [p[1] for p in pose[[13,14]] if p[2] >= cfg.keypoint_conf]
            hand_near = any(np.linalg.norm(center-p) <= cfg.wrist_radius_height*height for p in wrists)
            head_near = bool(heads) and np.linalg.norm(center-np.mean(heads,axis=0)) <= cfg.head_radius_height*height
            inside = bool(np.all(center >= actor[:2]-[.1*height,.05*height]) and
                          np.all(center <= actor[2:]+[.1*height,.05*height]))
            obs['device'] = cls in (2,3) and conf >= cfg.device_conf and hand_near and head_near and inside
            obs['drink'] = cls in (0,1) and conf >= cfg.support_conf and hand_near and head_near and inside
            obs['pickup'] = cls in (0,1,2,3,4,5) and conf >= cfg.support_conf and hand_near and inside
            obs['low'] = obs['pickup'] and bool(knees) and center[1] >= min(knees)-.05*height
        if any(obs[k] for k in ('robot','device','drink','pickup')):
            result.append(obs)
    return result


def chains(samples: list, cfg: PriorConfig) -> list:
    """Require consecutive observations of the same class and nearby object location."""
    active, completed = [], []
    for sample in samples:
        current = observations(sample, cfg)
        new, used = [], set()
        for chain in active:
            last = chain[-1]
            candidates = [(np.linalg.norm(np.asarray(o['center'])-last['center'])/max(o['height'],last['height']),i)
                          for i,o in enumerate(current) if i not in used and o['cls']==last['cls']]
            if candidates and min(candidates)[0] <= .3:
                _, index = min(candidates)
                new.append(chain+[current[index]])
                used.add(index)
            else:
                completed.append(chain)
        new += [[o] for i,o in enumerate(current) if i not in used]
        active = new
    return completed+active


def sustained(chain: list, field: str, cfg: PriorConfig) -> bool:
    run = []
    for row in chain:
        run = run+[row] if row[field] else []
        if len(run) >= cfg.consecutive and run[-1]['timestamp']-run[0]['timestamp'] >= cfg.min_span:
            return True
    return False


def decide(label: str, samples: list, cfg: PriorConfig = PriorConfig()) -> dict:
    if label not in ('drink', 'pick_up'):
        return dict(conservative=label, strict=label, reason='unaffected action', evidence={})
    paths = chains(samples, cfg)
    robot = any(sustained(c,'robot',cfg) for c in paths)
    device = any(sustained(c,'device',cfg) for c in paths)
    drink = any(sustained(c,'drink',cfg) for c in paths)
    pickup = False
    for chain in paths:
        if not sustained(chain, 'pickup', cfg):
            continue
        for i, lo in enumerate(chain):
            if not lo['low']:
                continue
            for hi in chain[i+1:]:
                if (hi['pickup'] and hi['timestamp']-lo['timestamp'] >= .2 and
                    lo['y']-hi['y'] >= cfg.pickup_rise_height*np.mean([lo['height'],hi['height']])):
                    pickup = True
    # A phone near the face is contradictory to drinking. For pickup, a device
    # alone does not disprove another simultaneous hand-object interaction.
    veto = robot or (label == 'drink' and device and not drink)
    support = drink if label == 'drink' else pickup
    return dict(conservative='other' if veto else label,
        strict=label if support and not veto else 'other',
        reason='sustained exhibit overlap' if robot else
               'sustained same-person phone/camera at head' if veto else
               'interaction supported' if support else 'unknown: preserve under conservative rule',
        evidence=dict(robot=robot, device_near_head=device, container_near_head=drink, pickup_lift=pickup),
        config=asdict(cfg))
