"""Causal same-person COCO17 speed, in body-box heights per second.

The ~0.2s displacement lag reduces sensitivity to adjacent-frame pose noise;
coordinates are never smoothed. This is an image-plane speed, not physical speed.
"""
import math
import numpy as np


def speed_evidence(samples, raw_luma, min_speed=0., lum_gate=35., pose_enabled=True, missing_policy='bypass'):
    if not math.isfinite(min_speed) or min_speed < 0:
        raise ValueError('running_pose_min_speed must be finite and nonnegative')
    if not math.isfinite(lum_gate) or lum_gate < 0:
        raise ValueError('lum_gate must be finite and nonnegative')
    if missing_policy not in ('bypass','reject'):
        raise ValueError('missing_policy must be bypass or reject')
    result = dict(applied=False, supported=True, min_speed=float(min_speed), speed=None,
        units='person_box_heights_per_second', lag_seconds=.2, min_joints=5,
        min_pairs=3, keypoint_conf=.5, pairs=[], reason='disabled', missing_policy=missing_policy)
    # Compute geometry even at threshold zero for pre-selection diagnostics.
    if not pose_enabled:
        result['reason'] = 'pose_branch_disabled'
        return result
    if lum_gate and (raw_luma is None or not math.isfinite(raw_luma) or raw_luma < lum_gate):
        result['reason'] = 'below_luma_gate' if raw_luma is not None and math.isfinite(raw_luma) else 'unknown_raw_luma'
        return result
    unique, seen = [], set()
    for sample in samples:
        if sample.frame_index in seen:
            continue
        seen.add(sample.frame_index)
        if not math.isfinite(sample.timestamp) or (unique and sample.timestamp <= unique[-1].timestamp):
            raise ValueError('Distinct source timestamps must be finite and increasing')
        unique.append(sample)
    for i, end in enumerate(unique):
        candidates = [(abs(end.timestamp-start.timestamp-.2), j, start)
                      for j, start in enumerate(unique[:i]) if .16-1e-6 <= end.timestamp-start.timestamp <= .28+1e-6]
        if not candidates:
            continue
        _, j, start = min(candidates, key=lambda x: x[0])
        if any(unique[k+1].timestamp-unique[k].timestamp > .120001 for k in range(j, i)):
            continue
        if any(lum_gate and (s.luma is None or not math.isfinite(s.luma) or s.luma < lum_gate) for s in (start, end)):
            continue
        points, boxes, good = [], [], []
        for s in (start, end):
            shape = getattr(s, 'frame_shape', None)
            box = getattr(s, 'person_box', None)
            p = np.asarray(s.pose, dtype=float)
            if shape is None or box is None or p.shape != (17, 3):
                break
            h, w = shape[:2]
            b = np.asarray(box, dtype=float)
            if not np.isfinite(b).all() or h <= 0 or w <= 0 or np.any(b[2:]-b[:2] <= 0):
                break
            xy = p[:, :2]*[w, h]
            margin = (b[2:]-b[:2])*.2
            valid = np.isfinite(p).all(1) & (p[:, 2] >= .5) & (p[:, 2] <= 1)
            valid &= ((p[:, :2] >= 0) & (p[:, :2] <= 1)).all(1)
            valid &= ((xy >= b[:2]-margin) & (xy <= b[2:]+margin)).all(1)
            points.append(xy); boxes.append(b); good.append(valid)
        if len(points) != 2:
            continue
        valid = good[0] & good[1]
        if valid.sum() < 5:
            continue
        dt = end.timestamp-start.timestamp
        height = sum(b[3]-b[1] for b in boxes)/2
        speed = float(np.linalg.norm(points[1][valid]-points[0][valid], axis=1).mean()/height/dt)
        result['pairs'].append(dict(start_frame=int(start.frame_index), end_frame=int(end.frame_index),
            start_time=float(start.timestamp), end_time=float(end.timestamp), joints=int(valid.sum()), speed=speed))
    pairs = result['pairs']
    enough = len(pairs) >= 3 and pairs[-1]['end_time']-pairs[0]['start_time'] >= .3
    if enough:
        result['speed'] = float(np.mean([p['speed'] for p in pairs]))
    elif min_speed > 0 and missing_policy == 'bypass':
        result.update(reason='insufficient_pose_bypass',unique_frames=len(unique),valid_pairs=len(pairs))
        return result
    result.update(applied=min_speed > 0, supported=(min_speed == 0 or (enough and result['speed'] >= min_speed)),
        unique_frames=len(unique), valid_pairs=len(pairs))
    result['reason'] = ('disabled' if min_speed == 0 else 'insufficient_reliable_pose' if not enough else
                        'speed_supported' if result['supported'] else 'below_speed_threshold')
    return result


def apply_running_prior(task, row, min_speed=0., lum_gate=35., pose_enabled=True, missing_policy='bypass'):
    if row['label'] != 'chase':
        row['running_prior'] = dict(applied=False, reason='unaffected_action', label=row['label'])
        return
    evidence = speed_evidence(task.window, row.get('raw_luma'), min_speed, lum_gate, pose_enabled, missing_policy)
    row['running_prior'] = evidence
    evidence['label'] = 'chase' if evidence['supported'] else 'other'
    if evidence['applied'] and not evidence['supported']:
        from darkpipe.constants import BEHAVIORS
        row.setdefault('action_label', row['label'])
        row.setdefault('action_confidence', row['confidence'])
        row.setdefault('action_decision_scores', list(row['decision_scores']))
        row['decision_scores'] = [float(c == 'other') for c in BEHAVIORS]
        row['label'], row['confidence'] = 'other', 1.
