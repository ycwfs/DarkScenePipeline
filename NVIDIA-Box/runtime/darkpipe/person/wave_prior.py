"""Same-person wave support from real, deduplicated COCO-17 pose observations.

This is a necessary geometric condition, not a wave classifier. A raised hand held
still can satisfy it. Unknown/dark raw brightness and disabled pose skip the rule;
bright windows with insufficient reliable wrist/shoulder evidence reject wave.
"""
import math


def _bright(luma, gate: float) -> bool:
    return gate == 0 or (luma is not None and math.isfinite(luma) and luma >= gate)


def wave_evidence(samples, raw_luma, min_frames: int = 3, lum_gate: float = 35.,
                  keypoint_conf: float = .5, consecutive: bool = False,
                  pose_enabled: bool = True) -> dict:
    if isinstance(min_frames, bool) or not isinstance(min_frames, int) or not 0 <= min_frames <= 32:
        raise ValueError('min_frames must be an integer within [0, 32]')
    if not math.isfinite(lum_gate) or lum_gate < 0 or not 0 < keypoint_conf <= 1:
        raise ValueError('Invalid brightness/keypoint confidence threshold')
    result = dict(applied=False, supported=True, reason='disabled', min_frames=min_frames,
        keypoint_conf=keypoint_conf, lum_gate=lum_gate, consecutive=consecutive,
        rule='same_wrist_y_less_than_both_shoulders', raw_luma=raw_luma,
        counts=[0, 0], longest_runs=[0, 0], reliable_counts=[0, 0], samples=[])
    if min_frames == 0:
        return result
    if not pose_enabled:
        result['reason'] = 'pose_branch_disabled'
        return result
    if not _bright(raw_luma, lum_gate):
        result['reason'] = 'unknown_raw_luma' if raw_luma is None or not math.isfinite(raw_luma) else 'below_luma_gate'
        return result
    seen = set()
    run = [0, 0]
    last_time = None
    for sample in samples:
        frame = int(sample.frame_index)
        if frame in seen:
            continue  # 25fps -> 32 samples contains duplicates; these are not extra evidence.
        seen.add(frame)
        stamp = float(sample.timestamp)
        if not math.isfinite(stamp) or (last_time is not None and stamp <= last_time):
            raise ValueError('Unique source frames must have increasing finite timestamps')
        gap = None if last_time is None else stamp-last_time
        last_time = stamp
        points = sample.pose
        if len(points) != 17 or any(len(p) != 3 for p in points):
            raise ValueError('Expected normalized COCO17 x/y/confidence keypoints')
        joints = [[float(v) for v in points[j]] for j in (5, 6, 9, 10)]
        good = [all(math.isfinite(v) for v in p) and 0 <= p[0] <= 1 and 0 <= p[1] <= 1
                and keypoint_conf <= p[2] <= 1 for p in joints]
        bright = _bright(sample.luma, lum_gate)
        usable = [bright and good[0] and good[1] and good[j+2] for j in range(2)]
        above = [usable[j] and joints[j+2][1] < min(joints[0][1], joints[1][1]) for j in range(2)]
        for j in range(2):
            result['reliable_counts'][j] += int(usable[j])
            result['counts'][j] += int(above[j])
            run[j] = (run[j]+1 if gap is not None and gap <= .12 else 1) if above[j] else 0
            result['longest_runs'][j] = max(result['longest_runs'][j], run[j])
        # Keep nonfinite coordinates out of JSON while retaining failed-observation evidence.
        clean = [[v if math.isfinite(v) else None for v in point] for point in joints]
        result['samples'].append(dict(frame=frame, timestamp=stamp, bright=bright,
            reliable=usable, above_both=above, shoulders_wrists=clean))
    counts = result['longest_runs'] if consecutive else result['counts']
    result.update(applied=True, supported=max(counts) >= min_frames,
                  unique_frames=len(seen), support_wrist=('left' if counts[0] >= min_frames else
                      'right' if counts[1] >= min_frames else None))
    result['reason'] = ('wrist_above_both_shoulders' if result['supported'] else
        'insufficient_reliable_pose' if max(result['reliable_counts']) < min_frames else 'insufficient_raised_wrist_frames')
    return result


def apply_wave_prior(task, row: dict, min_frames: int = 3, lum_gate: float = 35.,
                     keypoint_conf: float = .5, consecutive: bool = False,
                     pose_enabled: bool = True) -> None:
    """Apply after fusion/object policy but before display, event and clip publication."""
    if row['label'] != 'wave':
        row['wave_prior'] = dict(applied=False, reason='unaffected_action', label=row['label'])
        return
    result = wave_evidence(task.window, row.get('raw_luma'), min_frames, lum_gate,
                           keypoint_conf, consecutive, pose_enabled)
    result['label'] = 'wave' if result['supported'] else 'other'
    row['wave_prior'] = result
    if result['applied'] and not result['supported']:
        row.setdefault('action_label', row['label'])
        row.setdefault('action_confidence', row['confidence'])
        row.setdefault('action_decision_scores', list(row['decision_scores']))
        from darkpipe.constants import BEHAVIORS
        row['decision_scores'] = [float(c == 'other') for c in BEHAVIORS]
        row['label'], row['confidence'] = 'other', 1.
