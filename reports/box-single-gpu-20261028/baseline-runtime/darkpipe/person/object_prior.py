"""Single-frame support policy shared by evaluation and the production PersonEngine.

Class ids belong to the separate seven-class object model, never the person model.
Call allows_object_prior BEFORE running the detector to avoid work in dark windows.
"""
import math


def allows_object_prior(raw_luma: float | None, lum_gate: float = 35.) -> bool:
    if not math.isfinite(lum_gate) or lum_gate < 0:
        raise ValueError('lum_gate must be finite and nonnegative')
    if lum_gate == 0:
        return True
    return raw_luma is not None and math.isfinite(raw_luma) and raw_luma >= lum_gate


def filter_object_evidence(label: str, samples: list[dict], raw_luma: float | None,
                           object_min_conf: float = .581, lum_gate: float = 35.) -> dict:
    """One compatible detection in a valid actor ROI supports the existing action.

Missing brightness or all unavailable/ambiguous samples abstain. A valid, bright
observation with no compatible object rejects. This is positive support only;
it does not introduce single-frame phone/exhibit vetoes.
"""
    if not 0 <= object_min_conf <= 1:
        raise ValueError('object_min_conf must be finite and within [0, 1]')
    result = dict(label=label, applied=False, reason='unaffected_action', evidence=[],
                  object_min_conf=float(object_min_conf), lum_gate=float(lum_gate),
                  raw_luma=raw_luma)
    if label not in ('drink', 'pick_up'):
        return result
    if not allows_object_prior(raw_luma, lum_gate):
        result['reason'] = 'unknown_raw_luma' if raw_luma is None or not math.isfinite(raw_luma) else 'below_luma_gate'
        return result
    compatible = (0, 1) if label == 'drink' else (0, 1, 2, 3, 4, 5)
    usable = 0
    for sample in samples:
        if not sample.get('binding_valid') or not sample.get('available', True):
            continue
        if not allows_object_prior(sample.get('raw_luma', raw_luma), lum_gate):
            continue
        roi = sample.get('person_roi')
        if roi is None:
            continue
        usable += 1
        for box in sample['detections']:
            if len(box) != 6 or not all(math.isfinite(v) for v in box):
                continue
            x1, y1, x2, y2, conf, cls = box
            cx, cy = (x1+x2)/2, (y1+y2)/2
            if (int(cls) in compatible and conf >= object_min_conf and x2 > x1 and y2 > y1
                    and roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]):
                result['evidence'].append(dict(frame=sample['frame'], bbox=box[:4],
                                                confidence=conf, class_id=int(cls)))
    if usable == 0:
        result['reason'] = 'no_valid_bright_observation'
        return result
    result['applied'] = True
    result['label'] = label if result['evidence'] else 'other'
    result['reason'] = 'single_frame_support' if result['evidence'] else 'no_compatible_object'
    return result
