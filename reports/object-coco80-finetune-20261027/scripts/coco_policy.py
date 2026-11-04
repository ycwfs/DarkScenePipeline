"""Adapt COCO IDs to the existing single-frame prior, without adding model classes."""
import types

SUPPORT_MAP = {39: 0, 41: 1, 67: 2, 73: 4, 65: 5}


def adapted_filter(original):
    def call(label, samples, raw_luma, object_min_conf=.581, lum_gate=35.):
        mapped = []
        for sample in samples:
            row = dict(sample)
            row['detections'] = [list(b[:5])+[SUPPORT_MAP.get(int(b[5]), 99)] for b in sample['detections']]
            mapped.append(row)
        result = original(label, mapped, raw_luma, object_min_conf, lum_gate)
        reverse = {v:k for k,v in SUPPORT_MAP.items()}
        for evidence in result['evidence']:
            evidence['class_id'] = reverse[evidence['class_id']]
        result['class_schema'] = 'coco80'
        result['compatible_classes'] = [39,41] if label=='drink' else list(SUPPORT_MAP) if label=='pick_up' else []
        return result
    return call


def bind_apply(prior, original_apply, filename):
    environment = dict(original_apply.__globals__)
    environment['filter_object_evidence'] = adapted_filter(environment['filter_object_evidence'])
    environment['OBJECT_MODEL'] = filename
    function = types.FunctionType(original_apply.__code__, environment, original_apply.__name__,
        original_apply.__defaults__, original_apply.__closure__)
    prior.apply = types.MethodType(function, prior)


def shift_legacy_for_display(row):
    """Keep legacy labels visible alongside COCO IDs; offset is NOT a model class."""
    prior = row['object_prior']
    prior['class_schema'] = 'legacy7_display_offset80'
    for sample in prior['samples']:
        for box in sample['detections']:
            box[5] += 80
    for evidence in prior['evidence']:
        evidence['class_id'] += 80
