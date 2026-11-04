"""Compare vectorized association with the unchanged platform algorithm."""
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'NVIDIA-Box/runtime'))


def load_core(name, path):
    spec = importlib.util.spec_from_file_location('darkpipe.person.'+name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


box = load_core('_box_comparison', Path(os.environ.get('BOX_RUNTIME', str(ROOT/'NVIDIA-Box/runtime')))/'darkpipe/person/core.py')
reference = load_core('_platform_comparison', ROOT/'platform/dark-demo-v2/darkpipe/person/core.py')


def test_pose_assignment_matches_reference_with_crowds_and_low_confidence():
    rng = np.random.default_rng(28)
    for _ in range(80):
        count, nposes = rng.integers(0, 22, size=2)
        starts = rng.uniform(0, 500, (count, 2)).astype(np.float32)
        ends = starts+rng.uniform(20, 150, (count, 2)).astype(np.float32)
        boxes = np.column_stack([starts, ends, np.ones(count), np.zeros(count)]).astype(np.float32)
        points = rng.uniform(0, 640, (nposes, 17, 2)).astype(np.float32)
        for i in range(min(count, nposes)):
            points[i] = rng.uniform(starts[i], ends[i], (17, 2))
        scores = rng.random((nposes, 17)).astype(np.float32)
        actual = box.associate_poses(boxes, points, scores)
        expected = reference.associate_poses(boxes, points, scores)
        for a, b in zip(actual, expected):
            assert (a is None) == (b is None)
            if a is not None:
                np.testing.assert_array_equal(a, b)


def test_tracker_keeps_ids_velocities_and_overlap_binding():
    rng = np.random.default_rng(29)
    frame = rng.integers(0, 255, (404, 720, 3), dtype=np.uint8)
    for count in (0, 1, 4, 18):
        a, b = box.PersonTracker(), reference.PersonTracker()
        start = rng.uniform([5, 5], [520, 240], (count, 2)).astype(np.float32)
        size = rng.uniform([30, 60], [100, 150], (count, 2)).astype(np.float32)
        for index in range(30):
            offset = rng.uniform(-4, 4, (count, 2)).astype(np.float32)
            detections = np.column_stack([start+offset, start+offset+size,
                                         np.full(count, .9), np.zeros(count)]).astype(np.float32)
            if index % 7 == 0:
                detections = detections[1:]
            poses = [None]*len(detections)
            first = a.update(detections, poses, frame, index*.1)
            second = b.update(detections, poses, frame, index*.1)
            assert [t.tid for t in first] == [t.tid for t in second]
            for x, y in zip(first, second):
                np.testing.assert_array_equal(x.velocity, y.velocity)
            rects = np.asarray([t.box for t in first]).reshape(-1, 4)
            overlaps = box.pairwise_iou(rects, rects)
            for i, x in enumerate(first):
                assert any(overlaps[i, j] > .4 for j in range(len(first)) if i != j) == \
                       any(reference.iou(x.box, y.box) > .4 for y in first if y is not x)


def test_trt_failure_restores_process_locale(monkeypatch):
    import locale
    import pytest
    from darkpipe.box_onnx import TensorRTSession
    previous = locale.setlocale(locale.LC_ALL)
    def fail(self, path, device):
        locale.setlocale(locale.LC_ALL, 'C')
        raise RuntimeError('test build failure')
    monkeypatch.setattr(TensorRTSession, '_initialize', fail)
    with pytest.raises(RuntimeError, match='test build failure'):
        TensorRTSession('model.onnx', 'cuda:0')
    assert locale.setlocale(locale.LC_ALL) == previous
