"""Person-specific boundaries: swaps, vanished tracks, ROIs, gate and fused labels."""
import cv2
import numpy as np

from person_core import (BEHAVIORS, PersonTracker, associate_poses, crop_person,
                         fuse_person, person_boxes, render_people, place_labels)


def scene():
    frame = np.zeros((180, 300, 3), np.uint8)
    frame[20:160, 20:90] = (0, 0, 255)
    frame[20:160, 200:270] = (255, 0, 0)
    boxes = np.array([[20, 20, 90, 160, .9, 0], [200, 20, 270, 160, .8, 0]], np.float32)
    return frame, boxes


def test_crops_are_person_specific_and_letterboxed():
    frame, boxes = scene()
    a, _ = crop_person(frame, boxes[0, :4], margin=0)
    b, _ = crop_person(frame, boxes[1, :4], margin=0)
    assert a.shape == b.shape == (224, 400, 3)
    assert a[112, 200].tolist() == [255, 0, 0]
    assert b[112, 200].tolist() == [0, 0, 255]
    assert a[112, 0].tolist() == [114, 114, 114]


def test_person_detection_deduplicates_and_rejects_nonperson():
    _, boxes = scene()
    rows = np.vstack([boxes, boxes[0], [0, 0, 100, 150, .99, 16]])
    assert len(person_boxes(rows)) == 2


def test_detector_order_swap_preserves_histories():
    frame, boxes = scene()
    tracker = PersonTracker()
    first = tracker.update(boxes, [None]*2, frame, 0.)
    for t in first:
        t.append(frame, 0., 0, 100.)
    second = tracker.update(boxes[::-1], [None]*2, frame, .04)
    assert [t.tid for t in second] == [2, 1]
    for t in second:
        t.append(frame, .04, 1, 100.)
        assert np.array_equal(t.history[0].crop, t.history[1].crop)


def test_lost_tracks_do_not_display_and_long_gap_resets_action():
    frame, boxes = scene()
    tracker = PersonTracker()
    t = tracker.update(boxes[:1], [None], frame, 0.)[0]
    t.append(frame, 0., 0, None)
    t.prediction = {'label':'wave'}
    assert tracker.update(boxes[:0], [], frame, .1) == []
    recovered = tracker.update(boxes[:1], [None], frame, .3)[0]
    assert recovered.tid == t.tid and not recovered.history and recovered.prediction is None
    new = tracker.update(boxes[:1], [None], frame, 1.)[0]
    assert new.tid != t.tid and not new.history


def test_same_track_timestamps_supply_rgb_and_pose_window():
    frame, boxes = scene()
    tracker = PersonTracker()
    for index in range(26):
        tracks = tracker.update(boxes, [None]*2, frame, index/25)
        for t in tracks:
            t.append(frame, index/25, index, None)
    for t in tracks:
        window = t.window(25)
        assert len(window) == 32 and window[-1].frame_index == 25
        assert all(np.array_equal(s.crop, window[0].crop) for s in window)


def test_pose_assignment_is_one_to_one_and_order_independent():
    _, boxes = scene()
    left = np.column_stack([np.linspace(30, 80, 17), np.linspace(30, 150, 17)])
    right = left + [180, 0]
    poses = associate_poses(boxes, np.stack([right, left]), np.ones((2,17),np.float32))
    np.testing.assert_allclose(poses[0][:,:2], left)
    np.testing.assert_allclose(poses[1][:,:2], right)
    only_one = associate_poses(np.vstack([boxes[0], boxes[0]]), left[None], np.ones((1,17)))
    assert sum(p is not None for p in only_one) == 1


def test_person_fusion_never_uses_another_person_probability():
    wave = np.eye(10, dtype=np.float32)[0]
    other = np.eye(10, dtype=np.float32)[9]
    first = fuse_person(wave, wave, [100.]*32)
    second = fuse_person(other, other, [100.]*32)
    assert first['label'] == 'wave' and second['label'] == 'other'
    mixed = fuse_person(wave, other, [100.]*32)
    np.testing.assert_allclose(mixed['fused_probs'], .7*wave+.3*other)
    assert mixed['label'] == 'other'  # .7 is below the retained .75 threshold


def test_luma_gate_and_missing_pose_policy():
    wave, other = np.eye(10, dtype=np.float32)[[0,9]]
    low = fuse_person(wave, other, [20.]*32)
    assert low['label']=='wave' and low['mode']=='rgb_low_luma'
    unknown = fuse_person(wave, other, [None]*32)
    assert unknown['label']=='other' and unknown['mode']=='rgb_skeleton'
    missing = fuse_person(wave, None, [100.]*32)
    assert missing['mode']=='rgb_soft_no_pose'


def test_renderer_has_no_global_footer_and_does_not_mutate_inputs():
    frame, boxes = scene()
    tracker = PersonTracker()
    tracks = tracker.update(boxes, [None]*2, frame, 0.)
    original = frame.copy()
    result = render_people(frame, tracks, frame.shape, 0.)
    assert result.shape == frame.shape
    assert np.array_equal(frame, original) and not np.array_equal(frame, result)


def test_crowded_labels_are_staggered_below_boxes_without_overlap():
    patch = np.zeros((30, 150, 3), np.uint8)
    labels = [((500+i*10, 100), patch, (0,255,0)) for i in range(12)]
    placed = place_labels(labels, 1000, 800)
    rectangles = []
    for _, tile, _, x, y in placed:
        h,w = tile.shape[:2]
        assert y >= 100 and 0 <= x <= 1000-w and y+h <= 800
        for a,b,c,d in rectangles:
            assert not (x<c and x+w>a and y<d and y+h>b)
        rectangles.append((x,y,x+w,y+h))
