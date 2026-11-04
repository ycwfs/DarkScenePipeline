"""Real H.264 boundary checks for concurrent per-person recording."""
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'platform/dark-demo-v2'))
sys.path.insert(0, str(ROOT/'reports/person-stream-audit-20261023/scripts'))
from person_clips import PersonClips


def event(t, person=1, label='Drinking water', epoch=0, conf=.95):
    return dict(session=1, track_id=person, history_generation=epoch << 32,
                timestamp=t, frame_index=round(t*10), label=label, confidence=conf)


def image(index):
    result = np.full((90, 160, 3), 50, np.uint8)
    for bit in range(12):
        result[:16, bit*12:(bit+1)*12] = 255 if (index >> bit)&1 else 0
    return result


def decoded_indices(path):
    cap = cv2.VideoCapture(str(path))
    values = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        values.append(sum(int(frame[4:12,bit*12+3:bit*12+9].mean()>127) << bit for bit in range(12)))
    cap.release()
    assert values
    return values


def replay(tmp_path, rows, spans=((0,120,0),)):
    rec = PersonClips(tmp_path, time.monotonic()+1000, denoise='off',
                      queue_frames=2000, writer_queue=2000)
    for row in rows:
        rec.event(row)
    for start, stop, epoch in spans:
        for i in range(start,stop):
            rec.frame(image(i),dict(frame=i,timestamp=i/10,capture_epoch=epoch))
    result = rec.close()
    assert result.get('late_events',0) == result.get('dispatcher_frames_dropped',0) == 0
    assert all(result['totals'][key] == 0 for key in (
        'clips_abandoned','clips_forced','frames_dropped','clips_unfinished','clips_failed','write_errors'))
    for clip in result['clips']:
        frames = decoded_indices(clip['video'])
        assert abs(frames[0]/10-clip['start_source_time']) <= .12
        assert abs(frames[-1]/10-clip['end_source_time']) <= .22
        assert f"T{clip['person_id']:03d}" in Path(clip['video']).name
    return sorted(result['clips'],key=lambda r:(r['person_id'],r['start_source_time']))


@pytest.mark.parametrize('second_label',['Waving','Drinking water'])
def test_two_people_keep_overlapping_full_pre_and_post(tmp_path,second_label):
    rows=[event(i/10)for i in range(30,61,5)]
    rows += [event(i/10,2,second_label)for i in range(40,81,5)]
    clips=replay(tmp_path,rows)
    assert len(clips)==2
    assert clips[0]['start_source_time']==pytest.approx(1.,abs=.11)
    assert clips[0]['end_source_time']==pytest.approx(8.1,abs=.11)
    assert clips[1]['start_source_time']==pytest.approx(2.,abs=.11)
    assert clips[1]['end_source_time']==pytest.approx(10.1,abs=.11)


def test_switching_action_does_not_cut_previous_tail(tmp_path):
    clips=replay(tmp_path,[event(3),event(3.5),event(4),event(4.5,label='Waving'),event(5,label='Waving')])
    assert len(clips)==2
    drink=next(c for c in clips if c['action_key']=='drinking_water')
    wave=next(c for c in clips if c['action_key']=='waving')
    assert drink['end_source_time']>=6
    assert wave['start_source_time']<drink['end_source_time']


def test_long_action_splits_with_overlap_and_unique_names(tmp_path):
    clips=replay(tmp_path,[event(i/10)for i in range(30,231,5)],((0,260,0),))
    assert len(clips)>=2
    assert len({c['video']for c in clips})==len(clips)
    assert clips[0]['start_source_time'] <= 1.1
    assert clips[-1]['end_source_time'] >= 25.
    for a,b in zip(clips,clips[1:]):
        assert b['start_source_time'] <= a['end_source_time']
    assert all(c['duration_seconds']<=15.11 for c in clips)


def test_disconnect_closes_old_epoch_without_inventing_missing_frames(tmp_path):
    rows=[event(3),event(4),event(12,epoch=1),event(13,epoch=1)]
    clips=replay(tmp_path,rows,((0,51,0),(100,161,1)))
    assert len(clips)==2
    assert clips[0]['end_source_time']<=5.01
    assert '断流' in clips[0]['closed_because']
    assert clips[1]['capture_epoch']==1 and clips[1]['start_source_time']>=10.


def test_other_and_subthreshold_actions_do_not_save(tmp_path):
    assert replay(tmp_path,[event(3,label='Other'),event(4,conf=.749)])==[]


def test_service_stop_marks_incomplete_tail(tmp_path):
    clips=replay(tmp_path,[event(3),event(4)],((0,46,0),))
    assert len(clips)==1
    assert clips[0]['end_source_time']==pytest.approx(4.5,abs=.11)
    assert '服务停止' in clips[0]['closed_because']


def test_eight_simultaneous_same_actions_have_unique_files(tmp_path):
    clips=replay(tmp_path,[event(t,person)for person in range(1,9)for t in (3,3.5,4,4.5)])
    assert len(clips)==8 and len({c['video']for c in clips})==8
    assert {c['person_id']for c in clips}==set(range(1,9))


def test_delayed_result_is_attached_to_its_source_time(tmp_path):
    origin=time.monotonic()
    rec=PersonClips(tmp_path,origin,pre=.2,post=.2,lookahead=.35,denoise='off')
    for i in range(7):
        rec.frame(image(i),dict(frame=i,timestamp=i/20,capture_epoch=0))
    # Decision arrives after source frame 2, before the recording lookahead commits it.
    time.sleep(.15)
    rec.event(event(.1))
    time.sleep(.55)
    result=rec.close()
    assert result.get('late_events',0)==0
    assert len(result['clips'])==1
    assert result['clips'][0]['trigger_stream_time']==pytest.approx(.1,abs=.06)


def test_late_result_is_counted_not_silently_attached_to_new_video(tmp_path):
    rec=PersonClips(tmp_path,time.monotonic()-1,lookahead=.01,denoise='off')
    rec.frame(image(4),dict(frame=4,timestamp=.2,capture_epoch=0))
    deadline=time.monotonic()+2
    while rec.watermark<.2 and time.monotonic()<deadline:time.sleep(.01)
    assert rec.watermark==.2
    rec.event(event(.1))
    result=rec.close()
    assert result.get('late_action_events')==1 and not result['clips']
