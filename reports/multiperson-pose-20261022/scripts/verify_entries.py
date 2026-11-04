"""Check saved videos, person metadata and shutdown from the real packaged launchers."""
import json
from pathlib import Path
import subprocess

import cv2

REPORT = Path(__file__).resolve().parents[1]
results, event_sets = [], []
for name in ('platform-entry', 'box-entry'):
    session = json.loads((REPORT / name / 'session.json').read_text())
    assert session['status'] == 'finished'
    assert session['stats']['frames'] == session['stats']['frames_written'] == 160
    clips = session['clips']
    assert clips['clips_saved'] == 1 and not clips['writer_alive']
    for key in ('clips_abandoned', 'clips_forced', 'frames_dropped', 'clips_unfinished',
                'clips_failed', 'write_errors'):
        assert clips[key] == 0, (name, key)
    events = json.loads(Path(session['events_json'][0]).read_text())
    assert len(events) == 9
    for e in events:
        actors = e['actor_poses']
        assert e['skeleton_active'] and len(actors) == 2
        selected = [a for a in actors if a['selected_for_fusion']]
        assert len(selected) == 1 and selected[0]['slot'] == e['skeleton_selected_slot']
        assert selected[0]['confidence'] == max(a['confidence'] for a in actors)
    event_sets.append(events)
    media = []
    videos = session['output_video'] + [p for clip in session['clip_videos'] for p in clip]
    for video in videos:
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-count_frames', '-show_entries', 'stream=codec_name,width,height,nb_read_frames,duration',
            '-of', 'json', video]))['streams'][0]
        subprocess.run(['ffmpeg', '-v', 'error', '-i', video, '-f', 'null', '-'],
                       check=True, capture_output=True, timeout=120)
        assert probe['codec_name'] == 'h264'
        assert probe['width'] == 1440 and probe['height'] == 874
        if video == session['output_video'][0]:
            assert int(probe['nb_read_frames']) == 160
            cap = cv2.VideoCapture(video)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 75)
            ok, frame = cap.read()
            cap.release()
            assert ok
            cv2.imwrite(str(REPORT / f'{name}.jpg'), frame)
        media.append(dict(path=video, full_decode=True, **probe))
    results.append(dict(entry=name, status=session['status'], events=len(events),
                        all_events_two_people=True, all_selections_highest_person=True,
                        label_counts=session['label_counts'], clips=clips, media=media))

assert [e['label'] for e in event_sets[0]] == [e['label'] for e in event_sets[1]]
assert [e['skeleton_selected_slot'] for e in event_sets[0]] == [e['skeleton_selected_slot'] for e in event_sets[1]]
result = dict(runs=results, video_labels_match=True, selected_slots_match=True,
              lum_gate_override=0, production_default_lum_gate=75,
              physical_gpus={'platform-entry': 1, 'box-entry': 5},
              packaged_defaults_used_except=['lum_gate', 'gpu_ids', 'input/output paths', 'extra_hosts'],
              quality_clip_denoise=True, physical_nvidia_box_tested=False)
(REPORT / 'entry-verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
