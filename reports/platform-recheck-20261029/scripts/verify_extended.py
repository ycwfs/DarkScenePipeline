"""Additional source-time clip, repeated-scene, regression and offline checks."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
from analyze_live import stats, video

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
sys.path.insert(0, str(OUT / 'release-extracted/dark-demo-v2'))
from darkpipe.clips import label_key
from darkpipe.constants import BEHAVIOR_DISPLAY


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def save(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def clips_against_events(folder):
    events = rows(next((folder / 'clips').rglob('events.jsonl')))
    by_person = defaultdict(list)
    for e in events:
        if e['label'] != 'other':
            assert e['confidence'] >= e['threshold'] == .925
            key = label_key(BEHAVIOR_DISPLAY.get(e['label'], e['label']))
            by_person[(e['track_id'], e['history_generation'], key)].append(e)
    records = []
    for path in (folder / 'clips').rglob('*.json'):
        if not path.with_suffix('.mp4').exists():
            continue
        c = read(path)
        events_here = by_person[(c['person_id'], c['history_generation'], c['action_key'])]
        # Clip JSON rounds trigger time to milliseconds; source event is full precision.
        triggers = [e for e in events_here if abs(e['timestamp'] - c['trigger_stream_time']) <= .002]
        assert triggers, ('clip has no matching final person/action event', path)
        assert c['action_key'] == c['label_key'] == path.parent.name
        assert f"T{c['person_id']:03d}" in path.name
        assert c['capture_epoch'] == c['history_generation'] >> 32
        assert c['writer_frames_dropped'] == 0
        assert c['requested_pre'] == c['requested_post'] == 2
        assert c['start_source_time'] <= c['trigger_stream_time'] <= c['end_source_time']
        assert c['duration_seconds'] < 15.51, (path, c['duration_seconds'])
        expected_complete = c['pre_context_available'] >= 1.85 and c['post_context_available'] >= 1.85
        assert c['context_complete'] == expected_complete
        records.append(dict(path=str(path.with_suffix('.mp4').relative_to(OUT)),
            metadata=str(path.relative_to(OUT)), person_id=c['person_id'], generation=c['history_generation'],
            action=c['action_key'], start=c['start_source_time'], end=c['end_source_time'],
            trigger=c['trigger_stream_time'], frames=c['frames_written'],
            pre=c['pre_context_available'], post=c['post_context_available'],
            complete=c['context_complete'], closed_because=c['closed_because']))
    overlaps = []
    for i, a in enumerate(records):
        for b in records[i+1:]:
            if a['person_id'] == b['person_id'] and a['generation'] == b['generation']:
                continue
            amount = min(a['end'], b['end']) - max(a['start'], b['start'])
            if amount > .05:
                overlaps.append(dict(first=a['path'], second=b['path'], seconds=amount,
                                     different_actions=a['action'] != b['action']))
    split_continuations = []
    for a in records:
        if 'clip_max_sec' not in a['closed_because']:
            continue
        successors = sorted((b for b in records if b['start'] > a['start'] and
            (b['person_id'], b['generation'], b['action']) == (a['person_id'], a['generation'], a['action'])),
            key=lambda b:b['start'])
        # This recording has continuing triggers; preserve whether the successor overlaps.
        if successors:
            b = successors[0]
            split_continuations.append(dict(first=a['path'], next=b['path'], overlap_seconds=a['end']-b['start']))
    return dict(passed=True, clips=len(records), complete=sum(c['complete'] for c in records),
                every_clip_matches_final_person_action_event=True, records=records,
                overlapping_person_clip_pairs=overlaps, max_length_continuations=split_continuations)


def live():
    folder = OUT / 'live'
    timeline = read(folder/'source-timeline.json')
    start = timeline['source_start_monotonic']
    end = timeline['source_end_monotonic']
    frames = rows(folder/'frames.jsonl')
    health = read(folder/'health-history.json')
    events = rows(next((folder/'clips').rglob('events.jsonl')))
    origin = frames[0]['origin']
    cycles = []
    for cycle in range(3):
        left, right = start+cycle*300, start+(cycle+1)*300
        f = [r for r in frames if left <= r['origin']+r['timestamp'] < right]
        h = [r['health'] for r in health if left <= r['monotonic'] < right and 'health' in r]
        e = [r for r in events if left <= origin+r['timestamp'] < right]
        cycles.append(dict(cycle=cycle+1, frames=len(f), average_processed_fps=len(f)/300,
            people_max=max(r['people'] for r in f), health_fps=stats([r['fps_proc'] for r in h]),
            action_age=stats([r['decision_age'] for r in e]),
            parent_rss_mib=stats([r['rss_bytes']/1024**2 for r in f]),
            parent_rss_final_scene_mib=stats([r['rss_bytes']/1024**2 for r in f if origin+r['timestamp'] >= right-30]),
            output_interval=stats(np.diff([r['output_monotonic'] for r in f])),
            labels=dict(Counter(r['label'] for r in e)), events=len(e)))
    session = read(folder/'session.json')
    clip_stats = session['clips']
    zeros = ('clips_failed', 'clips_forced', 'clips_abandoned', 'clips_unfinished', 'write_errors',
             'frames_dropped', 'dispatcher_frames_dropped', 'events_queue_dropped',
             'late_action_events', 'person_uploads_dropped')
    for key in zeros:
        assert clip_stats.get(key, 0) == 0, (key, clip_stats.get(key))
    assert session['event_log_stats']['events_dropped'] == 0
    assert clip_stats['writer_alive'] is False and clip_stats['uploads_pending'] == 0
    assert session['person']['action_workers']
    for worker in session['person']['action_workers']:
        assert worker['queue'].get('dropped', 0) == 0
    snapshots = [f['engine'] for f in frames if f.get('engine')]
    assert max(s['object_history_bytes'] for s in snapshots) <= 128*1024**2
    degraded = [h for h in health if start <= h['monotonic'] <= end and h.get('health',{}).get('status') != 'ok']
    details = []
    for row in degraded:
        h = row.get('health', {})
        details.append(dict(time=row['monotonic']-start, http_status=row.get('http_status'),
            status=h.get('status'), processing=h.get('processing_status'), frame_age=h.get('frame_age_s'),
            degraded_reason=h.get('degraded_reason'),
            output={k:v for k,v in h.items() if k.startswith(('push_', 'hls_', 'capture_'))}))
    gpu = defaultdict(list)
    with (OUT/'live-gpu.csv').open() as stream:
        for row in csv.DictReader(stream):
            row = {k.strip():v.strip() for k,v in row.items()}
            gpu[row['index']].append(float(row['memory.used [MiB]'].split()[0]))
    clip_result = clips_against_events(folder)
    save('live-clips-check.json', clip_result)
    complete = read(folder/'complete.json')
    assert len(complete['http_clients']) == 18 and all(v == 0 for v in complete['http_clients'].values())
    assert read(folder/'cleanup.json')['owned_processes_alive'] == 0
    sse = read(folder/'sse-status.json')
    assert sse['events'] > 0 and sse['thread_stopped']
    result = dict(passed=True, seconds=end-start, cycles=cycles,
                  gpu_memory_mib={key:stats(value) for key,value in gpu.items()},
                  final_zero_clip_counters={key:clip_stats.get(key,0) for key in zeros},
                  late_events=clip_stats.get('late_events',0),
                  events_without_output_frame=clip_stats.get('events_without_output_frame',0),
                  event_log=session['event_log_stats'],
                  clips=clip_result['clips'], complete_clips=clip_result['complete'],
                  overlapping_pairs=len(clip_result['overlapping_person_clip_pairs']),
                  different_action_overlapping_pairs=sum(p['different_actions'] for p in clip_result['overlapping_person_clip_pairs']),
                  degraded_details=details, http_clients=complete['http_clients'], sse=sse,
                  bounded_object_history_bytes=max(s['object_history_bytes'] for s in snapshots),
                  concurrent_test_commands={stage:read(OUT/f'{stage}-command.json') for stage in ('cpu','demos')},
                  shared_host_limitation='Live uses GPU0/1/2, offline sample regression GPU4. CPU regression and offline inference may share CPU/RAM/IO with live; this is a stability rerun, not an isolated throughput comparison.',
                  memory_limitation='RSS covers parent only; CUDA allocation statistics include caches. Three finite cycles cannot prove absence of a slow leak.')
    save('live-extended-verification.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('degraded_details','cycles','http_clients','gpu_memory_mib')}, indent=2))


def demos():
    inputs = read(OUT/'inputs.json')
    old = {r['input']['id']:r for r in read(ROOT/'reports/full-prior-release-20261027/demo-results.json')}
    results, videos = [], []
    for item in inputs:
        dest = OUT/'demos'/item['id']
        summary, frames, events = read(dest/'summary.json'), rows(dest/'frames.jsonl'), read(dest/'events.json')
        assert len(frames) == summary['stats']['frames']
        assert all(r['label'] == r['running_prior']['label'] for r in events)
        assert all(r['label'] == 'other' or r['confidence'] >= .925 for r in events)
        for e in events:
            for key in ('object_prior', 'wave_prior', 'running_prior'):
                prior = e[key]
                if prior['applied']:
                    assert e['raw_luma'] is not None and e['raw_luma'] >= 35
        counts = summary['stats']['clips']
        for key in ('clips_failed','clips_forced','clips_abandoned','frames_dropped','write_errors','clips_unfinished'):
            assert counts.get(key,0) == 0, (item['id'], key)
        clips = [p for p in (dest/'clips').rglob('*.mp4')]
        videos += [dest/'output.mp4', *clips]
        previous_events = read(ROOT/'reports/full-prior-release-20261027/demos'/item['id']/'events.json')
        prev = {(e['track_id'], e['history_generation'], tuple(e['source_frames'])):e for e in previous_events}
        matched = [(e,prev[(e['track_id'],e['history_generation'],tuple(e['source_frames']))]) for e in events
                   if (e['track_id'],e['history_generation'],tuple(e['source_frames'])) in prev]
        changes = [dict(track=e['track_id'], timestamp=e['timestamp'], old=p['label'], new=e['label'])
                   for e,p in matched if e['label'] != p['label']]
        luma = [r['raw_luma'] for r in frames if r['raw_luma'] is not None]
        results.append(dict(id=item['id'], title=item['title'], input=item['input'], frames=len(frames),
            events=len(events), people_max=summary['people_max'], raw_luma=stats(luma), labels=summary['labels'],
            prior_reasons=summary['prior_reasons'], clips=[str(p.relative_to(OUT)) for p in clips],
            old_labels=old[item['id']]['labels'], old_events=len(previous_events), matched_windows=len(matched),
            changed_matched_labels=changes))
    with ThreadPoolExecutor(max_workers=2) as pool:
        checks = list(pool.map(video, videos))
    assert all(c['passed'] for c in checks)
    # Independent decoded frame counts guard against silently truncated complete outputs.
    for r in results:
        path = OUT/'demos'/r['id']/'output.mp4'
        count = subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
            '-show_entries','stream=nb_read_frames','-of','default=nw=1:nk=1',str(path)],text=True)
        assert int(count.strip()) == r['frames'], r['id']
    save('demo-video-verification.json', checks)
    save('demo-results.json', results)
    save('demo-verification.json', dict(passed=True, inputs=len(results), frames=sum(r['frames'] for r in results),
        events=sum(r['events'] for r in results), clips=sum(len(r['clips']) for r in results), decoded_videos=len(checks),
        matched_windows=sum(r['matched_windows'] for r in results),
        changed_matched_labels=sum(len(r['changed_matched_labels']) for r in results),
        limitation='Historical data, not an independent accuracy test. Exact-input window matches only for per-window comparison.'))
    print(json.dumps(read(OUT/'demo-verification.json'),indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['live','demos'])
    args = parser.parse_args()
    (live if args.stage == 'live' else demos)()


if __name__ == '__main__':
    main()
