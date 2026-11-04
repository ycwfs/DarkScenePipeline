"""Independent full-frame clips per tracked person/action, on capture time.

Recording waits briefly for asynchronous decisions; live output never waits for this
lookahead. A bounded producer queue makes any loss explicit. Each completed file carries its own person, action and capture generation.
"""
from collections import Counter, deque
import json
from pathlib import Path
import queue
import shutil
import tempfile
import threading
import time

from darkpipe.clips import ClipRecorder, label_key
from darkpipe.stages.base import RecognitionEvent
from darkpipe.constants import BEHAVIOR_DISPLAY
from .core import render_people
from .scheduling import ResultStore


class PersonClips:
    def __init__(self, root, origin, pre=2., post=2., maximum=15., lookahead=1.2,
                 denoise='off', queue_frames=128, writer_queue=64, threshold=.75,
                 on_saved=None, skip_labels=('other',), offline=False, pose_overlay=True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.stage_dir = Path(tempfile.mkdtemp(prefix='darkpipe_person_clips_'))
        self.origin = origin
        self.threshold, self.on_saved = threshold, on_saved
        self.skip = {label_key(s) for s in skip_labels}
        self.offline, self.pose_overlay = offline, pose_overlay
        self.store = ResultStore()
        self.totals = Counter()
        self.closed = False
        self.uploads = queue.Queue(maxsize=32)
        self.upload_done = threading.Event()
        self.upload_thread = None
        if on_saved is not None:
            self.upload_thread = threading.Thread(target=self._upload, daemon=True, name='person-upload')
            self.upload_thread.start()
        self.pre, self.post, self.maximum, self.lookahead = pre, post, maximum, lookahead
        self.denoise, self.writer_queue = denoise, writer_queue
        self.frames = queue.Queue(maxsize=queue_frames)
        self.events = queue.Queue()
        self.done = threading.Event()
        self.error = None
        self.counters = Counter()
        self.saved = deque(maxlen=128)
        self.active = {}
        self.history = deque()
        self.pending = []
        self.watermark = -1.
        self.epoch = None
        self.sequence = 0
        self.last_timestamp = 0.
        self.thread = threading.Thread(target=self._run, name='person-clip-dispatch', daemon=True)
        self.thread.start()

    def check(self):
        if self.error is not None:
            raise RuntimeError('Per-person clip recording failed') from self.error

    def event(self, row):
        self.check()
        self.events.put(dict(row))

    def frame(self, image, record):
        self.check()
        try:
            if self.offline:
                while not self.done.is_set():
                    self.check()
                    try:
                        self.frames.put((image, dict(record)), timeout=.1)
                        break
                    except queue.Full:
                        pass
            else:
                self.frames.put_nowait((image, dict(record)))
        except queue.Full:
            self.counters['dispatcher_frames_dropped'] += 1

    def _saved(self, identity, recording_id, files, rel_dir, meta):
        try:
            sid, tid, generation, key = identity
            prefix = f'E{generation >> 32:03d}_T{tid:03d}_G{generation & 0xffffffff:03d}_R{recording_id:05d}_'
            video, sidecar = map(Path, files)
            new_video = video.with_name(prefix + video.name)
            new_json = sidecar.with_name(prefix + sidecar.name)
            end = meta['first_event']['clip_start_source_time'] + meta['duration_seconds']
            info = dict(meta, person_id=tid, history_generation=generation,
                        capture_epoch=generation >> 32, action_key=key, recording_id=recording_id,
                        start_source_time=meta['first_event']['clip_start_source_time'],
                        end_source_time=end, video=str(new_video),
                        pre_context_available=meta['trigger_stream_time']-meta['first_event']['clip_start_source_time'],
                        post_context_available=end-meta['first_event']['last_action_source_time'],
                        requested_pre=self.pre, requested_post=self.post,
                        recording_policy='full frame per person and action; overlapping files allowed')
            info['writer_frames_dropped'] = meta['first_event']['writer_frames_dropped']
            info['context_complete'] = (info['pre_context_available'] >= self.pre-.15 and
                info['post_context_available'] >= self.post-.15 and not info['writer_frames_dropped'])
            video.rename(new_video)
            new_json.write_text(json.dumps(info, ensure_ascii=False, indent=2)+'\n')
            sidecar.unlink()
            self.saved.append(info)
            if self.on_saved is not None:
                try:
                    self.uploads.put_nowait(([str(new_video), str(new_json)],
                        str(new_video.parent.relative_to(self.root)), info))
                except queue.Full:
                    self.counters['person_uploads_dropped'] += 1
                    print('[clip] 上传队列已满，本地片段已保留', flush=True)
        except Exception as exc:
            self.error = exc
            raise

    def _make(self, identity):
        _, tid, generation, _ = identity
        self.sequence += 1
        recording_id = self.sequence
        session = f'E{generation >> 32:03d}_T{tid:03d}_G{generation & 0xffffffff:03d}/R{recording_id:05d}'
        rec = ClipRecorder(str(self.root), pre_sec=self.pre, post_sec=self.post,
            max_sec=self.maximum, denoise=self.denoise, min_confidence=self.threshold,
            skip_labels=self.skip,
            session=session, realtime=not self.offline, stage_dir=str(self.stage_dir/f'R{recording_id:05d}'), queue_frames=self.writer_queue,
            on_saved=lambda files, rel, meta: self._saved(identity, recording_id, files, rel, meta))
        # Capture timestamps are relative, not wall-clock epochs. Record the precise source
        # range in sidecars; the base class's started_at is not used for alignment.
        original_start = rec._start
        def start(event, ts):
            original_start(event, ts)
            if rec._active is not None and rec._active['first_event'] is not None:
                rec._active['first_event']['clip_start_source_time'] = rec._active['t0']
                rec._active['first_event']['writer_drop_start'] = rec.dropped_frames
        rec._start = start
        original_finish = rec._finish
        def finish(ts, why):
            if rec._active is not None:
                rec._active['first_event']['last_action_source_time'] = rec._active['t_last_event']
                rec._active['first_event']['writer_frames_dropped'] = (
                    rec.dropped_frames-rec._active['first_event']['writer_drop_start'])
            original_finish(ts, why)
        rec._finish = finish
        return rec

    def _drain_events(self):
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            self.counters['events_received'] += 1
            if event['timestamp'] <= self.watermark:
                self.counters['late_events'] += 1
                if label_key(event['label']) != 'other':
                    self.counters['late_action_events'] += 1
                continue
            self.pending.append(event)
        self.pending.sort(key=lambda r: (r['timestamp'], r['track_id']))

    def _finish_all(self, reason):
        for rec in self.active.values():
            if rec._active is not None:
                rec._finish(self.last_timestamp, reason)
            rec.close()
            self._retired(rec.stats())
        self.active.clear()

    def _consume(self, image, record):
        stamp, epoch = record['timestamp'], record.get('capture_epoch', 0)
        if self.epoch is not None and (epoch != self.epoch or stamp-self.last_timestamp > .5):
            self._finish_all('输入断流/时间间隙，尾部上下文不完整')
            self.history.clear()
            self.counters['capture_discontinuities'] += 1
        self.epoch = epoch
        # Repaint delayed recording frames using the same capture-time decisions that
        # trigger them; never save an earlier live overlay after a newer event arrives.
        tracks = record.get('tracks')
        if tracks is not None:
            for track in tracks:
                key = record['keys'][track.tid]
                # Rows due at this frame are applied below, before rendering and recording.
                track.prediction = self.store.for_track(key, stamp)
                track.last_inference = track.prediction['timestamp'] if track.prediction else -1e9
        due, later = [], []
        for event in self.pending:
            (due if event['timestamp'] <= stamp+1e-7 else later).append(event)
        self.pending = later
        grouped = {}
        for row in due:
            if row['history_generation'] >> 32 != epoch:
                self.counters['previous_epoch_events'] += 1
                continue
            self.store.update(row)
            if label_key(row['label']) in self.skip or row['confidence'] < self.threshold:
                continue
            action_key = label_key(BEHAVIOR_DISPLAY.get(row['label'], row['label']))
            identity = (row['session'], row['track_id'], row['history_generation'], action_key)
            grouped[identity] = row
        if tracks is not None:
            for track in tracks:
                track.prediction = self.store.for_track(record['keys'][track.tid], stamp)
                track.last_inference = track.prediction['timestamp'] if track.prediction else -1e9
            image = render_people(image, tracks, record['model_shape'], stamp, self.pose_overlay)
        self.history.append((image, stamp))
        while len(self.history)>1 and stamp-self.history[0][1] > self.pre:
            self.history.popleft()
        for identity, row in grouped.items():
            if identity not in self.active:
                self.active[identity] = self._make(identity)
                for old_image, old_stamp in list(self.history)[:-1]:
                    self.active[identity].push(old_image, None, old_stamp)
        retire = []
        for identity, rec in self.active.items():
            row = grouped.get(identity)
            event = (RecognitionEvent(row['frame_index'], row['timestamp'], BEHAVIOR_DISPLAY.get(row['label'], row['label']),
                                      row['confidence']) if row else None)
            rec.push(image, event, stamp)
            if rec._active is None and rec._pending_start is None:
                retire.append(identity)
        for identity in retire:
            rec = self.active.pop(identity)
            rec.close()
            self._retired(rec.stats())
        self.watermark = self.last_timestamp = stamp
        self.counters['frames_recording_input'] += 1
        self.counters['max_recorders'] = max(self.counters['max_recorders'], len(self.active))

    def _run(self):
        waiting = deque()
        try:
            while True:
                self.check()
                self._drain_events()
                try:
                    waiting.append(self.frames.get(timeout=.02))
                except queue.Empty:
                    pass
                while waiting and (self.done.is_set() or self.offline or
                        time.monotonic()-self.origin-waiting[0][1]['timestamp'] >= self.lookahead):
                    self._drain_events()
                    self._consume(*waiting.popleft())
                if self.done.is_set() and self.frames.empty() and not waiting:
                    self._drain_events()  # a final result may arrive during the timed frame wait
                    self._finish_all('服务停止，尾部上下文可能不完整')
                    self.counters['events_without_output_frame'] += len(self.pending)
                    break
                if (not self.offline and self.active and not waiting and
                        time.monotonic()-self.origin-self.last_timestamp > self.post+self.lookahead):
                    self._finish_all('输入断流/时间间隙，尾部上下文不完整')
                    self.history.clear()
        except Exception as exc:
            self.error = exc
            try:
                self._finish_all('写盘失败')
            except Exception:
                pass

    def close(self):
        if self.closed:
            self.check()
            return self.stats()
        self.done.set()
        self.thread.join(timeout=90)
        if self.thread.is_alive():
            self.upload_done.set()
            raise RuntimeError('Per-person clip dispatcher did not stop')
        self.upload_done.set()
        if self.upload_thread is not None:
            self.upload_thread.join(timeout=65)
            if self.upload_thread.is_alive():
                self.counters['upload_shutdown_pending'] += self.uploads.qsize()+1
                print('[clip] 上传仍未结束，本地文件已保存，待重试上传', flush=True)
        self.check()
        self.closed = True
        result = self.stats()
        (self.root/'clips-summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
        if not result['clips_unfinished'] and not result['write_errors']:
            shutil.rmtree(self.stage_dir, ignore_errors=True)
        return result

    def _upload(self):
        while not self.upload_done.is_set() or not self.uploads.empty():
            try:
                item = self.uploads.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                self.on_saved(*item)
            except Exception as exc:
                self.counters['upload_callback_errors'] += 1
                print(f'[clip] 上传失败，本地文件保留：{exc}', flush=True)
            finally:
                self.uploads.task_done()

    def _retired(self, stats):
        for key in ('clips_saved','clips_abandoned','clips_forced','frames_dropped',
                    'clips_unfinished','clips_failed','write_errors','uploads_dropped'):
            self.totals[key] += stats.get(key, 0)

    def stats(self):
        stats = [rec.stats() for rec in list(self.active.values())]
        totals = {key:self.totals[key]+sum(s.get(key, 0) for s in stats) for key in (
            'clips_saved','clips_abandoned','clips_forced','frames_dropped','clips_unfinished',
            'clips_failed','write_errors','uploads_dropped')}
        return dict(**dict(self.counters), **totals, totals=totals, clips=list(self.saved),
                    writer_alive=self.thread.is_alive(),
                    recovery_stage_dir=str(self.stage_dir), uploads_pending=self.uploads.qsize(),
                    write_error=str(self.error) if self.error else None,
                    error=str(self.error) if self.error else None)
