"""Platform adapters: bounded output, reconnectable capture and person/action clips."""
from collections import deque
import copy
import json
import os
from pathlib import Path
import queue
import threading
import time

import cv2
import numpy as np
import torch

from .clips import PersonClips
from .core import render_people
from .object_overlay import ObjectOverlay
from .runtime import PersonEngine
from darkpipe.media import VideoWriter


class PersonEvent:
    def __init__(self, row):
        self.row = dict(row, schema_version=3, recognition_scope='person')
        self.label = row['label']

    def to_dict(self):
        return self.row.copy()


def make_clips(cfg, origin, on_saved=None, offline=False):
    if not cfg.clip_dir:
        return None
    # Recovered events intentionally carry their model confidence (0.85–0.925 by
    # default). Lowering the recorder floor only when the feature is enabled lets those
    # events create/extend clips; unrecovered candidates are still relabeled ``other``
    # before they reach this path.
    event_floor = (cfg.candidate_recheck_lower if cfg.candidate_recheck_enabled
                   else cfg.reco_min_conf)
    return PersonClips(Path(cfg.clip_dir)/cfg.clip_session, origin,
        pre=cfg.clip_pre_sec, post=cfg.clip_post_sec, maximum=cfg.clip_max_sec,
        denoise=cfg.clip_denoise, threshold=max(event_floor, cfg.clip_min_conf),
        on_saved=on_saved, skip_labels=[v.strip() for v in cfg.clip_skip_labels.split(',') if v.strip()],
        offline=offline, pose_overlay=cfg.pose_overlay, writer_queue=0 if offline else 64)


class Output:
    def __init__(self, engine, callback, writer=None):
        self.engine, self.callback, self.writer = engine, callback, writer
        self.queue = queue.Queue(maxsize=2)
        self.error = None
        self.count = 0
        self.objects = ObjectOverlay()
        self.thread = threading.Thread(target=self._run, daemon=True, name='person-output')
        self.thread.start()

    def put(self, item):
        while True:
            if self.error is not None:
                raise RuntimeError('Person output failed') from self.error
            if not self.thread.is_alive():
                raise RuntimeError('Person output exited')
            try:
                self.queue.put(item, timeout=.1)
                return
            except queue.Full:
                pass

    def _run(self):
        waiting = deque()
        offline = not getattr(self.engine, 'realtime', True)
        def show(item):
            frame, tracks, record = item
            clean = self.engine.picture(frame)
            rows = self.engine.overlay_evidence() if offline else ()
            boxes = self.objects.update(frame, tracks, record, rows)
            record['object_overlays'] = boxes
            shown = render_people(clean, tracks, frame.shape, record['timestamp'], self.engine.cfg.pose_overlay, boxes)
            if self.writer is not None:
                self.writer.write(shown)
            self.callback(shown, clean, record)
            self.count += 1
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    while waiting:
                        show(waiting.popleft())
                    break
                if offline:
                    # File replay can draw the five historical detections on their actual
                    # source frames. Live output never incurs this visualization lookahead.
                    waiting.append(item)
                    while waiting and (item[2]['timestamp']-waiting[0][2]['timestamp'] >= 1.2 or len(waiting) > 64):
                        show(waiting.popleft())
                else:
                    show(item)
        except Exception as exc:
            self.error = exc
        finally:
            if self.writer is not None:
                try:
                    self.writer.close()
                except Exception as exc:
                    self.error = self.error or exc

    def close(self):
        if self.thread.is_alive():
            self.put(None)
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError('Person output did not stop')
        if self.error is not None:
            raise RuntimeError('Person output failed') from self.error


def capture_loop(st):
    cfg = st.cfg
    epoch, index, backoff = -1, 0, .5
    while not st.stop.is_set():
        cap = None
        try:
            cap = cv2.VideoCapture(str(cfg.input), cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000])
            if not cap.isOpened():
                raise RuntimeError('stream open failed')
            st.src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.
            st.src_fps_known.set()
            epoch += 1
            st.reconnects = epoch
            window, count = time.monotonic(), 0
            while not st.stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('stream read failed or timed out')
                now = time.monotonic()
                st.capture_alive = True
                st.raw.put((frame, index, now-st.person_origin, epoch, st.src_fps))
                index += 1
                st.frames_in = index
                count += 1
                if now-window >= 2.:
                    st.fps_in = count/(now-window)
                    count, window = 0, now
                backoff = .5
        except Exception as exc:
            st.capture_alive = False
            if not st.stop.is_set():
                print(f'[capture] {exc}，{backoff:g} 秒后重连', flush=True)
                st.stop.wait(backoff)
                backoff = min(4., backoff*2)
        finally:
            if cap is not None:
                cap.release()
    st.capture_alive = False


def process_loop(st):
    engine = output = None
    st.processing_alive = True
    try:
        torch.set_num_threads(2)
        cv2.setNumThreads(2)
        def on_event(row):
            event = PersonEvent(row)
            st.last_event = event
            st.events_total += 1
            st.bus.publish(event)
            if st.eventlog is not None:
                st.eventlog.write(event)
            if st.clipper is not None:
                st.clipper.event(row)
        engine = PersonEngine(st.cfg, st.person_origin, on_event, realtime=True)
        window, window_frames = time.monotonic(), 0
        output_shape = [404, 720, 3]
        def on_output(shown, clean, record):
            nonlocal window, window_frames
            if st.clipper is not None:
                st.clipper.frame(clean, record)
            ok, jpg = cv2.imencode('.jpg', shown, [cv2.IMWRITE_JPEG_QUALITY, st.cfg.jpeg_quality])
            if not ok:
                raise RuntimeError('JPEG encode failed')
            st.jpeg.put(jpg.tobytes())
            output_shape[:] = shown.shape
            st.frames_proc += 1
            now = time.monotonic()
            st.last_frame_at = time.time()
            st.latency_ms = max(0., now-st.person_origin-record['timestamp'])*1000
            window_frames += 1
            if now-window >= 2.:
                st.fps_proc = window_frames/(now-window)
                window, window_frames = now, 0
            if not st.processing_ready:
                st.processing_ready = True
                if st.on_ready:
                    st.on_ready('逐人管线首帧输出')
        writer = VideoWriter(st.cfg.record, fps=st.cfg.max_stream_fps) if st.cfg.record else None
        output = Output(engine, on_output, writer)
        sequence, warning_at, report_at = 0, 0., time.monotonic()
        while not st.stop.is_set():
            raw, current = st.raw.get()
            engine.collect()
            if st.clipper is not None:
                st.clipper.check()
            if output.error:
                raise RuntimeError('Person output failed') from output.error
            now = time.monotonic()
            stale = raw is None or now-st.person_origin-raw[2] > 2.
            if stale and now-warning_at >= 1.:
                warning = np.zeros(tuple(output_shape), np.uint8)
                cv2.putText(warning, 'INPUT LOST - RECONNECTING', (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 180, 255), 2)
                ok, jpg = cv2.imencode('.jpg', warning)
                if ok:
                    st.jpeg.put(jpg.tobytes())
                warning_at = now
            if engine.front_process is not None:
                if raw is not None and current != sequence and not stale:
                    if engine.front_process.submit(raw):
                        sequence = current
                result = engine.front_process.collect()
                if result is None:
                    st.stop.wait(.001)
                    continue
                metadata, prepared = result
                output.put(engine.process(None, *metadata, prepared=prepared))
            else:
                if raw is None or current == sequence or stale:
                    st.stop.wait(.005)
                    continue
                sequence = current
                output.put(engine.process(*raw))
            if now-report_at >= 5.:
                st.person_stats = engine.snapshot()
                print(f'[person] frames={st.frames_proc} proc_fps={st.fps_proc:.2f} '
                      f'latency={st.latency_ms:.1f}ms events={st.events_total} '
                      f'coalesced={st.person_stats["coalesced"]}', flush=True)
                report_at = now
        engine.collect(drain=True)
    except BaseException as exc:
        st.fail_processing('person-pipeline', exc)
    finally:
        if engine is not None:
            st.person_stats = engine.snapshot()
        for resource in (output, engine):
            if resource is not None:
                try:
                    resource.close()
                except BaseException as exc:
                    st.fail_processing('person-close', exc)
        st.processing_alive = False


def run_offline(cfg, on_ready=None):
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    cap = cv2.VideoCapture(cfg.input)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot read {cfg.input}')
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.
    origin = time.monotonic()
    clips = make_clips(cfg, origin, offline=True)
    rows, engine, output = [], None, None
    def on_event(row):
        rows.append(PersonEvent(row).to_dict())
        if clips is not None:
            clips.event(row)
    started, index = time.monotonic(), 0
    try:
        engine = PersonEngine(cfg, origin, on_event, realtime=False)
        def on_output(shown, clean, record):
            if clips is not None:
                clips.frame(clean, record)
        output = Output(engine, on_output, VideoWriter(cfg.output, fps=fps))
        started = time.monotonic()
        while cfg.max_frames is None or index < cfg.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            output.put(engine.process(frame, index, index/fps, 0, fps))
            if index == 0 and on_ready:
                on_ready('逐人离线首帧完成')
            index += 1
        engine.collect(drain=True)
    finally:
        cap.release()
        # Finish output, clips and models independently; retain the first real failure.
        error = None
        for resource in (output, clips, engine):
            if resource is not None:
                try:
                    resource.close()
                except BaseException as exc:
                    error = error or exc
        cfg.stats.update(frames=index, seconds=time.monotonic()-started,
                         object_overlay=dict(output.objects.stats) if output else {},
                         fps=index/max(time.monotonic()-started, 1e-6),
                         clips=clips.stats() if clips else {},
                         person=engine.snapshot() if engine else {})
        if error:
            raise error
    if cfg.events_json:
        Path(cfg.events_json).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.events_json).write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n')
    return rows
