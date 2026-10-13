"""Event-driven clip recording: cut the interesting seconds out of a live stream.

serve mode produces a continuous processed stream plus a sparse event stream; what is
actually worth keeping is the few seconds around each behaviour that is not `other`. This
module turns a (frame, event) sequence into standalone mp4 clips plus a sidecar JSON each.

Three properties drive the design and are not obvious from the outside:

  * **The GPU thread must never block on disk or network I/O.** The delivered pipeline has a
    hard <= 1 s end-to-end latency budget, so encoding happens on a writer thread behind a
    bounded queue; a full queue drops frames (loudly, counted) instead of stalling
    recognition, and a clip that cannot be opened is retried on later frames rather than
    lost. Uploading is pushed further out still, onto a thread of its own: it used to run on
    the writer thread via `on_saved`, which put an unbounded network wait exactly where the
    queue was already fullest -- see the `_up_q` comment in __init__ for what that cost.
  * **A clip needs the frames from BEFORE its trigger.** By the time an event fires, the
    recognition window it describes is already in the past — writing from the trigger
    onwards would save the aftermath, not the behaviour. A pre-roll ring buffer, bounded by
    time rather than frame count (serve mode's frame rate varies), is what fixes that.
  * **Consecutive events must not each start a clip, and a clip holds exactly one
    behaviour.** A recognizer emits one event every `stride` frames, so a five-second fall
    would otherwise produce a dozen overlapping files: an active clip is extended by further
    events *of its own label* and closed after `post_sec` without one. Extending it on any
    label instead is what produced 30 s files containing a drink, a wave and a fall in a row
    — in a scene with continuous activity some label is always firing, so the silence window
    never opens and only `max_sec` ever ends the clip. A different label therefore closes the
    current clip and opens its own (after `switch_after` events, so a boundary flicker
    between two labels does not shred one incident into a pile of files).
"""
import json
import os
import queue
import re
import shutil
import statistics
import tempfile
import threading
import time
from collections import deque

from .constants import DISPLAY_TO_ZH
from .media import VideoWriter

_SANITIZE = re.compile(r"[^a-z0-9]+")
# Post-trigger frames buffered before the mp4 is opened, so its header fps is measured from
# the clip rather than guessed.
OPEN_AFTER = 20
MAX_GAP = 1.0                    # longer than this is a stall, not a frame interval
FPS_RANGE = (5, 30)              # nominal output rates worth choosing between
MAX_DUP = 60                     # frames one stalled input frame may be stretched into
# How long a clip that could not be opened keeps trying. See _start.
START_RETRY_SEC = 3.0
# How long a clip that could not be *closed* keeps trying. See _finish/_flush_pending_end.
# Longer than START_RETRY_SEC because a lost `end` costs the whole file rather than its lead-in,
# and equal to the 10 s the close used to spend *blocking* the processing thread for -- the
# tolerance is unchanged, only who waits.
END_RETRY_SEC = 10.0
# How old a file in stage_dir must be before startup treats it as an orphan. See _clean_stage:
# the point is not the exact value but that it is far longer than any staged clip's real
# lifetime (seconds, or minutes behind a wedged NFS move), so a concurrent run is never robbed.
STAGE_ORPHAN_AGE = 3600.0


def measure_fps(times, fallback=15.0):
    """Frame rate from a list of timestamps, ignoring stalls. -> fps in [1, 60]."""
    gaps = [b - a for a, b in zip(times, times[1:]) if 0 < b - a <= MAX_GAP]
    if not gaps:
        return fallback
    return min(60.0, max(1.0, 1.0 / statistics.median(gaps)))


def label_key(label: str) -> str:
    """Display label -> filesystem-safe key: "Picking up object" -> "picking_up_object".

    Both the caller's skip list and the recognizer's labels go through this, so a skip list
    of "other" matches the event label "Other" without the caller having to know which
    spelling the recognizer emits.
    """
    return _SANITIZE.sub("_", str(label).strip().lower()).strip("_") or "unknown"


class EventLog:
    """Append every recognition event to a JSONL file next to the clips.

    Without this the only durable record of an event is the sidecar of the clip it triggered
    — which covers neither `other` nor anything filtered by `clip_skip_labels`. That is fine
    while `/events` (SSE) is reachable, and useless when it is not: the spec gives an operator
    no way to declare a published port, so a deployment where the platform does not expose one
    has the live event stream simply unavailable. The file is on the same mounted directory as
    the clips, so it survives the container either way.

    Writes go through a thread for the same reason clip encoding does: the destination is
    normally NFS, and the recognition loop must not wait on it.
    """

    def __init__(self, path, queue_size=256):
        self.path = path
        self.written = 0
        self.dropped = 0
        self._q = queue.Queue(maxsize=max(16, int(queue_size)))
        self._thread = threading.Thread(target=self._loop, daemon=True, name="event-log")
        self._thread.start()

    def write(self, event, **extra):
        rec = event.to_dict() if hasattr(event, "to_dict") else dict(event)
        rec["wall_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        rec.update(extra)
        try:
            self._q.put_nowait(rec)
        except queue.Full:
            self.dropped += 1

    def close(self):
        if self._thread.is_alive():
            self._q.put(None)
            self._thread.join(timeout=15)

    def stats(self):
        return {"path": self.path, "events_written": self.written,
                "events_dropped": self.dropped}

    def _loop(self):
        fh = None
        try:
            while True:
                rec = self._q.get()
                if rec is None:
                    break
                try:
                    if fh is None:
                        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
                        # line buffered: a killed container still leaves every event already
                        # handed over on disk, rather than a block-sized tail in a lost buffer
                        fh = open(self.path, "a", encoding="utf-8", buffering=1)
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    self.written += 1
                except OSError as e:
                    self.dropped += 1
                    if self.dropped in (1, 10) or self.dropped % 100 == 0:
                        print(f"[events] 写入 {self.path} 失败（已丢 {self.dropped} 条）: {e}")
        finally:
            if fh is not None:
                fh.close()


class ClipRecorder:
    """push() every processed frame with the event (or None) produced by that frame."""

    def __init__(self, out_dir, pre_sec=2.0, post_sec=2.0, max_sec=15.0, switch_after=2,
                 skip_labels=("other",), min_confidence=0.0, on_saved=None,
                 queue_frames=64, session=None, stage_dir=None, denoise="off",
                 naming="key", realtime=True):
        self.out_dir = out_dir
        # "key": <timestamp>_<label_key>_<seq>.mp4 under a per-label subdirectory (serve
        # mode, live wall-clock timestamps). "range": <start>-<end>s-<中文label>.mp4, flat,
        # named from the clip's own position on the source timeline (offline mode, where
        # "when did this happen" means seconds into the file, not a wall-clock moment).
        self.naming = naming
        self._used_names = set()
        # True (serve/live default): a wedged writer must not stall the live pipeline
        # forever, so the queue is bounded and a writer that falls behind costs dropped
        # frames or an abandoned clip -- see queue_frames/START_RETRY_SEC/_finish() below.
        # False (offline batch runs pass this): there is no live deadline to protect and
        # the whole point is a complete result, so the queue below is built unbounded
        # instead -- a slow writer only makes the run take longer, nothing is ever dropped.
        self.realtime = bool(realtime)
        # Denoising applied to clips ONLY, on the writer thread. The live stream and the
        # clips share one frame (server.py hands the same array to both), so anything strong
        # enough to be worth doing -- NLM at 119-376 ms/frame -- cannot go in the shared
        # path without spending the entire latency budget. Here it is already off the
        # critical path, and a clip is what someone will actually sit and study.
        self.denoise = denoise
        # Clips are encoded to local disk and moved to out_dir only once complete. out_dir is
        # normally a mounted NFS share, and writing frame-by-frame across it puts network
        # latency on the writer thread: one stall longer than the queue holds turns into
        # dropped frames. The move is a single sequential copy, off the critical path.
        # It also means a browsing user never sees a half-written mp4 growing in the folder.
        self.stage_dir = stage_dir or os.path.join(tempfile.gettempdir(), "darkpipe_clipstage")
        self.pre_sec = max(0.0, float(pre_sec))
        self.post_sec = max(0.0, float(post_sec))
        self.max_sec = max(1.0, float(max_sec))
        self.switch_after = max(1, int(switch_after))
        self.skip = {label_key(s) for s in skip_labels if str(s).strip()}
        self.min_confidence = float(min_confidence)
        self.on_saved = on_saved
        self.session = session or f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        self.root = os.path.join(out_dir, self.session)

        self._pre = []                      # [(frame, ts)] trimmed to pre_sec
        self._gaps = deque(maxlen=150)      # recent inter-frame intervals, for the fps guess
        self._last_ts = None
        self._active = None
        self._pending_start = None           # (event, deadline) -- see _start
        self._pending_end = None             # (msg, deadline, name, report) -- see _finish
        self._seq = 0
        self.saved = 0
        self.dropped_frames = 0
        self.abandoned = 0
        self.forced_close = 0
        self.upload_dropped = 0
        # Leftovers from a previous run: a crash, a kill -9, or (before force_close existed) an
        # abandoned close each leave a staged mp4 behind, and nothing ever removed them -- on a
        # long-lived container this is a slow /tmp leak of files nobody can play. Deliberately
        # age-gated rather than a clean sweep, because the dir may be shared with a live process
        # -- see _clean_stage.
        self.stage_orphans = self._clean_stage()
        # Sized to cover the writer's longest single stint: closing a clip means releasing the
        # encoder and moving two files to NFS, during which nothing is dequeued. 64 frames is
        # ~4 s of slack at the 15 fps cap, against ~2 s for the 32 it used to be; the cost is
        # ~90 MB of frame references at the delivered 920 px working size. The upload used to be
        # part of that stint too and no longer is (see _up_q below), so the 64 now covers only
        # work with a bounded cost -- it is left as it is because an NFS move has no ceiling
        # either, and 4 s of slack is what keeps a hiccup from costing the clip.
        # realtime=False (offline): maxsize=0 makes this queue unlimited, so put_nowait()/
        # put() below can never raise queue.Full -- the abandon/drop branches in _start(),
        # _flush_pending_end() and _send_frame() simply never fire. A slow writer just backs the
        # queue up; it always drains once the source file finishes feeding frames.
        self._q = queue.Queue(maxsize=max(4, int(queue_frames)) if self.realtime else 0)
        self._thread = threading.Thread(target=self._writer_loop, daemon=True,
                                        name="clip-writer")
        self._thread.start()
        # on_saved (an HDFS upload) used to run on the writer thread, right where the queue is
        # already at its fullest: 0.5 s typically but unbounded in principle, and one wedged
        # WebHDFS hop (60 s per call) is enough to make the *next* clip's close undeliverable
        # for END_RETRY_SEC -- which is exactly the abandoned close force_close() has to mop up.
        # Uploading has no deadline of its own, so it gets its own thread and its own bounded
        # queue, and the writer goes straight back to dequeuing frames.
        # Offline (realtime=False) keeps it inline: nothing sets on_saved there today --
        # pipeline.py builds the recorder without one, and offline delivery is main.py's
        # deliver() after the run, which is where a failed upload is fatal -- and with maxsize=0
        # there is no bounded queue to protect anyway. The inline branch exists so a caller that
        # does pass one keeps today's ordering instead of silently gaining a thread.
        self._up_q = queue.Queue(maxsize=32) if (self.realtime and self.on_saved) else None
        self._up_thread = None
        if self._up_q is not None:
            self._up_thread = threading.Thread(target=self._upload_loop, daemon=True,
                                               name="clip-upload")
            self._up_thread.start()

    def _clean_stage(self):
        """Delete leftovers in stage_dir. -> how many files were removed.

        Only files older than STAGE_ORPHAN_AGE, because this directory is not provably private:
        the serve default is one shared path under /tmp, and offline shards writing into one
        output directory would share `.clip_stage` too. A staged clip that another live process
        is still writing is always seconds old, so the age gate is what keeps this from deleting
        working data and turning its move into a FileNotFoundError. A real orphan is only
        delayed by it, never missed -- the next start an hour on collects it.

        Failures are swallowed on purpose: this is housekeeping, and a stage dir that really
        cannot be used will say so loudly on the first clip.
        """
        n, cutoff = 0, time.time() - STAGE_ORPHAN_AGE
        try:
            for name in os.listdir(self.stage_dir):
                if not name.endswith((".mp4", ".json")):
                    continue
                path = os.path.join(self.stage_dir, name)
                try:
                    if os.path.getmtime(path) > cutoff:
                        continue
                    os.remove(path)
                    n += 1
                except OSError:
                    pass
        except OSError:
            return 0
        if n:
            print(f"[clip] 清理上次运行残留的暂存文件 {n} 个 -> {self.stage_dir}")
        return n

    def _upload_loop(self):
        """Runs on_saved off the writer thread. See the _up_q comment in __init__."""
        while True:
            item = self._up_q.get()
            try:
                if item is None:
                    break
                self._run_on_saved(*item)
            finally:
                self._up_q.task_done()

    def _run_on_saved(self, files, rel_dir, meta):
        try:
            self.on_saved(files, rel_dir, meta)
        except Exception as e:                            # noqa: BLE001
            # A failed upload must not take the recorder -- or the service -- down; the local
            # copy is already on disk either way.
            print(f"[clip] on_saved 回调失败（本地文件已保存）: {e}")

    def _queue_upload(self, files, rel_dir, meta):
        """Hand a saved clip to the upload thread. Never blocks the writer thread."""
        if self._up_q is None:                            # offline: no queue to protect
            self._run_on_saved(files, rel_dir, meta)
            return
        try:
            self._up_q.put_nowait((files, rel_dir, meta))
        except queue.Full:
            self.upload_dropped += 1
            print(f"[clip] 上传队列已满，{os.path.basename(files[0])} 仅保留在本地"
                  f"（累计 {self.upload_dropped}）")

    # ---------------------------------------------------------------- producer side

    def push(self, frame, event, ts):
        """Called from the processing thread, once per processed frame. Never blocks."""
        if self._last_ts is not None:
            gap = ts - self._last_ts
            if 0 < gap <= 1.0:     # > 1 s is a stall (model warmup, reconnect), not a rate
                self._gaps.append(gap)
        self._last_ts = ts
        self._pre.append((frame, ts))
        while len(self._pre) > 1 and ts - self._pre[0][1] > self.pre_sec:
            self._pre.pop(0)

        # Retry a close that could not be enqueued earlier. Nothing else may be queued until it
        # lands: the writer consumes sequentially, so a `start` overtaking a pending `end` would
        # hand it a new path while the previous file is still open. _active is always None while
        # this is outstanding (_finish clears it, and _start refuses), so no frame is lost here.
        if not self._flush_pending_end(ts):
            return

        fires = (event is not None
                 and label_key(event.label) not in self.skip
                 and float(getattr(event, "confidence", 1.0)) >= self.min_confidence)

        if self._active is None:
            if self._pending_start is not None:
                ev, deadline = self._pending_start
                if ts <= deadline:
                    self._start(ev, ts)          # still full -> stays pending, retried again
                    return
                self._pending_start = None
                self.abandoned += 1
                print(f"[clip] 写入队列持续满 {START_RETRY_SEC:.0f}s，放弃片段 "
                      f"{label_key(ev.label)}（累计放弃 {self.abandoned}）")
            if fires:
                self._start(event, ts)
            return

        # The pre-roll already carried this frame into the file at _start time, so only
        # frames after the start are streamed in.
        self._send_frame(frame, ts)
        act = self._active
        act["frames"] += 1
        if fires:
            # Counted whichever label it is: `labels_in_clip` is then an honest record of
            # what else the recognizer saw while this clip ran.
            act["labels"][event.label] = act["labels"].get(event.label, 0) + 1
            if label_key(event.label) == act["key"]:
                act["t_last_event"] = ts
                act["pending"], act["pending_n"] = None, 0
                # The headline label is fixed at _start and never reassigned: the file is
                # already filed under that label's directory, so letting a later event
                # rename it would leave the metadata disagreeing with the path.
                if float(getattr(event, "confidence", 0.0)) > act["best_confidence"]:
                    act["best_confidence"] = float(event.confidence)
            else:
                nxt = self._confirm_switch(act, event)
                if nxt is not None:
                    self._finish(ts, f"行为切换 -> {nxt.label}")
                    self._start(nxt, ts)     # pre-roll gives the new clip its lead-in
                    return
        if ts - act["t0"] >= self.max_sec:
            self._finish(ts, "达到 clip_max_sec 上限")
        elif ts - act["t_last_event"] > self.post_sec:
            self._finish(ts, "post 静默期结束")

    def close(self):
        """Flush an in-flight clip and stop the writer thread. Safe to call twice."""
        if self._active is not None:
            # The last real frame, not t_last_event + post_sec: the stream stopped here, and
            # claiming a duration that runs past the final frame would pad the clip with a
            # frozen image to match.
            self._finish(self._last_ts, "服务停止")
        if self._pending_end is not None:
            # Shutdown is the one place where blocking is the right answer: there is no later
            # frame to retry on, and the alternative is a truncated file. Nothing downstream of
            # here is holding the pipeline open any more.
            msg, _, name, report = self._pending_end
            self._pending_end = None
            try:
                self._q.put(msg, timeout=END_RETRY_SEC)
                print(report)
            except queue.Full:
                self.abandoned += 1
                print(f"[clip] 写入队列持续阻塞，片段 {name} 可能不完整")
        if self._thread.is_alive():
            # Bounded, for the same reason as above: on a wedged writer an unbounded put here
            # never returns, and this runs on the shutdown path -- serve mode still has to write
            # session_json with status=stopped after it, so hanging here turns a lost clip into a
            # lost session record. The thread is a daemon and the join below is already bounded,
            # so a `stop` that cannot be delivered costs nothing beyond an unclean exit.
            try:
                self._q.put(("stop",), timeout=END_RETRY_SEC)
            except queue.Full:
                print("[clip] 写入线程无响应，跳过停止信号")
            self._thread.join(timeout=30)
        if self._up_thread is not None and self._up_thread.is_alive():
            # Uploads outlive the clips they belong to, so shutdown gives them a chance to
            # finish -- bounded for the same reason as above (session_json still has to be
            # written). Whatever is left is missing from HDFS only; the local clips are on disk.
            try:
                self._up_q.put(None, timeout=END_RETRY_SEC)
            except queue.Full:
                pass
            self._up_thread.join(timeout=30)
            if self._up_thread.is_alive():
                print("[clip] 上传线程仍未结束，剩余片段仅保留在本地")

    def stats(self):
        return {"session": self.session, "clips_saved": self.saved,
                "clips_abandoned": self.abandoned,
                # An `end` that never reached the writer, so the clip was dropped and its
                # encoder force-released instead of leaking. Tracks clips_abandoned but only
                # counts the ones that had actually opened a file; non-zero means the writer
                # thread went unresponsive for END_RETRY_SEC at a time.
                "clips_forced": self.forced_close,
                "frames_dropped": self.dropped_frames,
                "uploads_dropped": self.upload_dropped,
                "stage_orphans_cleaned": self.stage_orphans}

    # ---------------------------------------------------------------- clip lifecycle

    def _confirm_switch(self, act, event):
        """Another behaviour fired inside an active clip. -> the event to cut over to, or
        None while it is still unconfirmed.

        One event of disagreement is not a new behaviour. Near the decision boundary the
        recognizer alternates between two labels from window to window, and cutting on the
        first of them would shred one incident into a pile of one-second files — the same
        failure the merge rule exists to prevent, just in the other direction. The evidence
        required is `switch_after` events in a row naming the same new label, with no event
        of the clip's own label in between (that resets the count).
        """
        p = act["pending"]
        if p is None or label_key(p.label) != label_key(event.label):
            act["pending"], act["pending_n"] = event, 1
        else:
            act["pending_n"] += 1
            if float(getattr(event, "confidence", 0.0)) > float(getattr(p, "confidence", 0.0)):
                act["pending"] = event       # 用置信度最高的那个事件给新片段命名
        return act["pending"] if act["pending_n"] >= self.switch_after else None

    def _start(self, event, ts):
        if self._pending_end is not None:
            # The writer has not taken the previous clip's close yet, and a `start` may not
            # overtake it in the queue. Defer through the same retry path a full queue uses.
            if self._pending_start is None:
                self._pending_start = (event, ts + START_RETRY_SEC)
            return
        self._seq += 1
        key = label_key(event.label)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        rel_dir = key
        name = f"{stamp}_{key}_{self._seq:04d}"
        fps = self._estimate_fps()
        pre = list(self._pre)                       # [(frame, ts)] -- the writer needs both
        self._active = {
            "seq": self._seq, "label": event.label, "key": key, "rel_dir": rel_dir,
            "name": name, "fps": fps, "t0": self._pre[0][1] if self._pre else ts,
            "t_trigger": ts, "t_last_event": ts, "frames": len(pre),
            "labels": {event.label: 1}, "best_confidence": float(event.confidence),
            "first_event": event.to_dict() if hasattr(event, "to_dict") else None,
            "pending": None, "pending_n": 0,     # 待确认的行为切换，见 _confirm_switch
        }
        path = os.path.join(self.root, rel_dir, name + ".mp4")
        try:
            self._q.put_nowait(("start", path, fps, pre))
        except queue.Full:
            # Dropping the opening message would leave every later frame unanchored, so the
            # clip cannot be written half-formed -- but giving up on it here loses the whole
            # behaviour, and the queue is at its fullest at exactly the worst moment: a
            # behaviour switch starts the next clip while the writer is still closing and
            # uploading the previous one. So it is retried on the following frames instead.
            # The pre-roll is re-snapshotted each attempt, so a clip that opens 200 ms late
            # still contains its lead-in; only a queue that stays full for START_RETRY_SEC
            # (the writer is wedged, not merely busy) counts as an abandoned clip.
            self._seq -= 1                       # 这次没算数，序号留给下一次，保持连号
            self._active = None
            if self._pending_start is None:
                self._pending_start = (event, ts + START_RETRY_SEC)
            return
        self._pending_start = None
        print(f"[clip] 开始录制 {name} label={event.label} "
              f"conf={event.confidence:.2f} 预卷={len(pre)}帧")

    def _finish(self, ts, why):
        act = self._active
        self._active = None
        duration = max(0.0, ts - act["t0"])
        meta = {
            "session": self.session, "seq": act["seq"], "label": act["label"],
            "label_key": act["key"], "confidence": round(act["best_confidence"], 4),
            "labels_in_clip": act["labels"], "frames": act["frames"],
            "fps": round(act["fps"], 3), "duration_seconds": round(duration, 3),
            "trigger_stream_time": round(act["t_trigger"], 3),
            "first_event": act["first_event"], "closed_because": why,
        }
        final_path = final_rel_dir = None
        if self.naming == "range":
            # Whole seconds for the filename (matches the "2-3s-喝水.mp4" spec verbatim);
            # end<=start only when a clip's whole duration rounds into one second, which
            # would otherwise render as the nonsensical "3-3s".
            start_s, end_s = round(act["t0"]), round(ts)
            if end_s <= start_s:
                end_s = start_s + 1
            label_zh = DISPLAY_TO_ZH.get(act["label"], act["label"])
            name = f"{start_s}-{end_s}s-{label_zh}"
            if name in self._used_names:
                # seq is unique and monotonic per recorder, so this always resolves the
                # collision in one step rather than needing a retry loop.
                name = f"{name}-{act['seq']}"
            self._used_names.add(name)
            final_rel_dir = ""
            final_path = os.path.join(self.root, name + ".mp4")
            meta["start_seconds"] = round(act["t0"], 2)
            meta["end_seconds"] = round(ts, 2)
        else:
            meta["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(act["t0"]))
        # `end` must get through: a dropped close leaves an unreleased VideoWriter and a
        # truncated file. It used to buy that with `put(..., timeout=10)`, which made this the
        # one place a clip could block the processing thread -- and the docstring on push() says
        # it never blocks. Measured with clip_denoise=quality: 1.3-6.8 s per close, p90 end-to-end
        # latency 3251 ms against the <= 1 s budget, and the cadence thread papering over it by
        # re-sending the last JPEG up to 174 times (~7 s of frozen picture at 25 fps) so that
        # timestamps, DTS and bitrate all stayed perfect while nothing moved.
        #
        # So it is deferred instead, exactly as _start defers an opening it cannot enqueue: keep
        # the message, retry it on the following frames, and only give up after END_RETRY_SEC of
        # a queue that never drains (a wedged writer, not a busy one). The close is still
        # delivered before anything else -- push() returns early while one is pending and _start
        # refuses -- so the writer never sees a `start` while a file is open.
        self._pending_end = (("end", meta, ts, final_path, final_rel_dir),
                             ts + END_RETRY_SEC, act["name"],
                             f"[clip] 结束录制 {act['name']} 帧数={act['frames']} "
                             f"时长={duration:.1f}s（{why}）")
        self._flush_pending_end(ts)

    def _flush_pending_end(self, ts):
        """Try to hand the deferred close to the writer. -> True when nothing is pending.

        Returns True on the give-up path too: a close that can never be delivered must not wedge
        the recorder forever, so the clip is written off and later behaviours can still record.
        """
        if self._pending_end is None:
            return True
        msg, deadline, name, report = self._pending_end
        try:
            self._q.put_nowait(msg)
        except queue.Full:
            if ts <= deadline:
                return False
            self._pending_end = None
            self.abandoned += 1
            print(f"[clip] 写入队列持续满 {END_RETRY_SEC:.0f}s，片段 {name} 可能不完整"
                  f"（累计放弃 {self.abandoned}）")
            return True
        self._pending_end = None
        print(report)
        return True

    def _estimate_fps(self):
        """Fallback rate, used only if a clip ends before OPEN_AFTER frames arrive.

        The real number is measured by the writer from the clip's own timestamps; this is
        just what a two-frame clip gets. Recent gaps beat the pre-roll span because the
        pre-roll is time-bounded and a stall can leave it holding a single frame.
        """
        if self._gaps:
            return min(60.0, max(1.0, 1.0 / statistics.median(self._gaps)))
        return measure_fps([t for _, t in self._pre])

    def _send_frame(self, frame, ts):
        try:
            self._q.put_nowait(("frame", frame, ts))
        except queue.Full:
            self.dropped_frames += 1
            if self.dropped_frames in (1, 10, 100) or self.dropped_frames % 1000 == 0:
                print(f"[clip] 写入跟不上，已丢弃 {self.dropped_frames} 帧"
                      f"（片段仍会保存，只是略有跳帧）")

    # ---------------------------------------------------------------- writer thread

    def _clean(self, frame):
        if self.denoise == "off":
            return frame
        from .denoise_util import denoise_frame
        return denoise_frame(frame, self.denoise)

    def _writer_loop(self):
        """Owns every cv2 call, and re-times frames onto a uniform clock as it writes.

        A clip has to last as long as the incident did — a 12 s fall must produce a 12 s
        video — and simply stamping a header fps cannot deliver that. The capture rate is not
        constant (it sags once encoding starts, and again while a clip is being uploaded), so
        every fps estimated from a prefix of the clip is biased fast: measured 30 fps against
        a sustained 27.6, and 30 against 21.6 on a clip cut short at shutdown, i.e. playback
        7-41% too quick. Estimating harder cannot fix that — the sample is always taken
        before the slowdown it is trying to predict.

        So the duration is made correct by construction instead: frames are resampled onto a
        fixed `used_fps` grid, duplicated when the pipeline falls behind it and dropped when
        it runs ahead. `used_fps` is then only a quality/size knob, not a correctness one.
        """
        writer, meta_path, mp4_path, rel_dir = None, None, None, None
        stage_mp4 = None
        pending, pre_times, post_times, hint, used_fps = [], [], [], 15.0, 15.0
        next_due, stretched, last_frame = None, 0, None

        def force_close(reason):
            """Release a writer whose `end` never arrived. -> nothing.

            An abandoned close (_flush_pending_end's give-up path) leaves this thread holding
            an open VideoWriter, and the next `start` used to simply rebind over it. Rebinding
            does NOT release it: the writer's stderr drain thread targets a bound method, so
            the object stays reachable, its stdin is never finalised, and the ffmpeg child
            never sees EOF -- it waits for input for as long as the process lives. Measured
            end to end with clip_leak3.py (9 abandoned closes -> 4 leaked ffmpeg children, 4
            threads, 8 fds, monotonic, each with a staged mp4 nothing ever deleted) and
            deterministically with clip_leak5.py, which drives this queue by hand because only
            an abandon *after* OPEN_AFTER frames leaks and most abandons are start-side ones.
            In the field the trigger fired ~1.2 times a minute, i.e. ~250 stuck processes per
            shift. clip_leak5.py is the regression case: run it against HEAD and this tree.

            The staged file is deleted rather than delivered even though it would play: its
            tail is missing and its metadata went with the abandoned `end`, while clip_dir's
            contract is that a file there is a complete clip with a json beside it.
            """
            nonlocal writer
            if writer is None:
                return
            self.forced_close += 1
            print(f"[clip] 片段 {os.path.basename(stage_mp4)} {reason}，"
                  f"已强制释放编码器并丢弃（累计强制释放 {self.forced_close}）")
            try:
                writer.close()
            except Exception as e:                        # noqa: BLE001
                print(f"[clip] 强制释放编码器失败: {e}")
            writer = None
            # stage_mp4 is always set when a writer is open (open_now built it from it), so no
            # None-handling here -- and none on purpose: a str(None) fallback would build the
            # relative path "None.json" and hand it to os.remove.
            for p in (stage_mp4, os.path.splitext(stage_mp4)[0] + ".json"):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except OSError:
                    pass

        def emit(frame, ts):
            """Write `frame` for every grid slot up to ts (>=0 times)."""
            nonlocal next_due, stretched, last_frame
            if next_due is None:
                next_due = ts
            last_frame = frame
            step, n = 1.0 / used_fps, 0
            while next_due <= ts and n < MAX_DUP:
                writer.write(frame)
                next_due += step
                n += 1
            if n >= MAX_DUP:                 # a stall longer than MAX_DUP/used_fps
                next_due = ts + step
                stretched += 1

        def open_now():
            nonlocal writer, used_fps, next_due
            measured = measure_fps(post_times, measure_fps(pre_times, hint))
            used_fps = float(min(FPS_RANGE[1], max(FPS_RANGE[0], round(measured))))
            writer = VideoWriter(stage_mp4, used_fps)
            next_due = None
            for f, t in pending:
                emit(self._clean(f), t)
            pending.clear()

        while True:
            item = self._q.get()
            kind = item[0]
            try:
                if kind == "stop":
                    break
                if kind == "start":
                    _, path, hint, pre = item
                    # Must precede every rebinding below: force_close reads the *previous*
                    # clip's mp4_path/stage_mp4.
                    force_close("未收到收尾消息")
                    mp4_path, rel_dir = path, os.path.basename(os.path.dirname(path))
                    meta_path = os.path.splitext(path)[0] + ".json"
                    os.makedirs(self.stage_dir, exist_ok=True)
                    stage_mp4 = os.path.join(self.stage_dir, os.path.basename(path))
                    writer, pending, next_due, stretched = None, list(pre), None, 0
                    pre_times, post_times, last_frame = [t for _, t in pre], [], None
                elif kind == "frame":
                    if writer is not None:
                        emit(item[1], item[2])
                    elif mp4_path is not None:
                        pending.append((item[1], item[2]))
                        post_times.append(item[2])
                        if len(post_times) >= OPEN_AFTER:
                            open_now()
                elif kind == "end":
                    if mp4_path is None:
                        continue
                    if len(item) > 3 and item[3]:
                        # naming="range": _finish already computed the real final path from
                        # the clip's now-known duration: override what "start" guessed.
                        mp4_path, rel_dir = item[3], (item[4] or "")
                        meta_path = os.path.splitext(mp4_path)[0] + ".json"
                    if writer is None:                # clip shorter than OPEN_AFTER frames
                        open_now()
                    # Fill the grid out to the clip's real end. Without this a queue overflow
                    # near the end silently shortens the video: the dropped tail frames never
                    # advance the timeline, so the file stops early while the metadata still
                    # claims the full duration.
                    if last_frame is not None and len(item) > 2:
                        emit(last_frame, item[2])
                    writer.close()
                    meta = item[1]
                    meta["video"] = mp4_path
                    meta["frames_written"] = writer.count
                    meta["fps"] = round(used_fps, 3)   # what the header actually says
                    # Resampling should make these come out at 1.0; they are recorded anyway
                    # so a clip that did drift says so instead of looking fine.
                    if meta.get("duration_seconds", 0) > 0 and writer.count:
                        meta["capture_fps"] = round(
                            meta["frames"] / meta["duration_seconds"], 3)
                        meta["playback_speed"] = round(
                            meta["duration_seconds"] / (writer.count / used_fps), 3)
                    if stretched:
                        meta["stalls_clamped"] = stretched
                    stage_meta = os.path.splitext(stage_mp4)[0] + ".json"
                    with open(stage_meta, "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)
                    os.makedirs(os.path.dirname(mp4_path), exist_ok=True)
                    shutil.move(stage_mp4, mp4_path)      # mp4 first: the json is the marker
                    shutil.move(stage_meta, meta_path)    # that says the clip is complete
                    self.saved += 1
                    print(f"[clip] 已保存 {mp4_path}（{writer.count} 帧）")
                    if self.on_saved:
                        self._queue_upload([mp4_path, meta_path], rel_dir, meta)
                    writer, mp4_path = None, None      # stray frames must not reopen a clip
            except Exception as e:                        # noqa: BLE001
                print(f"[clip] 写入失败: {e}")
                if writer is not None:
                    try:
                        writer.close()
                    except Exception:                     # noqa: BLE001
                        pass
                writer, mp4_path, pending = None, None, []
            finally:
                self._q.task_done()
        # Reached only on `stop` with a file still open, i.e. an `end` lost at shutdown. Same
        # treatment as mid-run: release the child and take the staged file with it, rather than
        # leaving an orphan behind for the next run's _clean_stage to find.
        force_close("服务停止时仍未收尾")
