"""Bounded per-person task scheduling, result ordering, and GPU ownership."""
from collections import Counter, OrderedDict, deque
import queue
import threading
import time

from .action_engine import ActionEngine, ActionTask


class Mailbox:
    def __init__(self, capacity: int = 128, coalesce: bool = False):
        self.capacity, self.coalesce = capacity, coalesce
        self.pending = OrderedDict()
        self.condition = threading.Condition()
        self.stopped = False
        self.busy = 0
        self.sequence = 0
        self.stats = Counter()

    def put(self, task: ActionTask) -> None:
        with self.condition:
            key = task.key if self.coalesce else self.sequence
            self.sequence += 1
            while len(self.pending) >= self.capacity and key not in self.pending and not self.stopped:
                self.condition.wait(.1)
            if self.stopped:
                raise RuntimeError('Action worker stopped')
            if key in self.pending:
                if task.timestamp <= self.pending[key].timestamp:
                    self.stats['outdated_submission'] += 1
                    return
                self.stats['coalesced'] += 1
            self.pending[key] = task
            self.stats['submitted'] += 1
            self.stats['max_pending'] = max(self.stats['max_pending'], len(self.pending))
            self.condition.notify_all()

    def take(self, batch: int, wait: bool = True) -> list:
        with self.condition:
            while not self.pending and not self.stopped and wait:
                self.condition.wait(.1)
            if self.stopped or not self.pending:
                return []
            tasks = []
            # Arrival order gives persistent tracks a turn even when their latest task is replaced.
            for _ in range(min(batch, len(self.pending))):
                _, task = self.pending.popitem(last=False)
                tasks.append(task)
            self.busy += len(tasks)
            self.condition.notify_all()
            return tasks

    def finished(self, count: int) -> None:
        with self.condition:
            self.busy -= count
            self.condition.notify_all()

    def idle(self) -> bool:
        with self.condition:
            return not self.pending and self.busy == 0

    def stop(self) -> None:
        with self.condition:
            self.stopped = True
            self.condition.notify_all()


class ResultStore:
    def __init__(self):
        self.latest = {}
        self.rejected_old = 0

    def update(self, row: dict) -> bool:
        key = (row['session'], row['track_id'], row['history_generation'])
        previous = self.latest.get(key)
        if previous and previous['timestamp'] >= row['timestamp']:
            self.rejected_old += 1
            return False
        self.latest[key] = row
        self.latest = {k:v for k,v in self.latest.items() if row['timestamp']-v['timestamp'] < 3.}
        return True

    def for_track(self, key: tuple, timestamp: float, age_limit: float = 1.) -> dict | None:
        row = self.latest.get(key)
        if row and 0 <= timestamp-row['timestamp'] <= age_limit:
            return row
        return None


class ActionPool:
    """One submitting thread per action GPU; inline processing when frontend shares GPU."""
    def __init__(self, devices: list[str], inline: bool, max_batch: int,
                 diagnostics: bool = False, coalesce: bool = False, options: dict | None = None):
        self.options = options
        self.devices, self.inline, self.max_batch = devices, inline, max_batch
        self.mailboxes = [Mailbox(coalesce=coalesce) for _ in devices]
        self.results = queue.Queue()
        self.engines = [None]*len(devices)
        self.errors = [None]*len(devices)
        self.threads = []
        self.ready = [threading.Event() for _ in devices]
        self.assignment = {}
        self.assignment_next = 0
        if inline:
            if len(devices) != 1:
                raise ValueError('Inline mode owns exactly one GPU')
            self.engines[0] = ActionEngine(devices[0], max_batch, diagnostics=diagnostics, options=self.options)
        else:
            for i in range(len(devices)):
                t = threading.Thread(target=self._worker, args=(i, diagnostics), daemon=True,
                                     name=f'person-actions-{devices[i]}')
                self.threads.append(t)
                t.start()
            for event in self.ready:
                while not event.wait(.1):
                    self.check()
            self.check()

    def _worker(self, index: int, diagnostics: bool) -> None:
        box = self.mailboxes[index]
        try:
            self.engines[index] = ActionEngine(self.devices[index], self.max_batch, diagnostics=diagnostics, options=self.options)
            self.ready[index].set()
            while not box.stopped:
                tasks = box.take(self.max_batch)
                if not tasks:
                    continue
                try:
                    for row in self.engines[index].predict(tasks):
                        self.results.put(row)  # publish this batch without waiting for another GPU/batch
                finally:
                    box.finished(len(tasks))
        except Exception as exc:
            self.errors[index] = exc
            box.stop()
            self.ready[index].set()
        finally:
            if self.engines[index] is not None:
                self.engines[index].close()

    def check(self) -> None:
        for error in self.errors:
            if error is not None:
                raise RuntimeError(f'Action worker failed: {error}') from error

    def submit(self, task: ActionTask) -> None:
        self.check()
        # Track ownership persists for this session/generation to preserve the input cache.
        if task.key not in self.assignment:
            self.assignment[task.key] = self.assignment_next % len(self.devices)
            self.assignment_next += 1
        self.mailboxes[self.assignment[task.key]].put(task)

    def step(self) -> None:
        if not self.inline:
            return
        tasks = self.mailboxes[0].take(self.max_batch, wait=False)
        if tasks:
            try:
                for row in self.engines[0].predict(tasks):
                    self.results.put(row)
            finally:
                self.mailboxes[0].finished(len(tasks))

    def collect(self) -> list:
        self.check()
        rows = []
        while True:
            try:
                rows.append(self.results.get_nowait())
            except queue.Empty:
                return rows

    def drain(self) -> list:
        rows = []
        deadline = time.monotonic()+30.
        while not all(box.idle() for box in self.mailboxes):
            if time.monotonic() >= deadline:
                raise TimeoutError('Action workers did not drain within 30s')
            self.step()
            rows.extend(self.collect())
            if not self.inline:
                time.sleep(.001)
        rows.extend(self.collect())
        return rows

    def snapshot(self) -> dict:
        return dict(workers=[dict(device=d, stats=dict(e.stats), calls=list(e.calls),
                                 queue=dict(b.stats)) for d, e, b in zip(self.devices, self.engines, self.mailboxes)])

    def close(self) -> None:
        for box in self.mailboxes:
            box.stop()
        for thread in self.threads:
            thread.join(timeout=30)
            if thread.is_alive():
                raise RuntimeError('Action worker did not stop')
        if self.inline:
            self.engines[0].close()
        self.check()
