"""Optional one-process-per-action-GPU backend, isolating Python submission contention."""
from collections import Counter
import multiprocessing as mp
import queue
import threading
from types import SimpleNamespace

import torch

from action_engine import ActionEngine
from scheduling import ActionPool, Mailbox


def action_process(device, max_batch, diagnostics, incoming, outgoing):
    engine = None
    try:
        torch.set_num_threads(1)
        torch.cuda.set_device(device)
        engine = ActionEngine(device, max_batch, diagnostics=diagnostics)
        outgoing.put(('ready', None))
        while True:
            tasks = incoming.get()
            if tasks is None:
                break
            rows = engine.predict(tasks)
            outgoing.put(('result', (rows, dict(engine.stats), engine.calls[-1])))
    except Exception as exc:
        import traceback
        outgoing.put(('error', f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}'))
    finally:
        if engine is not None:
            engine.close()


class ProcessActionPool(ActionPool):
    """Reuse bounded fair mailboxes; send one batch at a time to each GPU process."""
    def __init__(self, devices, max_batch=4, diagnostics=False, coalesce=False):
        self.devices, self.inline, self.max_batch = devices, False, max_batch
        self.mailboxes = [Mailbox(coalesce=coalesce) for _ in devices]
        self.results = queue.Queue()
        self.engines = [SimpleNamespace(stats=Counter(), calls=[]) for _ in devices]
        self.errors = [None]*len(devices)
        self.assignment, self.assignment_next = {}, 0
        self.threads, self.processes, self.channels = [], [], []
        ctx = mp.get_context('spawn')
        try:
            for i, device in enumerate(devices):
                incoming, outgoing = ctx.Queue(maxsize=1), ctx.Queue(maxsize=2)
                process = ctx.Process(target=action_process,
                    args=(device,max_batch,diagnostics,incoming,outgoing), name=f'action-{device}')
                process.start()
                self.processes.append(process)
                self.channels.append((incoming,outgoing))
            for i in range(len(devices)):
                kind, data = self._receive(i)
                if kind != 'ready':
                    raise RuntimeError(data)
            for i in range(len(devices)):
                thread = threading.Thread(target=self._dispatch,args=(i,),daemon=True,name=f'action-dispatch-{i}')
                self.threads.append(thread)
                thread.start()
        except Exception:
            self.close()
            raise

    def _receive(self, index):
        while True:
            try:
                return self.channels[index][1].get(timeout=.1)
            except queue.Empty:
                if not self.processes[index].is_alive():
                    raise RuntimeError(f'GPU process {index} exited unexpectedly')

    def _dispatch(self, index):
        box = self.mailboxes[index]
        try:
            while not box.stopped:
                tasks = box.take(self.max_batch)
                if not tasks:
                    continue
                try:
                    self.channels[index][0].put(tasks)
                    kind, data = self._receive(index)
                    if kind != 'result':
                        raise RuntimeError(data)
                    rows, stats, call = data
                    self.engines[index].stats = Counter(stats)
                    self.engines[index].calls.append(call)
                    for row in rows:
                        self.results.put(row)
                finally:
                    box.finished(len(tasks))
        except Exception as exc:
            self.errors[index] = exc
            box.stop()

    def close(self):
        for box in self.mailboxes:
            box.stop()
        for thread in self.threads:
            thread.join(timeout=30)
        for process, (incoming, _) in zip(self.processes,self.channels):
            if process.is_alive():
                try:
                    incoming.put(None,timeout=1)
                except queue.Full:
                    pass
        for process in self.processes:
            process.join(timeout=20)
            if process.is_alive():
                process.terminate()  # only an owned failed worker; never another GPU task
                process.join(timeout=5)
        for incoming,outgoing in self.channels:
            incoming.close()
            outgoing.close()
        self.check()
