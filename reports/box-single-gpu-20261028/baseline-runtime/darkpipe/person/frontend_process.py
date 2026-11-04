"""Isolate CUDA submission from HTTP, recording and JPEG Python threads."""
import multiprocessing as mp
import time
from collections import deque
from dataclasses import replace


def _worker(cfg, channel):
    stages = []
    try:
        import cv2
        import numpy as np
        import torch
        from .runtime import Frontend, prepare_stages
        from darkpipe.stages import build_stages
        torch.set_num_threads(2)
        cv2.setNumThreads(2)
        torch.cuda.set_device(cfg.device)
        stages, _ = build_stages(replace(cfg, recognize='off', enhance_chunk=1))
        stages = [s for s in stages if not s.post_recognition]
        for stage in stages:
            stage.load(cfg.device)
        front = Frontend(cfg)
        dummy = np.full((452 if cfg.input_processed else 404, 720, 3), 48, np.uint8)
        for _ in range(2):
            frame, _ = prepare_stages(stages, cfg, dummy)
            front(frame, True)
        channel.send(('ready', None))
        while True:
            raw = channel.recv()
            if raw is None:
                break
            raw, source_shape = raw
            started = time.monotonic()
            frame, luma = prepare_stages(stages, cfg, raw, input_cropped=True)
            prepared = time.monotonic()
            boxes, poses = front(frame, cfg.skel_fusion_alpha < .999 and
                                 (luma is None or luma >= cfg.lum_gate))
            channel.send(('result', (frame, luma, boxes, poses,
                prepared-started, time.monotonic()-prepared, source_shape)))
    except BaseException as exc:
        import traceback
        try:
            channel.send(('error', f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}'))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        for stage in stages:
            stage.close()
        channel.close()


class FrontendProcess:
    def __init__(self, cfg):
        from darkpipe.stages.downscale import DownscaleStage
        self.cfg = cfg
        self.downscale = DownscaleStage(cfg.proc_max_side) if cfg.proc_max_side else None
        ctx = mp.get_context('spawn')
        self.channel, child = ctx.Pipe()
        self.process = ctx.Process(target=_worker, args=(cfg, child), name='person-frontend')
        self.process.start()
        child.close()
        try:
            self._receive('ready', 90.)
        except BaseException:
            self.close()
            raise

    def _receive(self, expected, timeout):
        deadline = time.monotonic()+timeout
        while not self.channel.poll(.1):
            if not self.process.is_alive():
                raise RuntimeError('Frontend process exited unexpectedly')
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Frontend produced no response for {timeout:g}s')
        kind, data = self.channel.recv()
        if kind != expected:
            raise RuntimeError(f'Frontend {kind}: {data}')
        return data

    def __call__(self, raw):
        self.submit(raw)
        return self.receive()

    def submit(self, raw):
        from .runtime import content_height
        if self.cfg.input_processed:
            raw = raw[:content_height(raw.shape[0])]
        source_shape = raw.shape
        if self.downscale is not None:
            raw = self.downscale([raw])[0]
        self.channel.send((raw, source_shape))

    def receive(self):
        return self._receive('result', 10.)

    def close(self):
        if self.process.is_alive():
            try:
                self.channel.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
            self.process.join(timeout=12)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
        self.channel.close()


class FrontendPool:
    """At most one frame per worker; deliver in capture order without growing queues."""
    def __init__(self, cfg, devices):
        self.workers, self.pending = [], deque()
        try:
            for device in devices:
                self.workers.append(FrontendProcess(replace(cfg, device=device)))
        except BaseException:
            self.close()
            raise

    def submit(self, raw):
        busy = {index for index, _, _ in self.pending}
        index = next((i for i in range(len(self.workers)) if i not in busy), None)
        if index is None:
            return False
        self.workers[index].submit(raw[0])
        self.pending.append((index, raw[1:], time.monotonic()))
        return True

    def collect(self):
        if not self.pending:
            return None
        index, metadata, started = self.pending[0]
        worker = self.workers[index]
        if not worker.process.is_alive():
            raise RuntimeError('Frontend worker exited unexpectedly')
        if not worker.channel.poll():
            if time.monotonic()-started > 10.:
                raise TimeoutError('Frontend frame exceeded10s')
            return None
        result = worker.receive()
        self.pending.popleft()
        return metadata, result

    def close(self):
        # Receive already submitted frames before requesting exit, so a child cannot
        # block on a full result pipe while the parent waits for it to terminate.
        for index, _, _ in list(self.pending):
            worker = self.workers[index]
            try:
                worker.receive()
            except (RuntimeError, TimeoutError, EOFError, OSError):
                pass
        self.pending.clear()
        for worker in self.workers:
            worker.close()
