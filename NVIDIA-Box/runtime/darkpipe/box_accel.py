"""Box-only bounded CUDA graph reuse; weights and temporal observations are unchanged."""
import os

import torch


def enabled() -> bool:
    return os.environ.get('DARKPIPE_BOX_PROFILE', 'balanced') == 'balanced'


def graphs_enabled() -> bool:
    # CUDA graph pools retain activations; default off on the 4 GB shared-memory target.
    return enabled() and os.environ.get('DARKPIPE_BOX_GRAPHS', '0') == '1'


def action_batch() -> int:
    value = int(os.environ.get('DARKPIPE_BOX_ACTION_BATCH', '2'))
    if value not in (1, 2):
        raise ValueError('Box action batch must be 1 or 2')
    return value


class GraphForward:
    """Own stable inputs/outputs for at most two shapes, used by one inference thread.

    Returned storage is valid until the next invocation of that shape. Callers consume
    it synchronously. Unseen shapes beyond the bound use the original model. Capture
    failures fall back to eager execution, whose actual model errors still propagate.
    """
    def __init__(self, model, capacity: int = 2):
        self.model = model
        self.capacity = capacity
        self.entries = {}
        self.failed = set()

    @torch.inference_mode()
    def __call__(self, x):
        key = (tuple(x.shape), tuple(x.stride()), x.dtype, x.device)
        if key not in self.entries and key not in self.failed and len(self.entries) < self.capacity:
            # Leave headroom for ORT and recording; a small card retains the eager path.
            if torch.cuda.mem_get_info(x.device)[0] < 1536*1024**2:
                self.failed.add(key)
            else:
                try:
                    static = torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)
                    static.copy_(x)
                    stream = torch.cuda.Stream(device=x.device)
                    stream.wait_stream(torch.cuda.current_stream(x.device))
                    # Autocast's outer cache may own half-precision parameter copies that
                    # are freed on leaving the caller's context. Capture their conversion
                    # inside the graph pool instead of recording pointers to that cache.
                    autocast_enabled = torch.is_autocast_enabled('cuda')
                    autocast_dtype = torch.get_autocast_dtype('cuda')
                    with torch.cuda.stream(stream), torch.autocast('cuda',
                            enabled=autocast_enabled, dtype=autocast_dtype, cache_enabled=False):
                        for _ in range(3):
                            self.model(static)
                    torch.cuda.current_stream(x.device).wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.autocast('cuda', enabled=autocast_enabled,
                            dtype=autocast_dtype, cache_enabled=False), torch.cuda.graph(graph):
                        output = self.model(static)
                    self.entries[key] = (static, output, graph)
                except RuntimeError as exc:
                    self.failed.add(key)
                    print(f'[box] CUDA graph不可用，保持原模型执行：{exc}', flush=True)
        if key not in self.entries:
            return self.model(x)
        static, output, graph = self.entries[key]
        static.copy_(x)
        graph.replay()
        return output

    def close(self) -> None:
        self.entries.clear()
        self.model = None
