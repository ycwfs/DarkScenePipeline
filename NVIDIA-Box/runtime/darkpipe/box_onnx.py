"""Native TensorRT 10 / ORT CUDA sessions for the Box, with local engine caching."""
from dataclasses import dataclass
import fcntl
import hashlib
import json
import locale
import os
from pathlib import Path
import platform
import tempfile

import numpy as np


def backend() -> str:
    choice = os.environ.get('DARKPIPE_BOX_BACKEND', 'auto')
    if choice == 'auto':
        return 'tensorrt' if platform.machine() == 'aarch64' else 'cuda'
    if choice not in ('cuda', 'tensorrt'):
        raise ValueError(f'Unknown Box inference backend: {choice}')
    return choice


def create_session(path: str, device: str):
    if backend() == 'tensorrt':
        if Path(path).name == 'rtmo-s_person_20261024.onnx':
            path = str(Path(path).with_name('rtmo-s_person_20261024_box.onnx'))
        return TensorRTSession(path, device)
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options,
        providers=[('CUDAExecutionProvider', {
            'device_id': int(device.split(':')[-1]), 'arena_extend_strategy': 'kSameAsRequested',
            'cudnn_conv_use_max_workspace': '0', 'gpu_mem_limit': 384*1024**2})])
    if session.get_providers()[0] != 'CUDAExecutionProvider':
        raise RuntimeError(f'{path} unexpectedly fell back to CPU')
    return session


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple


class TensorRTSession:
    """Fixed-shape ORT-style interface; no CPU fallback or cross-device engine reuse.

    JetPack supplies TensorRT on ARM64. Build on the target device; never ship a 3090
    engine to Jetson. Hash the ONNX, hardware and build settings into the cache key.
    Each instance has its own context and buffers and a single inference owner.
    """
    def __init__(self, path: str, device: str):
        # Some TensorRT builder/library paths change the process locale to C. Preserve
        # Chinese filenames/config reading in the caller, including failure paths.
        previous_locale = locale.setlocale(locale.LC_ALL)
        try:
            self._initialize(path, device)
        finally:
            locale.setlocale(locale.LC_ALL, previous_locale)

    def _initialize(self, path: str, device: str):
        import tensorrt as trt
        import torch
        self.torch, self.trt, self.device = torch, trt, device
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        raw = Path(path).read_bytes()
        prop = torch.cuda.get_device_properties(device)
        workspace_mib = 128
        # Pose confidence near .5 controls crop geometry and geometric priors; FP16
        # changed a true-drink crop in the paired regression. Preserve FP32 there.
        precision = 'fp32' if Path(path).name.startswith('rtmo-') else 'fp16'
        metadata = dict(onnx_sha256=hashlib.sha256(raw).hexdigest(), trt=trt.__version__,
            gpu=prop.name, capability=[prop.major, prop.minor], memory=prop.total_memory,
            architecture=platform.machine(), precision=precision, workspace_mib=workspace_mib,
            auxiliary_streams=0, format_version=1)
        key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        root = Path(os.environ.get('DARKPIPE_BOX_ENGINE_CACHE',
                    str(Path.home()/'.cache/darkpipe/tensorrt')))
        root.mkdir(parents=True, exist_ok=True)
        cache = root / (Path(path).stem+'-'+key+'.engine')
        with torch.cuda.device(device), (root/(key+'.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not cache.exists():
                print(f'[box] 构建本机 TensorRT engine：{Path(path).name}', flush=True)
                builder = trt.Builder(self.logger)
                network = builder.create_network(0)
                parser = trt.OnnxParser(network, self.logger)
                if not parser.parse(raw):
                    raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
                config = builder.create_builder_config()
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mib*1024**2)
                if precision == 'fp16':
                    config.set_flag(trt.BuilderFlag.FP16)
                else:
                    config.clear_flag(trt.BuilderFlag.TF32)
                config.max_aux_streams = 0
                serialized = builder.build_serialized_network(network, config)
                if serialized is None:
                    raise RuntimeError(f'TensorRT build failed: {path}')
                with tempfile.NamedTemporaryFile(dir=root, delete=False) as tmp:
                    tmp.write(bytes(serialized))
                    temporary = Path(tmp.name)
                temporary.replace(cache)
                cache.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
            self.engine = self.runtime.deserialize_cuda_engine(cache.read_bytes())
            if self.engine is None:
                raise RuntimeError(f'Cannot load cached engine; remove and rebuild: {cache}')
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError(f'Cannot allocate TensorRT context: {path}')
            self.stream = torch.cuda.Stream(device=device)
            self.inputs, self.outputs, self.buffers, self.allocators = [], [], {}, {}

            class OutputAllocator(trt.IOutputAllocator):
                def __init__(self, dtype, device):
                    trt.IOutputAllocator.__init__(self)
                    self.dtype, self.device = dtype, device
                    self.buffer, self.shape = None, None

                def reallocate_output(self, name, memory, size, alignment):
                    with torch.cuda.device(self.device):
                        needed = max(1, (size+torch.empty((), dtype=self.dtype).element_size()-1) //
                                     torch.empty((), dtype=self.dtype).element_size())
                        if self.buffer is None or self.buffer.numel() < needed:
                            self.buffer = torch.empty(needed, device=self.device, dtype=self.dtype)
                        return self.buffer.data_ptr()

                def reallocate_output_async(self, name, memory, size, alignment, stream):
                    return self.reallocate_output(name, memory, size, alignment)

                def notify_shape(self, name, shape):
                    self.shape = tuple(shape)

            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                shape = tuple(self.context.get_tensor_shape(name))
                dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
                tensor_dtype = torch.from_numpy(np.empty((), dtype=dtype)).dtype
                is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                if any(d < 0 for d in shape):
                    if is_input:
                        raise ValueError(f'Box requires fixed input shapes: {name} {shape}')
                    allocator = OutputAllocator(tensor_dtype, device)
                    self.allocators[name] = allocator
                    self.context.set_output_allocator(name, allocator)
                else:
                    tensor = torch.empty(shape, device=device, dtype=tensor_dtype)
                    self.buffers[name] = tensor
                    self.context.set_tensor_address(name, tensor.data_ptr())
                info = TensorInfo(name, shape)
                (self.inputs if is_input else self.outputs).append(info)
        print(f'[box] TensorRT ready：{Path(path).name} / {prop.name}', flush=True)

    def get_inputs(self):
        return self.inputs

    def get_providers(self):
        return ['TensorrtExecutionProvider']

    def run(self, output_names, feed):
        torch = self.torch
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream), torch.inference_mode():
            if set(feed) != {info.name for info in self.inputs}:
                raise ValueError('TensorRT input names do not match model')
            for info in self.inputs:
                value = np.ascontiguousarray(feed[info.name])
                if value.shape != info.shape:
                    raise ValueError(f'{info.name}: expected {info.shape}, received {value.shape}')
                self.buffers[info.name].copy_(torch.from_numpy(value))
            stream = torch.cuda.current_stream(self.device)
            if not self.context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError('TensorRT execute_async_v3 failed')
            names = output_names or [info.name for info in self.outputs]
            if self.allocators:
                stream.synchronize()
            outputs = []
            for name in names:
                if name in self.allocators:
                    allocator = self.allocators[name]
                    if allocator.shape is None or allocator.buffer is None:
                        raise RuntimeError(f'TensorRT did not allocate dynamic output {name}')
                    tensor = allocator.buffer[:int(np.prod(allocator.shape))].reshape(allocator.shape)
                else:
                    tensor = self.buffers[name]
                outputs.append(tensor.cpu().numpy())
            return outputs
