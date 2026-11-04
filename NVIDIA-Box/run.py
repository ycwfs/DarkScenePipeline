#!/usr/bin/env python3
"""Native Box launcher, including the isolated Orin memory/performance profile."""
import argparse
import importlib.util
import os
import sys
import sysconfig
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def runtime_environment() -> dict[str, str]:
    env = os.environ.copy()
    nvidia = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    libraries = [str(nvidia / name / "lib") for name in
                 ("cudnn", "cublas", "cuda_runtime") if (nvidia / name / "lib").is_dir()]
    previous = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(libraries + ([previous] if previous else []))
    env.setdefault("TRITON_CACHE_DIR", str(ROOT / ".cache" / "triton"))
    env.setdefault("DARKPIPE_BOX_ENGINE_CACHE", str(ROOT / ".cache" / "tensorrt"))
    env.setdefault("TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT", "256")
    env.setdefault("CUDA_MODULE_LOADING", "LAZY")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONUTF8", "1")
    env.pop("PYTHONPATH", None)
    return env


def build_parser() -> argparse.ArgumentParser:
    spec = importlib.util.spec_from_file_location("_box_platform_main", ROOT / "runtime/main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = module.build_parser()
    parser.prog = "run.py"
    parser.description = "NVIDIA Box 本地完整管线：视频/目录离线处理，或实时流服务与动作片段留存"
    parser.set_defaults(ckpt_dir=str(ROOT / "weights"), gpu_ids="0",
                        clip_dir=str(ROOT / "outputs"), extra_hosts="", session_json=None,
                        sr="off")
    local_help = {
        "gpu_ids": "可见 GPU 编号，默认 0；逐人实时≥3卡：前两张交替前端，其余动作；离线第一张前端、其余动作；编号相对于 CUDA_VISIBLE_DEVICES",
        "ckpt_dir": "权重目录，默认包内 weights/；显式相对路径按当前工作目录解析",
        "clip_dir": "本地结果目录，默认包内 outputs/；离线整段视频和事件也存放于此",
        "stream_formats": "实时出流格式；FLV/HLS 使用本机 FFmpeg",
        "run_seconds": "在线运行时长上限，0 表示持续运行到 Ctrl+C/SIGTERM；离线忽略",
        "session_json": "会话 JSON 保存位置；默认自动生成到输出目录 sessions/ 下",
        "extra_hosts": "可选 WebHDFS 主机映射，默认不设置；仅显式提供时尝试追加 /etc/hosts",
        "hdfs_output_dir": "可选 WebHDFS 上传目录，默认留空，仅保存本地结果",
    }
    for action in parser._actions:
        if action.dest in local_help:
            action.help = local_help[action.dest]
        if action.dest == "session_json":
            action.required = False
    parser.add_argument("--output-dir", dest="clip_dir", default=argparse.SUPPRESS,
                        help="本地结果目录，等价于 --clip_dir；默认包内 outputs/")
    parser.add_argument('--box-profile', choices=['balanced', 'reference'], default='balanced',
                        help='balanced：低内存及异步逐人识别；reference：原512帧缓存与串行调度')
    parser.add_argument('--box-backend', choices=['auto', 'cuda', 'tensorrt'], default='auto',
                        help='auto：ARM64使用JetPack TensorRT，x86_64使用ORT CUDA；均不回退CPU')
    parser.add_argument('--box-cuda-graphs', action='store_true',
                        help='可选CUDA Graph，增加常驻内存，4GB Orin默认关闭')
    parser.add_argument('--box-action-batch', type=int, choices=[1, 2], default=2,
                        help='动作小批次上限，默认2以减少多人排队；显存紧张可降为1')
    return parser


def prepare_argv(argv: list[str] | None = None) -> list[str]:
    parser = build_parser()
    args = parser.parse_args(argv)
    os.environ['DARKPIPE_BOX_PROFILE'] = args.box_profile
    os.environ['DARKPIPE_BOX_BACKEND'] = args.box_backend
    os.environ['DARKPIPE_BOX_GRAPHS'] = '1' if args.box_cuda_graphs else '0'
    os.environ['DARKPIPE_BOX_ACTION_BATCH'] = str(args.box_action_batch)
    del args.box_profile, args.box_backend, args.box_cuda_graphs, args.box_action_batch
    if not args.clip_dir.strip():
        parser.error("盒子部署需要本地结果目录，--clip_dir / --output-dir 不能为空")
    args.clip_dir = str(Path(args.clip_dir).expanduser().resolve())
    args.ckpt_dir = str(Path(args.ckpt_dir).expanduser().resolve())
    if args.session_json is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.session_json = str(Path(args.clip_dir) / "sessions" /
                                f"{stamp}_{os.getpid()}_{uuid.uuid4().hex[:8]}.json")
    elif not args.session_json.strip():
        parser.error("--session_json 不能为空")
    else:
        args.session_json = str(Path(args.session_json).expanduser().resolve())
    return [part for key, value in vars(args).items()
            for part in (f"--{key}", str(value).lower() if isinstance(value, bool) else str(value))]


def main() -> None:
    argv = prepare_argv()
    print(f"[box] profile={os.environ['DARKPIPE_BOX_PROFILE']} backend={os.environ['DARKPIPE_BOX_BACKEND']} "
          f"cuda_graphs={os.environ['DARKPIPE_BOX_GRAPHS']}；模型输入长边720，默认直接输出增强图", flush=True)
    print(f"[box] 本地完整流程；会话记录 -> {argv[argv.index('--session_json') + 1]}", flush=True)
    os.execve(sys.executable, [sys.executable, "-u", str(ROOT / "runtime/main.py"), *argv],
              runtime_environment())


if __name__ == "__main__":
    main()
