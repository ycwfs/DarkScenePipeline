"""Run the real offline operator with a CPU model substitute and a failed clip writer."""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--op-dir', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.op_dir)
    import main as operator
    from darkpipe import clips, config, pipeline
    from darkpipe.stages.base import RecognitionEvent

    class Recognizer:
        name, window, stride = 'test-cpu', 1, 1

        def load(self, device):
            pass

        def push(self, frame, idx, ts, **kw):
            return RecognitionEvent(idx, ts, 'wave', .9)

        def close(self):
            pass

    class BrokenWriter:
        def __init__(self, path, fps):
            Path(path).write_bytes(b'recoverable-stage')
            self.count = 0

        def write(self, frame):
            raise OSError('probe clip disk failure')

        def close(self):
            pass

    pipeline.build_stages = lambda cfg: ([], Recognizer())
    clips.VideoWriter = BrokenWriter
    config.validate = lambda cfg: cfg
    operator.parse_gpu_ids = lambda value: ['0']
    operator._print_gpu_inventory = lambda gpus: None
    operator.apply_extra_hosts = lambda spec: 0
    operator.main(['--video_path', args.input, '--session_json', args.out,
                   '--clip_dir', str(Path(args.out).parent / 'delivered'),
                   '--enhance', 'off', '--sr', 'off', '--denoise', 'off',
                   '--clip_denoise', 'off', '--label_bar', 'false'])


if __name__ == '__main__':
    main()
