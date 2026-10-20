import json, py_compile, subprocess, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/operator')
import main
from darkpipe.stages.downscale import DownscaleStage
from darkpipe.stages.sr_bicubic import BicubicStage
from darkpipe.render import append_label_bar
from darkpipe.media import VideoWriter
args = main.build_parser().parse_args(['--video_path', 'probe', '--session_json', '/evidence/session.json'])
manifest = json.loads(Path('/operator/suanzi.json').read_text())
assert args.proc_max_side == 720 == next(i['default'] for i in manifest['inputs'] if i['name'] == 'proc_max_side')
assert args.sr == 'bicubic' and args.sr_scale == 2 and args.label_bar
assert args.clip_pre_sec == args.clip_post_sec == 2.0
py_compile.compile('/operator/main.py', cfile='/evidence/main.pyc', doraise=True)
help_text = subprocess.check_output([sys.executable, '/operator/main.py', '--help'], text=True)
assert '默认720' in help_text
Path('/evidence/help.txt').write_text(help_text)
frame = np.full((1080, 1920, 3), (32, 64, 96), np.uint8)
processed = DownscaleStage(args.proc_max_side)([frame])[0]
sr = BicubicStage(args.sr_scale)([processed])[0]
out = append_label_bar(sr, None)
assert processed.shape == (404, 720, 3)
assert sr.shape == (808, 1440, 3)
assert out.shape == (874, 1440, 3)
writer = VideoWriter('/evidence/dimensions.mp4', 15)
try:
    for _ in range(15):
        writer.write(out)
finally:
    writer.close()
ffprobe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
    '-select_streams', 'v:0', '-show_entries', 'stream=codec_name,width,height,nb_read_frames,pix_fmt',
    '-of', 'json', '/evidence/dimensions.mp4'], text=True))
stream = ffprobe['streams'][0]
assert (stream['width'], stream['height']) == (1440, 874)
assert stream['codec_name'] == 'h264' and stream['pix_fmt'] == 'yuv420p'
assert int(stream['nb_read_frames']) == 15
subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', '/evidence/dimensions.mp4', '-f', 'null', '-'], check=True)
report = dict(defaults=dict(proc_max_side=args.proc_max_side, sr=args.sr, sr_scale=args.sr_scale,
    label_bar=args.label_bar, clip_pre_sec=args.clip_pre_sec, clip_post_sec=args.clip_post_sec,
    clip_denoise=args.clip_denoise, reco_min_conf=args.reco_min_conf),
    input_wh=[1920, 1080], processing_wh=list(processed.shape[1::-1]),
    sr_wh=list(sr.shape[1::-1]), output_wh=list(out.shape[1::-1]), ffprobe=ffprobe,
    decode='passed', scope='CPU sizing and real H.264 encoding; no model or throughput benchmark')
Path('/evidence/dimensions.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
