"""Freeze the proposed three-frame rule before opening its performance results."""
import json
from common import OUT, ROOT, OP, digest, save


def main():
    path=OUT/'protocol.json'
    assert not path.exists(), 'Protocol already frozen'
    previous=json.loads((ROOT/'reports/four-parameter-tuning-20261026/protocol.json').read_text())
    save(path,dict(primary_rule=dict(min_frames=3,keypoint_conf=.5,consecutive=False,
        criterion='Same wrist y strictly less than BOTH shoulder y coordinates in >=3 distinct source frames of the existing 1s window.',
        unknown_or_dark='Skip under original raw-luma gate35 or unknown raw brightness; do not force pose in darkness.',
        missing_pose='Bright wave candidates without enough confident shoulder/wrist observations become other; count this as potential added miss.',
        duplicate_frames='Deduplicate original frame IDs; never count repeated 32-frame model samples twice.',
        association='Existing same-track COCO17 pose, no new detector, smoothing, cross-person evidence or extra lookahead.'),
        comparison_counts=[1,3,5,8],comparison_modes=['total_same_wrist','consecutive_observations_gap_le_120ms'],
        primary='Three total frames fixed per user proposal; other settings diagnostic. Recommendation uses old validation first; historical test/new full videos are reported separately, not used to silently retune.',
        fixed=dict(alpha=.972,lum_gate=35.,action=.925,object=.581,enhanced_size=720,object_model='official seven-class; no COCO80 or eight-class candidate'),
        models=previous['models'],old_sources=previous['sources'],
        long_videos=[1,7,9],short_videos='Exact previous new01/07/09 excerpts plus old onsite wave positives selected by prior truth/source split, before applying the new rule.',
        scope='Finite offline paired full-pipeline outputs and clips. No new realtime/NFS/infinite-stream claim.',
        files={str(p.relative_to(ROOT)):digest(p)for p in [OUT/'implementation/wave_prior.py',OP/'darkpipe/person/action_engine.py',OP/'darkpipe/person/core.py',OP/'darkpipe/person/object_runtime.py',ROOT/'reports/four-parameter-tuning-20261026/inputs.npz',ROOT/'reports/four-parameter-tuning-20261026/predictions.json']},
        production_updated=False))
    print('Frozen 3-frame primary rule, 1/3/5/8 diagnostic grid, original seven-class object detector')


if __name__=='__main__':main()
