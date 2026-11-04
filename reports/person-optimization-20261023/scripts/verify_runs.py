"""Check complete decoded videos and per-person input/output parity against the baseline."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--diagnostics', action='store_true')
    args = parser.parse_args()
    assert (args.folder/'complete.json').is_file()
    checked = []
    for file in sorted(args.folder.glob('*gpu/*/*/result.json')):
        if file.parent.parent.name.startswith('warmup'):
            continue
        result = json.loads(file.read_text())
        reference = json.loads((ROOT/'reports/person-finetune-20261023/demo/events'/f"{result['name']}.json").read_text())
        reference.sort(key=lambda r:(r['frame_index'],r['track_id']))
        rows = json.loads((file.parent/'events.json').read_text())
        assert len(rows) == len(reference), file
        error = 0.
        for row, old in zip(rows, reference):
            for key in ('track_id', 'frame_index', 'source_frames', 'mode', 'label', 'pose_valid'):
                assert row[key] == old[key], (file, key)
            if args.diagnostics:
                assert row['crop_sha256'] == old['crop_sha256'], file
            for key in ('rgb_probs', 'fused_probs'):
                error = max(error, float(np.abs(np.asarray(row[key])-old[key]).max()))
        assert error < .005, (file,error)
        cap = cv2.VideoCapture(str(file.parent/'video.mp4'))
        assert cap.isOpened()
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            count += 1
        cap.release()
        assert count == result['frames'], file
        checked.append(dict(run=str(file.parent.relative_to(args.folder)), decoded_frames=count,
            events=len(rows), max_probability_error=error, same_labels_modes_sampling=True,
            crop_hashes_checked=args.diagnostics))
    (args.folder/'verification.json').write_text(json.dumps(checked,ensure_ascii=False,indent=2)+'\n')
    print('PASS',len(checked),'videos',sum(x['decoded_frames']for x in checked),'frames',
          sum(x['events']for x in checked),'decisions',max(x['max_probability_error']for x in checked))


if __name__ == '__main__':
    main()
