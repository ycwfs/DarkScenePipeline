"""Check completed live replay accounting and fully decode the saved H.264 outputs."""
import json
from pathlib import Path

import cv2

OUT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def main():
    cv2.setNumThreads(1)
    checked = []
    for folder in ('live', 'live-extra', 'soak'):
        base = OUT / folder
        assert read(base / 'complete.json')['completed']
        for file in sorted(base.glob('*gpu/*/result.json')):
            if file.parent.name.startswith('warmup'):
                continue
            result = read(file)
            frames = read(file.parent / 'frames.json')
            events = read(file.parent / 'events.json')
            indices = result['source_indices']
            assert len(indices) == len(set(indices)) == len(frames) == result['frames'], file
            assert indices == sorted(indices) == [f['frame'] for f in frames], file
            assert len(indices) + result['dropped_source_frames'] == result['source_frames_offered'], file
            assert result['submitted'] == result['completed'] + result['coalesced'], file
            assert result['completed'] == len(events), file
            assert all(e['frame_index'] == e['source_frames'][-1] for e in events), file
            cap = cv2.VideoCapture(str(file.parent / 'video.mp4'))
            assert cap.isOpened(), file
            decoded = 0
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                assert image.shape[1::-1] == tuple(frames[decoded]['output_size']), file
                decoded += 1
            cap.release()
            assert decoded == result['frames'], file
            checked.append(dict(run=str(file.parent.relative_to(OUT)), decoded_frames=decoded,
                                events=len(events), accounting_passed=True))
    target = OUT / 'evidence/live-verification.json'
    target.write_text(json.dumps(checked, ensure_ascii=False, indent=2) + '\n')
    print('PASS', len(checked), 'videos;', sum(r['decoded_frames'] for r in checked),
          'frames;', sum(r['events'] for r in checked), 'decisions')


if __name__ == '__main__':
    main()
