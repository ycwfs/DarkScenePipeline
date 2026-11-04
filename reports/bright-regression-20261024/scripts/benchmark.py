"""Paired bright-frame front-end timing; excludes enhancement/action/streaming."""
import json
import time
import cv2
import numpy as np
from replay import Frontend, OUT, PREVIOUS, ROOT, save


def main() -> None:
    cv2.setNumThreads(1)
    videos = json.loads((OUT / 'protocol.json').read_text())['videos']
    images = [cv2.imread(str(OUT/'data'/v['id']/'enhanced.png')) for v in videos]
    assert all(im is not None for im in images)
    op = ROOT/'platform/dark-demo-v2/ckpts'
    models = {'original': Frontend(op/'yolo26m_detect.onnx',op/'rtmo-s_end2end.onnx'),
              'candidate': Frontend(PREVIOUS/'models/yolo26m_dark_candidate.onnx',PREVIOUS/'models/rtmo-s_dark_candidate.onnx')}
    for model in models.values():
        for image in images[:5]:
            model.detector._frames_to_dets([image])
            model.poses(image)
    rows = []
    for repeat in range(5):
        for i, image in enumerate(images):
            names = ['original','candidate'] if (i+repeat)%2 else ['candidate','original']
            for name in names:
                model = models[name]
                start = time.perf_counter()
                model.detector._frames_to_dets([image])
                detector = (time.perf_counter()-start)*1000
                start = time.perf_counter()
                model.poses(image)
                pose = (time.perf_counter()-start)*1000
                rows.append(dict(repeat=repeat, video_id=videos[i]['id'], model=name, detector_ms=detector, pose_ms=pose))
    summary = {}
    for name in models:
        summary[name] = {}
        for metric in ['detector_ms','pose_ms']:
            values = [r[metric] for r in rows if r['model']==name]
            summary[name][metric] = dict(n=len(values),mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)),
                median=float(np.median(values)),p95=float(np.percentile(values,95)))
    save(OUT/'benchmark.json',dict(rows=rows,summary=summary,
        hardware='Single idle RTX3090; CUDA_VISIBLE_DEVICES=2', warmup_frames=5,repeats=5,
        scope='preprocessing+ORT+NMS; excludes enhancement, actions, association, encoding and networking; not end-to-end FPS'))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
