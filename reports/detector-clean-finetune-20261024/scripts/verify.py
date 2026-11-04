"""Verify exact model/data provenance, baseline reuse, videos and browser playback."""
import argparse
import hashlib
import json
from pathlib import Path
import py_compile

import cv2
import numpy as np
from compare import ROOT, OUT, OLD, BRIGHT, NAMES, save


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifacts() -> None:
    protocol = json.loads((OUT/'protocol.json').read_text())
    selected = json.loads((OUT/'selection.json').read_text())['selected']
    target = OUT/'models/yolo26m_clean_candidate.pt'
    assert sha(target) == sha(Path(selected['path']))
    export = json.loads((OUT/'export-verification.json').read_text())
    assert export['passed'] and len(export['checks']) == 10
    previous_manifest = json.loads((OLD/'model-manifest.json').read_text())
    models = []
    for entry in previous_manifest['models']:
        if entry['path'].startswith('platform/') or entry['path'] in (
                'models/yolo26m_dark_candidate.onnx', 'models/rtmo-s_dark_candidate.onnx'):
            path = (ROOT if entry['path'].startswith('platform/') else OLD)/entry['path']
            assert sha(path) == entry['sha256']
            models.append(dict(path=str(path.relative_to(ROOT)), sha256=entry['sha256'], bytes=path.stat().st_size))
    for path in sorted((OUT/'models').glob('*')):
        if path.is_file():
            models.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size))
    for relative, digest in protocol['input_hashes'].items():
        assert sha(ROOT/relative) == digest
    inventory = json.loads((ROOT/'reports/dark-annotations-20261023/inventory.json').read_text())['raw_videos']
    source_hashes = {v['source']: v['sha256'] for v in inventory}
    replay = json.loads((OUT/'replay-summary.json').read_text())
    assert len(replay['videos']) == 49
    assert len(replay['pixel_checks']) == 41
    assert max(c['max_pixel_delta'] for c in replay['pixel_checks']) <= 1
    frames = 0
    videos = []
    for v in replay['videos']:
        assert sha(ROOT/v['source']) == source_hashes[v['source']]
        result = json.loads((OUT/'data/test'/v['id']/'outputs.json').read_text())
        assert result['candidate_sha256'] == sha(OUT/'models/yolo26m_clean_candidate.onnx')
        old_path = BRIGHT/'data'/v['id']/'outputs.json' if v['domain'] == 'bright' else OLD/'data'/v['id']/'demo.json'
        old = {r['frame']: r for r in json.loads(old_path.read_text())['frames']}
        assert [r['frame'] for r in result['frames']] == v['indices']
        for row in result['frames']:
            reference = old[row['frame']]
            assert row['original']['boxes'] == reference['original']['boxes']
            assert row['previous']['boxes'] == reference['candidate']['boxes']
            assert row['fixed_pose'] == {k: reference['candidate'][k] for k in row['fixed_pose']}
            assert (row['luma'] < 75) == (v['domain'] == 'dark')
            for name in NAMES:
                assert np.isfinite(np.array(row[name]['boxes'])).all()
        frames += len(result['frames'])
        if v['video']:
            path = OUT/v['video']
            cap = cv2.VideoCapture(str(path))
            assert cap.isOpened()
            assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 2160
            assert cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == 494
            assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(result['frames'])
            assert abs(cap.get(cv2.CAP_PROP_FPS)-v['fps']) < 1e-3
            cap.release()
            assert v['full_decode']
            videos.append(dict(path=v['video'], frames=len(result['frames']), sha256=sha(path)))
    assert frames == 3885
    dataset = json.loads((OUT/'dataset-manifest.json').read_text())
    groups = {sp: {r['group'] for r in dataset['records'] if r['split'] == sp} for sp in ('train', 'val', 'test')}
    assert all(not (groups[a] & groups[b]) for a, b in [('train','val'),('train','test'),('val','test')])
    for r in dataset['records']:
        image = Path(r['image'])
        label = Path(str(image).replace('/images/','/labels/')).with_suffix('.txt')
        boxes = np.loadtxt(label, ndmin=2)
        assert len(boxes) == r['boxes'] and np.all(boxes[:,0] == 0)
        assert np.isfinite(boxes).all() and np.all((boxes[:,1:] >= 0) & (boxes[:,1:] <= 1))
        assert image.is_file()
    scripts = []
    for path in (OUT/'scripts').glob('*.py'):
        py_compile.compile(str(path), doraise=True)
        scripts.append(dict(path=str(path.relative_to(OUT)), sha256=sha(path)))
    save(OUT/'model-manifest.json', dict(models=models, scripts=scripts, selected=str(selected['path']),
        production_weights_verified_unchanged=True))
    save(OUT/'verification.json', dict(passed=True, frames=frames, source_videos=49,
        preview_videos=videos, preview_frames=sum(v['frames'] for v in videos),
        pixel_checks=replay['pixel_checks'], checks=['PT equals validation-selected epoch', '10-image PT/ONNX comparison',
            'baseline and frozen model hashes', 'historical labels unchanged', '49 source video hashes',
            'all 3885 cached baseline/pose outputs exact', 'split group separation', 'all labels finite and normalized',
            'H264 full decode on export, resolution/fps/frame count', 'Python compile']))
    print('Artifacts verified:', frames, 'frames,', len(videos), 'videos')


def browser() -> None:
    from playwright.sync_api import sync_playwright
    expected = [v for v in json.loads((OUT/'replay-summary.json').read_text())['videos'] if v['video']]
    errors = []
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=['--no-sandbox'])
        page = b.new_page(viewport={'width':1440, 'height':1000})
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto((OUT/'index.html').as_uri())
        assert page.locator('video').count() == len(expected)
        assert page.locator('figure').count() == 49
        for i in range(len(expected)):
            v = page.locator('video').nth(i)
            v.evaluate('(v)=>{v.muted=true;v.load()}')
            page.wait_for_function('(i)=>document.querySelectorAll("video")[i].readyState>=2', arg=i, timeout=30000)
            v.evaluate('(v)=>v.play()')
            page.wait_for_timeout(200)
            assert v.evaluate('(v)=>v.currentTime>0 && v.videoWidth===2160 && v.videoHeight===494')
            v.evaluate('(v)=>v.pause()')
        evidence = OUT/'evidence'
        evidence.mkdir(exist_ok=True)
        page.screenshot(path=str(evidence/'index-desktop.png'))
        page.set_viewport_size({'width':640,'height':950})
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        assert not errors, errors
        save(evidence/'browser-check.json', dict(passed=True, browser=b.version, played_videos=len(expected),
            images=49, checks=['H264 actual playback', '2160x494', '640px responsive layout', 'no JS errors']))
        b.close()
    print('Browser playback verified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--browser', action='store_true')
    browser() if parser.parse_args().browser else artifacts()
