"""Verify source pixels, brightness labels and portable offline-page playback."""
import argparse
import json
from pathlib import Path
import py_compile
import tempfile
import zipfile

import cv2
import numpy as np
from build_samples import ROOT, OUT, BINS, downscale, frame_luma, save, sha


def artifacts() -> None:
    samples = json.loads((OUT/'samples.json').read_text())['samples']
    protocol = json.loads((OUT/'protocol.json').read_text())
    assert len(samples) == 18 and sum(r['clip_frames'] for r in samples) == 1372
    assert sum(b['frames'] for b in protocol['bins']) == protocol['all_source_frames'] == 42193
    assert sha(ROOT/protocol['enhancer']['path']) == protocol['enhancer']['sha256']
    checks = []
    for r in samples:
        assert r['split'] == 'train'
        source = ROOT/r['source']
        assert sha(source) == r['source_sha256']
        for path in r['files'].values():
            assert sha(OUT/path) == r['file_sha256'][Path(path).name]
        cap = cv2.VideoCapture(str(source))
        for i in range(r['poster_frame']+1):
            ok, native = cap.read()
            assert ok, (r['source'], i)
        cap.release()
        raw = downscale(native)
        stored = cv2.imread(str(OUT/r['files']['raw']))
        assert np.array_equal(raw, stored), r['sample_id']
        luma = frame_luma(stored)
        assert abs(luma-r['poster_luma']) < 1e-8
        lo, hi = BINS[r['bin_index']]
        assert lo <= luma < hi
        assert [f['frame'] for f in r['frames']] == list(range(r['frame_start'], r['frame_end_exclusive']))
        lumas = [f['luma'] for f in r['frames']]
        assert abs(np.mean(lumas)-r['mean_luma']) < 1e-8
        assert min(lumas) == r['min_luma'] and max(lumas) == r['max_luma']
        video = cv2.VideoCapture(str(OUT/r['files']['video']))
        assert video.isOpened() and video.get(cv2.CAP_PROP_FRAME_WIDTH) == 1440 and video.get(cv2.CAP_PROP_FRAME_HEIGHT) == 498
        assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == r['clip_frames']
        assert abs(video.get(cv2.CAP_PROP_FPS)-r['fps']) < .01
        video.release()
        assert r['full_decode']
        checks.append(dict(sample_id=r['sample_id'], source_sha256=r['source_sha256'], luma=luma,
                           raw_pixels_exact=True, frames=r['clip_frames']))
    for script in (OUT/'scripts').glob('*.py'):
        py_compile.compile(str(script), doraise=True)
    bundle = json.loads((OUT/'bundle.json').read_text())
    assert sha(OUT/bundle['path']) == bundle['sha256']
    with zipfile.ZipFile(OUT/bundle['path']) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == bundle['members']
        for name in z.namelist():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            assert z.read(name) == (OUT/name).read_bytes()
    save(OUT/'verification.json', dict(passed=True, samples=18, frames=1372, checks=checks,
        tests=['18 original source hashes', 'all sample asset hashes', 'lossless raw PNG equals decoded source resized pixels',
               'same luma as production function', 'correct bins and contiguous source frames',
               'fps, resolution and frame count', 'all H264 videos fully decoded during build',
               'ZIP CRC and all members equal original files', 'scripts compile', 'enhancer unchanged']))
    print('Verified 18 samples,1372 frames and portable ZIP')


def browser() -> None:
    from playwright.sync_api import sync_playwright
    samples = json.loads((OUT/'samples.json').read_text())['samples']
    errors = []
    with tempfile.TemporaryDirectory(prefix='luma-review-browser-', dir='/tmp') as td:
        with zipfile.ZipFile(OUT/'luma-samples-review.zip') as z:
            assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
            z.extractall(td)
        with sync_playwright() as p:
            b = p.chromium.launch(headless=True, args=['--no-sandbox'])
            page = b.new_page(viewport={'width':1440,'height':1100})
            page.on('pageerror', lambda e:errors.append(str(e)))
            page.goto((Path(td)/'index.html').as_uri())
            assert page.locator('article').count() == 18
            for bi in range(6):
                page.locator(f'[data-filter="{bi}"]').click()
                assert page.locator('article:visible').count() == 3
            page.locator('[data-filter="all"]').click()
            assert page.locator('article:visible').count() == 18
            for threshold in (75,60,45,30,15,0):
                page.locator(f'[data-threshold="{threshold}"]').click()
                for r in samples:
                    label = page.locator(f'[data-sample="{r["sample_id"]}"] .gate-state').inner_text()
                    expected = ('不按亮度区分' if threshold == 0 else
                                f'全部帧低于 {threshold}' if r['max_luma'] < threshold else
                                f'全部帧不低于 {threshold}' if r['min_luma'] >= threshold else
                                f'跨越 {threshold}')
                    assert expected in label, (r['sample_id'], threshold, label)
            for i in range(18):
                details = page.locator('details').nth(i)
                details.locator('summary').click()
                video = details.locator('video')
                video.evaluate('(v)=>{v.muted=true;v.load()}')
                page.wait_for_function('(i)=>document.querySelectorAll("video")[i].readyState>=2', arg=i, timeout=30000)
                video.evaluate('(v)=>v.play()')
                page.wait_for_function('(i)=>document.querySelectorAll("video")[i].currentTime>0.08', arg=i, timeout=15000)
                assert video.evaluate('(v)=>v.videoWidth===1440 && v.videoHeight===498')
                details.locator('summary').click()
                page.wait_for_function('(i)=>document.querySelectorAll("video")[i].paused', arg=i)
            # Check one PNG link as served from the extracted package.
            page.locator('[data-filter="2"]').click()
            page.locator('[data-threshold="45"]').click()
            page.locator('#band-2').scroll_into_view_if_needed()
            page.wait_for_function('Array.from(document.querySelectorAll("#band-2 img")).every(i=>i.complete&&i.naturalWidth>0)')
            evidence = OUT/'evidence'
            evidence.mkdir(exist_ok=True)
            page.locator('#band-2').screenshot(path=str(evidence/'samples-desktop.png'))
            page.evaluate('window.scrollTo(0,0)')
            page.screenshot(path=str(evidence/'index-desktop.png'))
            page.set_viewport_size({'width':390,'height':844})
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
            page.locator('#band-2').scroll_into_view_if_needed()
            page.screenshot(path=str(evidence/'index-mobile.png'))
            assert not errors, errors
            save(evidence/'browser-check.json', dict(passed=True, browser=b.version, extracted_zip=True,
                samples=18, videos_played=18, checks=['six filters return3 samples each', 'six threshold previews correct for all18 samples',
                    'all18 H264 videos play', 'closing details pauses video', 'images load offline', '390px responsive layout', 'no JS errors']))
            b.close()
    print('Portable page:18 actual video playbacks, filters and threshold labels passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--browser', action='store_true')
    browser() if parser.parse_args().browser else artifacts()
