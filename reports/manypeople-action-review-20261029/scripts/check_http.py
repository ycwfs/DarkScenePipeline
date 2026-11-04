"""Verify the actual review server's byte-range delivery and browser playback."""
import argparse
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    base = f'http://127.0.0.1:{args.port}'
    file = ROOT/'clips/M01-014.mp4'
    expected = file.read_bytes()
    cases = [('bytes=0-1023',0,1023), ('bytes=1024-2047',1024,2047),
             ('bytes=-512',len(expected)-512,len(expected)-1),
             (f'bytes={len(expected)-512}-',len(expected)-512,len(expected)-1)]
    for value, start, end in cases:
        with urlopen(Request(base+'/clips/M01-014.mp4', headers={'Range':value})) as response:
            assert response.status == 206
            assert response.headers['Content-Range'] == f'bytes {start}-{end}/{len(expected)}'
            assert response.headers['Content-Type'] == 'video/mp4'
            assert response.read() == expected[start:end+1]
    for value in ('bytes=9999999999999-', 'bytes=-0', 'bytes=10-5'):
        try:
            urlopen(Request(base+'/clips/M01-014.mp4', headers={'Range':value}))
        except HTTPError as exc:
            assert exc.code == 416
        else:
            raise AssertionError(f'Invalid range accepted: {value}')
    with urlopen(Request(base+'/clips/M01-014.mp4', method='HEAD')) as response:
        assert response.status == 200 and int(response.headers['Content-Length']) == len(expected)
        assert not response.read()
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={'width':1440,'height':950})
        page = context.new_page()
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(base+'/index.html')
        page.wait_for_load_state('networkidle')
        assert page.locator('#count').inner_text() == '当前 473 条'
        page.locator('#search').fill('M01-014')
        page.wait_for_function('document.querySelector("video").readyState>=2')
        page.locator('#video').evaluate('(v)=>v.play()')
        page.wait_for_function('document.querySelector("video").currentTime>0.2')
        page.locator('#video').evaluate('(v)=>v.pause()')
        page.locator('#raw').click()
        page.wait_for_function('document.querySelector("video").readyState>=2 && document.querySelector("video").currentTime>140')
        page.locator('#video').evaluate('(v)=>{v.currentTime=150;return v.play()}')
        page.wait_for_function('document.querySelector("video").currentTime>150.2')
        page.locator('#video').evaluate('(v)=>v.pause()')
        assert page.locator('#video').evaluate('(v)=>v.error') is None
        page.screenshot(path=str(ROOT/'ui-http.png'), full_page=True)
        context.close()
        browser.close()
    assert not errors, errors
    result = dict(passed=True, url=base+'/index.html', range_cases=4, invalid_ranges=3,
                  head=True, browser_clip_playback=True, browser_raw_seek_150_seconds=True,
                  javascript_errors=errors, annotation_data_unchanged=True,
                  limitation='Local HTTP/browser verified; user VS Code forwarding must be configured on their client.')
    (ROOT/'http-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()
