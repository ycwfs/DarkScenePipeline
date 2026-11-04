"""Verify data/checkpoint provenance, cached predictions and review-page playback."""
import argparse
import hashlib
import json
from pathlib import Path
import py_compile
import numpy as np
from replay import ROOT, OUT, PREVIOUS, save


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifacts() -> None:
    protocol=json.loads((OUT/'protocol.json').read_text())
    prior=json.loads((PREVIOUS/'model-manifest.json').read_text())
    prior_models={str((ROOT if m['path'].startswith('platform/') else PREVIOUS)/m['path']):m['sha256']
                  for m in prior['models']}
    for models in protocol['checkpoints'].values():
        for m in models:
            path=ROOT/m['path']
            assert sha(path)==m['sha256']==prior_models[str(path)],m['path']
    for name in ['inference.py','demos.py','prepare.py']:
        entry=next(s for s in prior['scripts'] if s['path']==f'scripts/{name}')
        assert sha(PREVIOUS/entry['path'])==entry['sha256']
    enhance=ROOT/'platform/dark-demo-v2/ckpts/NTIRE.pth'
    assert sha(enhance)==prior_models[str(enhance)]
    assert not protocol['inference_smoothing'] and not protocol['duration_confirmation']
    frames=0;reused=0;videos=[];lum=[]
    for video in protocol['videos']:
        assert sha(ROOT/video['source'])==video['sha256']
        result=json.loads((OUT/'data'/video['id']/'outputs.json').read_text())
        rows=result['frames'];assert [r['frame'] for r in rows]==video['bright_indices']
        assert result['summary']['original_frames_decoded']==video['frames']
        old=PREVIOUS/'data'/video['id']/'demo.json'
        previous={r['frame']:r for r in json.loads(old.read_text())['frames']} if old.exists() else {}
        for row in rows:
            assert row['luma']>=75;lum.append(row['luma'])
            for name in ['original','candidate']:
                for field in ['boxes','pose_boxes','pose_xy','pose_scores','pose_conf']:
                    assert np.isfinite(np.array(row[name][field])).all()
                    if row['frame'] in previous:
                        assert row[name][field]==previous[row['frame']][name][field]
                assert len(row[name]['pose_xy'])==len(row[name]['pose_scores'])==len(row[name]['pose_conf'])
            if row['frame'] in previous:
                assert abs(row['luma']-previous[row['frame']]['luma'])<1e-7
                reused+=1
        frames+=len(rows)
        if result['summary']['video']:
            assert result['summary']['full_decode']
            videos.append(dict(path=result['summary']['video'],frames=len(rows),sha256=sha(OUT/result['summary']['video'])))
    assert frames==protocol['frames']==3163 and len(protocol['videos'])==41
    for path in (OUT/'scripts').glob('*.py'):py_compile.compile(str(path),doraise=True)
    save(OUT/'verification.json',dict(passed=True,videos=41,frames=frames,luma_range=[min(lum),max(lum)],
        cached_frames_verified=reused,fresh_inference_frames=frames-reused,preview_videos=videos,
        preview_total_frames=sum(v['frames'] for v in videos),
        checks=['source SHA256','four model SHA256 match previous checkpoint manifest','enhancer SHA256',
                'imported inference helpers unchanged','exact cached outputs and luma','all frozen bright frames',
                'finite outputs','no temporal filter','scripts compile','six videos fully decoded during export']))
    print('Artifacts verified',frames,reused)


def browser() -> None:
    from playwright.sync_api import sync_playwright
    errors=[]
    with sync_playwright() as p:
        b=p.chromium.launch(headless=True,args=['--no-sandbox'])
        page=b.new_page(viewport={'width':1440,'height':1000})
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto((OUT/'index.html').as_uri())
        assert page.locator('video').count()==6
        assert page.locator('figure').count()==41
        for i in range(6):
            v=page.locator('video').nth(i)
            v.evaluate('(v)=>{v.muted=true;v.load()}')
            page.wait_for_function('(i)=>document.querySelectorAll("video")[i].readyState>=2',arg=i,timeout=30000)
            v.evaluate('(v)=>v.play()');page.wait_for_timeout(200)
            assert v.evaluate('(v)=>v.currentTime>0 && v.videoWidth===1440 && v.videoHeight===940')
            v.evaluate('(v)=>v.pause()')
        evidence=OUT/'evidence';evidence.mkdir(exist_ok=True)
        page.screenshot(path=str(evidence/'index-desktop.png'))
        page.set_viewport_size({'width':640,'height':950})
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        assert not errors,errors
        save(evidence/'browser-check.json',dict(passed=True,browser=b.version,
            checks=['six real H264 videos play','41 comparison images','1440x940','640px no horizontal overflow','no JS errors']))
        b.close()
    print('Browser verified')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--browser',action='store_true')
    if parser.parse_args().browser:browser()
    else:artifacts()
