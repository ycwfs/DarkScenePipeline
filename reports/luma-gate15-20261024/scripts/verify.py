"""Verify full coverage, real gate semantics, immutable models and playable artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

import numpy as np
from evaluate import ROOT, OUT, save, sha, fuse_person, BEHAVIORS


def artifacts() -> None:
    protocol=json.loads((OUT/'protocol.json').read_text());digest=sha(OUT/'protocol.json')
    results=json.loads((OUT/'results.json').read_text());assert results['complete']
    assert results['additional_labels_sha256']==sha(OUT/'label-review.json')
    assert json.loads((OUT/'pose-runtime-parity.json').read_text())['passed']
    extra_ids={v['id']for v in json.loads((OUT/'extra-demos.json').read_text())['videos']}if (OUT/'extra-demos.json').exists()else set()
    events=frames=0;videos=[]
    for name,m in protocol['models'].items():assert sha(ROOT/m['path'])==m['sha256'],name
    for v in protocol['videos']:
        assert sha(ROOT/v['source'])==v['sha256']
        d=json.loads((OUT/'data'/v['id']/'result.json').read_text());assert d['protocol_sha256']==digest
        assert len(d['frames'])==v['frames'];frames+=v['frames']
        for gate in (75,15):
            for e in d['events'][str(gate)]:
                ls=[d['frames'][i]['luma']for i in e['source_frames']]
                assert all(abs(t-i/v['fps'])<1e-8 for t,i in zip(e['timestamps'],e['source_frames']))
                sk=None if e['skeleton_probs']is None else np.array(e['skeleton_probs'],np.float32)
                final=fuse_person(np.array(e['rgb_probs'],np.float32),sk,ls,lum_gate=gate)
                assert e['mode']==final['mode']and e['label']==final['label']
                assert np.max(np.abs(np.array(e['fused_probs'])-final['fused_probs']))<1e-7
                assert e['pose_valid']==(e['mode']=='rgb_skeleton')
                if e['mode']=='rgb_soft_no_pose':
                    assert max(e['fused_probs'][:9])<=.700001 and e['label']=='other'
                if e['label']!='other':assert e['confidence']>=.75
                events+=1
        if v['demo']or v['id']in extra_ids:
            vp=OUT/'videos'/f'{v["id"]}.mp4'
            check=subprocess.run(['ffmpeg','-v','error','-threads','2','-i',str(vp),'-f','null','-'],capture_output=True,check=True)
            assert not check.stderr,check.stderr
            meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
                'stream=width,height,nb_frames,r_frame_rate','-of','json',str(vp)]))['streams'][0]
            assert (meta['width'],meta['height'],int(meta['nb_frames']))==(1440,488,v['frames'])
            videos.append(dict(path=str(vp.relative_to(OUT)),sha256=sha(vp),frames=v['frames']))
    for p in (OUT/'scripts').glob('*.py'):compile(p.read_text(),str(p),'exec')
    bundle=json.loads((OUT/'bundle.json').read_text());assert sha(OUT/bundle['path'])==bundle['sha256']
    with zipfile.ZipFile(OUT/bundle['path'])as z:
        assert z.testzip()is None
        assert set(z.namelist())==set(bundle['members'])
        for name in z.namelist():assert hashlib.sha256(z.read(name)).hexdigest()==sha(OUT/name)
    sources=[ROOT/'reports/person-action-20261023/scripts/person_core.py',
        ROOT/'reports/person-action-20261023/scripts/demo.py',
        ROOT/'reports/dark-finetune-20261024/scripts/inference.py',
        ROOT/'platform/dark-demo-v2/darkpipe/stages/recognize.py',
        ROOT/'platform/dark-demo-v2/darkpipe/stages/skeleton_aux.py',
        ROOT/'platform/dark-demo-v2/darkpipe/stages/downscale.py',
        ROOT/'platform/dark-demo-v2/darkpipe/stages/enhance_retinexformer.py',
        ROOT/'reports/person-finetune-20261023/scripts/assemble.py']
    evidence=OUT/'evidence';evidence.mkdir(exist_ok=True)
    save(evidence/'implementation.json',dict(files=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p))for p in sources],
        experiment_scripts=[dict(path=str(p.relative_to(OUT)),sha256=sha(p))for p in (OUT/'scripts').glob('*.py')]))
    save(OUT/'verification.json',dict(passed=True,sources=len(protocol['videos']),frames=frames,events=events,videos=videos,
        checks=['all source/model hashes unchanged','full source frame coverage','actual fuse_person re-evaluated per event',
            'raw luma and timestamps used','non-other threshold enforced','no-pose soft ceiling confirmed',
            'all preview videos fully decoded','ZIP CRC, SHA and all member bytes equal','scripts compile']))
    print('Verified',frames,'source frames,',events,'events,',len(videos),'videos',flush=True)


def browser() -> None:
    from playwright.sync_api import sync_playwright
    evidence=OUT/'evidence';evidence.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='gate15-browser-')as tmp:
        with zipfile.ZipFile(OUT/'gate15-review.zip')as z:z.extractall(tmp)
        with sync_playwright()as p:
            b=p.chromium.launch(headless=True,args=['--no-sandbox'])
            page=b.new_page(viewport={'width':1440,'height':1000});errors=[]
            page.on('pageerror',lambda error:errors.append(str(error)))
            page.goto((Path(tmp)/'index.html').as_uri())
            n=page.locator('video').count()
            for i in range(n):
                v=page.locator('video').nth(i);v.scroll_into_view_if_needed()
                v.evaluate('(v)=>{v.muted=true;v.load()}')
                page.wait_for_function('(i)=>document.querySelectorAll("video")[i].readyState>=2',arg=i,timeout=30000)
                v.evaluate('(v)=>v.play()')
                page.wait_for_function('(i)=>document.querySelectorAll("video")[i].currentTime>0.08',arg=i,timeout=15000)
                v.evaluate('(v)=>v.pause()')
            page.evaluate('scrollTo(0,0)');page.screenshot(path=str(evidence/'index-desktop.png'))
            page.set_viewport_size({'width':390,'height':844})
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
            assert not errors,errors
            save(evidence/'browser-check.json',dict(passed=True,videos_played=n,extracted_zip=True,no_js_errors=True,mobile_no_overflow=True))
            b.close()
    print('Browser verified',n,'actual video playbacks',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--browser',action='store_true')
    browser()if p.parse_args().browser else artifacts()
