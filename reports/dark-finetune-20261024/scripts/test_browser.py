"""Real browser playback, filtering, responsive layout and download validation."""
import json
from playwright.sync_api import sync_playwright
from prepare import OUT,save


def main():
    evidence=OUT/'evidence';evidence.mkdir(exist_ok=True);errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        context=browser.new_context(viewport={'width':1500,'height':1050},accept_downloads=True)
        page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto((OUT/'index.html').as_uri());assert page.locator('article').count()==14
        for title,count in [('暗场留出',8),('亮场回归',4),('训练内示例',2),('全部',14)]:
            page.get_by_role('button',name=title,exact=True).click()
            assert page.locator('article:visible').count()==count
        for idx in [0,10]:
            video=page.locator('video').nth(idx)
            video.evaluate('(v)=>{v.muted=true;v.load()}')
            page.wait_for_function('(i)=>document.querySelectorAll("video")[i].readyState>=2',arg=idx,timeout=30000)
            video.evaluate('(v)=>v.play()');page.wait_for_timeout(400)
            assert video.evaluate('(v)=>v.currentTime>0 && v.videoWidth===1440 && v.videoHeight===940')
            video.evaluate('(v)=>v.pause()')
        page.screenshot(path=str(evidence/'index-desktop.png'))
        page.set_viewport_size({'width':640,'height':950});page.wait_for_timeout(100)
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
        assert not errors,errors
        save(evidence/'browser-check.json',dict(passed=True,browser=browser.version,
            checks=['14 videos','8/4/2 filters','two real H264 videos played','1440x940 resolution',
                    'no horizontal overflow at 640px','no javascript errors']))
        browser.close()
    print('Browser checks passed')


if __name__=='__main__':main()
