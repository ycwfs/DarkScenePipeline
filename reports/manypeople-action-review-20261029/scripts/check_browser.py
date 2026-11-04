"""Exercise local-file review persistence, export/import and actual media seeking."""
import json
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

from collect import OUT, read, save
from import_labels import validate


def main():
    errors=[];checks=[]
    with tempfile.TemporaryDirectory(prefix='manypeople-ui-') as tmp, sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        context=browser.new_context(viewport={'width':1440,'height':1000},accept_downloads=True)
        page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto((OUT/'index.html').as_uri());page.wait_for_load_state('networkidle')
        assert page.locator('#count').inner_text()=='当前 473 条'
        page.wait_for_function('document.querySelector("video").readyState>=2')
        page.locator('#video').evaluate('(v)=>v.play()');page.wait_for_function('document.querySelector("video").currentTime>.2')
        page.locator('#video').evaluate('(v)=>v.pause()')
        assert page.locator('#video').evaluate('(v)=>v.error') is None
        checks.append('file:// initial page and real H264 preview playback')
        page.locator('#note').fill('测试专用：逗号, 引号"；换行\n不是真实人工标注')
        page.locator('#correct').click();page.reload();page.wait_for_load_state('networkidle')
        assert '已保存' in page.locator('#message').inner_text()
        assert '不是真实人工标注' in page.locator('#note').input_value()
        page.locator('#next').click();page.locator('#other').click()
        page.locator('#next').click();page.locator('#exclude').click()
        page.locator('#next').click();page.locator('#uncertain').click()
        with page.expect_download() as d:page.locator('#exportjson').click()
        export=Path(tmp)/'review.json';d.value.save_as(str(export))
        payload=read(export);assert len(payload['annotations'])==4
        validated=validate(export);assert len(validated)==4
        assert sorted(x['review_status']for x in validated)==['confirmed','confirmed','exclude','uncertain']
        with page.expect_download() as d:page.locator('#exportcsv').click()
        csvpath=Path(tmp)/'review.csv';d.value.save_as(str(csvpath));assert len(validate(csvpath))==4
        checks.append('four review states, persistence, unicode notes, JSON/CSV export and Python validation')
        page.evaluate('localStorage.clear()');page.reload();page.wait_for_load_state('networkidle')
        page.locator('#importfile').set_input_files(str(export))
        page.wait_for_function('document.querySelector("#message").textContent.includes("已导入 4")')
        invalid={**payload,'manifest_sha256':'invalid'};bad=Path(tmp)/'bad.json';bad.write_text(json.dumps(invalid))
        page.locator('#importfile').set_input_files(str(bad));page.wait_for_function('document.querySelector("#message").textContent.includes("导入失败")')
        assert '已确认／排除 3 条' in page.locator('#progress').inner_text()
        checks.append('import restores labels; incompatible manifest rejected without overwriting')
        page.locator('#source').select_option('7');page.locator('#label').select_option('wave')
        assert page.locator('#count').inner_text()=='当前 33 条'
        page.locator('#source').select_option('1');page.locator('#label').select_option('drink')
        known=next(g for g in read(OUT/'review-items.json')if g['category']=='videomamba' and g['number']==1 and g['label']=='drink' and g['trigger_start']<=149.92<=g['trigger_end'])
        page.locator('#search').fill(known['id']);assert page.locator('#count').inner_text()=='当前 1 条'
        page.wait_for_function('document.querySelector("video").readyState>=2')
        page.screenshot(path=str(OUT/'ui-desktop.png'),full_page=True)
        checks.append('source/action/id filters and known true-drink hint')
        page.locator('#raw').click();page.wait_for_function('document.querySelector("video").readyState>=2 && document.querySelector("video").currentTime>140')
        assert page.locator('#video').evaluate('(v)=>v.error') is None
        checks.append('original source remux seeking and target overlay')
        page.locator('#search').fill('');page.locator('#source').select_option('all');page.locator('#label').select_option('all');page.locator('#scope').select_option('xclip')
        assert page.locator('#count').inner_text()=='当前 2011 条'
        assert page.locator('#raw').is_disabled()
        page.wait_for_function('document.querySelector("video").readyState>=2')
        assert page.locator('#video').evaluate('(v)=>v.error') is None
        checks.append('XCLIP appendix uses original-source playback and separate target trace')
        page.locator('#scope').select_option('videomamba');page.set_viewport_size({'width':390,'height':844})
        page.wait_for_load_state('networkidle')
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth+1')
        page.screenshot(path=str(OUT/'ui-mobile.png'),full_page=True)
        checks.append('mobile layout without horizontal overflow')
        context.close();browser.close()
    assert not errors,errors
    save('browser-verification.json',dict(passed=True,checks=checks,javascript_errors=errors,
        synthetic_labels='Temporary isolated browser context only; no user annotations or staging labels changed'))
    print(json.dumps(checks,ensure_ascii=False))


if __name__=='__main__':main()
