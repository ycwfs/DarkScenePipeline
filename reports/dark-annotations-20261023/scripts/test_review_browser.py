"""Real Chromium checks: file:// assets, native-coordinate edits, export/import, unknowns."""
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

from scan import OUT, save
from check_feedback import inspect_feedback


def main() -> None:
    evidence=OUT/'evidence';evidence.mkdir(exist_ok=True)
    errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        context=browser.new_context(viewport={'width':1560,'height':1100},accept_downloads=True)
        page=context.new_page();page.on('pageerror',lambda err:errors.append(str(err)))
        page.goto((OUT/'index.html').as_uri())
        page.wait_for_function('data && currentImage && currentImage.complete && canvas.width > 0')
        # Pick a video containing a person; initial partial indexes may start on an empty scene.
        vid=page.evaluate('CATALOG.videos.find(v=>v.boxes>0).id')
        page.evaluate('(id)=>loadVideo(id)',vid)
        page.wait_for_function('(id)=>data && data.video.id===id && currentImage && currentImage.complete',arg=vid)
        page.evaluate('fi=data.frames.findIndex(f=>f.people.length>0); showFrame()')
        page.wait_for_timeout(200)
        before=page.evaluate('clone(person().bbox_xyxy)')
        # Move the upper-left corner 8 native pixels in x and y, using actual pointer events.
        canvas=page.locator('#canvas').bounding_box();size=page.evaluate('[data.video.width,data.video.height]')
        x=canvas['x']+before[0]/size[0]*canvas['width'];y=canvas['y']+before[1]/size[1]*canvas['height']
        page.mouse.move(x,y);page.mouse.down();page.mouse.move(x+8/size[0]*canvas['width'],y+8/size[1]*canvas['height'],steps=3);page.mouse.up()
        after=page.evaluate('clone(person().bbox_xyxy)')
        assert abs(after[0]-before[0]-8)<1 and abs(after[1]-before[1]-8)<1,(before,after)
        page.select_option('#mode','pose');page.select_option('#joint','9')
        # Add a selected joint at a known location far from existing joints.
        page.evaluate('editable().people[pi].keypoint_usable.fill(false); editable().people[pi].keypoint_visibility.fill(0);draw()')
        target=[(after[0]+after[2])/2,(after[1]+after[3])/2]
        page.mouse.click(canvas['x']+target[0]/size[0]*canvas['width'],canvas['y']+target[1]/size[1]*canvas['height'])
        point=page.evaluate('person().keypoints[9]')
        assert abs(point[0]-target[0])<1 and abs(point[1]-target[1])<1
        page.select_option('#visibility','1')
        page.fill('#note','浏览器自动测试：此记录不是真实人工审核，不用于训练。')
        page.click('#approve-pose')
        expected=page.evaluate('clone(edits)')
        with page.expect_download() as download:page.click('#export')
        target_path=evidence/'synthetic-browser-feedback.json';download.value.save_as(target_path)
        payload=json.loads(target_path.read_text());audit=inspect_feedback(payload)
        assert audit['valid'] and audit['reviewed_frames']==1
        page.reload();page.wait_for_function('data && currentImage && currentImage.complete')
        assert page.evaluate('clone(edits)')==expected
        page.evaluate('edits={};localStorage.removeItem(STORE)')
        page.set_input_files('#import',str(target_path))
        page.wait_for_function('Object.keys(edits).length===1')
        assert page.evaluate('clone(edits)')==expected
        # Restore clean page before evidence screenshot; synthetic feedback never enters the dataset.
        page.evaluate('edits={};localStorage.removeItem(STORE)')
        page.reload();page.wait_for_function('data && currentImage && currentImage.complete')
        for value in ['raw','enhanced','gamma']:
            page.select_option('#view',value);page.wait_for_timeout(200)
            assert page.evaluate('currentImage.complete && currentImage.naturalWidth>0')
        page.screenshot(path=str(evidence/'review-desktop.png'),full_page=True)
        page.set_viewport_size({'width':650,'height':1000});page.wait_for_timeout(150)
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1')
        page.screenshot(path=str(evidence/'review-mobile.png'),full_page=True)
        assert not errors,errors
        save(evidence/'browser-check.json',dict(passed=True,browser=browser.version,
            checks=['file_url_assets','native_bbox_drag_roundtrip','native_joint_drag_roundtrip',
                    'visibility_edit','feedback_export_validation','reload_persistence','feedback_import_roundtrip',
                    'raw_gamma_enhanced_views','no_mobile_horizontal_overflow','no_page_errors'],
            synthetic_feedback=True,real_user_reviews_added=0))
        browser.close()
    print('Browser review checks passed; no real user annotations were added.')


if __name__=='__main__':main()
