"""Render visually reviewed draft boxes; never infer labels or export negatives."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

# Coordinates manually placed by the assistant after inspecting original, display-
# adjusted, Retinexformer and neighboring frames. They are not detector predictions.
# Reference images were 960x540; output coordinates map back to native 1920x1080.
DRAFTS = [
    ("捡东西1", 13, [320, 85, 435, 450], False, "人物可定位；深色衣服左边缘与背景相近，底部阴影不计入人体。"),
    ("捡东西1", 32, [340, 187, 535, 413], False, "人物弯腰；头、躯干及腿可定位，手部边缘模糊。"),
    ("捡东西1", 52, [345, 172, 537, 404], False, "人物弯腰；框包含可见身体，地面阴影不计入人体。"),
    ("挥手18", 12, [251, 25, 522, 528], False, "框包含抬起及外伸的手臂；黑色上衣边缘仍需复核。"),
    ("挥手18", 30, [290, 3, 559, 540], True, "框包含抬起手臂；脚部接近/触及图像底边，不外推不可见部分。"),
    ("挥手18", 48, [375, 26, 595, 540], True, "框包含抬起手臂及身体右侧可见手臂；脚部在底边截断，不外推完整身体。"),
    ("喝水8", 12, None, None, "存在疑似移动人影，但躯干、手臂及上下边界不能可靠区分。"),
    ("喝水8", 31, None, None, "相邻帧可辅助判断移动区域，仍不足以确定准确人体框。"),
    ("喝水8", 49, None, None, "全身边界不可靠；不根据视频文件名或动作标签猜框。"),
]


def main() -> None:
    (OUT / "visual").mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(FONT, 22)
    small = ImageFont.truetype(FONT, 18)
    sources, rows, overview = {}, [], []
    for name in ("捡东西1", "挥手18", "喝水8"):
        source = ROOT / "real-actiondata-all" / f"{name}.mp4"
        cap = cv2.VideoCapture(str(source))
        assert cap.isOpened(), source
        fps = cap.get(cv2.CAP_PROP_FPS)
        sheet = Image.new("RGB", (1920, 3*625+65), "#172333")
        draw = ImageDraw.Draw(sheet)
        draw.text((15, 12), f"{name} | AI视觉标注草稿，未经独立复核；不是检测器输出", font=font, fill="white")
        sources[name] = dict(path=str(source.relative_to(ROOT)),
                             sha256=hashlib.sha256(source.read_bytes()).hexdigest(), fps=fps)
        for i, (_, fi, box, truncated, note) in enumerate(r for r in DRAFTS if r[0] == name):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            assert ok
            assert frame.shape[:2] == (1080, 1920)
            row = dict(video=name, frame_index_zero_based=fi, timestamp_seconds=fi/fps,
                       image_size=[1920, 1080], review_status="pending_independent_review",
                       include_in_training=False, exhaustive_person_annotation=False,
                       note=note, classification="person", action_label=None,
                       adjacent_frames_reviewed=[fi-1, fi, fi+1])
            if box is None:
                row.update(status="uncertain_do_not_use_as_negative", bbox_xyxy=None,
                           bbox_xywh=None, yolo_xywh_normalized=None, truncated=None)
            else:
                x1, y1, x2, y2 = [v*2 for v in box]
                assert 0 <= x1 < x2 <= 1920 and 0 <= y1 < y2 <= 1080
                row.update(status="draft_box", bbox_xyxy=[x1,y1,x2,y2],
                           bbox_xywh=[x1,y1,x2-x1,y2-y1], truncated=truncated,
                           yolo_xywh_normalized=[(x1+x2)/3840, (y1+y2)/2160,
                                                 (x2-x1)/1920, (y2-y1)/1080])
            rows.append(row)
            original = cv2.resize(frame, (960,540), interpolation=cv2.INTER_AREA)
            gamma = .35 if name == "喝水8" else .65
            lut = np.rint(255*(np.arange(256)/255)**gamma).astype(np.uint8)
            for j, image in enumerate((original, cv2.LUT(original, lut))):
                tile = Image.fromarray(image[:,:,::-1])
                td = ImageDraw.Draw(tile)
                if box is not None:
                    td.rectangle(box, outline="#48E2B0", width=3)
                    label_y = max(2, box[1]-28)
                    td.rectangle((box[0],label_y,min(959,box[0]+202),label_y+27), fill="#172333")
                    td.text((box[0]+4,label_y+1), "person · 待复核", font=small, fill="#48E2B0")
                else:
                    td.rectangle((12,12,600,51), fill="#172333")
                    td.text((20,16), "边界不可可靠确认：不画确定框，不作为无人负样本", font=small, fill="#FFD17A")
                y = 65+i*625
                variant = "原图" if j == 0 else f"仅调整显示 Gamma={gamma}"
                draw.text((j*960+12,y+5), f"{name} · 帧 {fi} / {fi/fps:.2f}s · {variant}", font=font, fill="white")
                sheet.paste(tile, (j*960,y+43))
            draw.text((15,65+i*625+588), note, font=small, fill="#C7D4E0")
            if i == 1:
                overview.append(sheet.crop((960,65+i*625,1920,65+(i+1)*625)))
        cap.release()
        sheet.save(OUT / "visual" / f"{name}_标注核对.jpg", quality=94)
    overview_sheet = Image.new("RGB", (1920, 2*625), "#172333")
    for i, tile in enumerate(overview):
        overview_sheet.paste(tile, ((i%2)*960,(i//2)*625))
    text = ImageDraw.Draw(overview_sheet)
    text.text((990,730), "6 帧：可定位人体，已画候选框\n3 帧：边界不可靠，不猜框\n\n全部是草稿，未加入训练。", font=font, fill="white", spacing=16)
    overview_sheet.save(OUT / "visual/标注示例总览.jpg", quality=94)
    result = dict(schema="assistant_visual_box_drafts_v1", coordinate_system="native pixels; xyxy; right/bottom exclusive",
                  reference_size=[960,540], annotator="Codex visual review",
                  independent_human_review=False, interpolation_used=False,
                  source_images_used=["original frames", "gamma display adjustment", "existing Retinexformer previews", "adjacent frames"],
                  sources=sources, frames=rows, summary=dict(reviewed_keyframes=9, frames_viewed_including_neighbors=27,
                  draft_boxes=6, uncertain_frames=3, frames_added_to_training=0))
    (OUT / "annotations-draft.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result["summary"],ensure_ascii=False))


if __name__ == "__main__":
    main()
