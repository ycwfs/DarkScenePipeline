"""Persist assistant visual observations of the generated 40-case audit gallery.

These are manual visual findings, NOT automatically generated object ground truth.
No training labels or model outputs are rewritten.
"""
import json
from pathlib import Path

OUT=Path(__file__).resolve().parents[1]
NOTES={
 1:('container_plausible','原图ROI框位于人物嘴边的蓝色瓶子；增强ROI仍未过阈值。'),
 2:('container_plausible','两种人物ROI都框中嘴边蓝色瓶子。'),
 3:('unrelated_background','新增bottle框位于人物头上方的背景展品区域，不是其手中容器；无可用手腕先验。'),
 4:('unrelated_background','cup候选在人物头上方背景；另有人物旁handbag候选，不构成喝水物体证据。'),
 5:('container_plausible','原图ROI小框落在嘴边所持透明容器，bottle/cup细分类不可靠。'),
 6:('crop_regression','整帧框中蓝色瓶子0.265；增强人物ROI降至0.236并把手臂附近误报为skateboard，原图ROI漏检。'),
 7:('container_plausible','整帧和增强ROI都框中蓝色瓶子；原图ROI未检出。'),
 8:('container_plausible','喝水35胸前容器：整帧0.067，增强ROI cup0.738，原图ROI cup0.451。'),
 9:('container_plausible','喝水35嘴边容器被两种ROI检出；椅子也进入脚边先验，表明物体存在不代表捡物。'),
 10:('localized_wrong_class','原图ROI框落在被拾起的纸片/薄片上，但类别误报skateboard0.426；只能算弱定位线索。'),
 11:('person_misclassified','原图ROI把弯腰者身体误报dog/cow，框并非所捡纸片。'),
 12:('person_misclassified','原图ROI dog0.436包住弯腰者身体，纸片并未被正确检出。'),
 13:('body_part_misclassified','增强ROI skateboard0.584落在手臂/裤腿附近，未框中地上的纸片。'),
 14:('person_misclassified','增强ROI dog0.852、原图ROI dog0.409包住弯腰的人，属于明确假物体。'),
 15:('person_misclassified','增强ROI dog0.898、原图ROI dog0.915包住弯腰的人，分数高但类别错误。'),
 16:('body_part_misclassified','增强ROI tennis racket0.316对应人物裤腿，不是其手边纸片。'),
 17:('container_plausible','旧目录喝水1，瓶子位于嘴边，两种ROI检出；增强ROI还把头部小区域误报tennis racket。'),
 18:('unrelated_background','旧目录喝水12，增强ROI bottle框在头上方背景物品，缺少本人手边关联。'),
 19:('container_plausible','旧目录喝水3，原图ROI bottle框位于嘴边蓝色容器。'),
 20:('other_ambiguous_container','other0054人物手部附近cup0.260；小物体外观不足以确认杯子，单帧也不能证明喝水。'),
 21:('other_occlusion_ambiguity','other0098的bottle0.272落在前景遮挡人物/衣物边缘；物体类别和归属无法可靠确认。'),
 22:('other_occlusion_ambiguity','同一other0098后续帧bottle0.275仍在前景衣物边缘，重复检测并不自动证明正确。'),
 23:('other_ambiguous_container','other0137的cup0.329在手持物附近；画面模糊，类别不确定，既有动作真值仍为other。'),
 24:('other_ambiguous_container','新回传0001中，cup候选位于人物腰前手持物附近，不能从外观确认；动作标签为other。'),
 25:('other_ambiguous_container','同一新回传0001的4秒处仍出现cup候选；静态手边物体无法证明喝水。'),
 26:('unrelated_object','整帧在人物腿侧边缘给tennis racket，裁剪后消失；非人框不是捡物证据。'),
 27:('foot_misclassified','原图ROI把脚部/鞋附近误报skis0.416。'),
 28:('robot_and_external_object','人体框实际是展品机器人；整帧脚边支持来自ROI之外的邻近框，裁剪后被移除。'),
 29:('robot_and_external_object','同一机器人候选，整帧脚边支持在可见人物ROI之外；裁剪减少干扰但未解决人体误检。'),
 30:('robot_and_external_object','机器人轨迹，整帧非人先验来自局部画面之外；裁剪后没有支持。'),
 31:('foot_misclassified','两种ROI在行走者鞋边给skateboard候选，没有可见滑板或拾取动作。'),
 32:('robot_and_external_object','机器人轨迹；整帧高脚边分数并不对应真实人物动作。'),
 33:('foot_misclassified','原图ROI在脚边把鞋/地面局部误报skateboard。'),
 34:('unrelated_real_object','画面确有随身包，但它与走路者同行，不是正在捡起的地面物；多人重叠也使归属不明确。'),
 35:('foot_misclassified','原图ROI在鞋边给skateboard0.336，人物在站立/行走。'),
 36:('robot_and_external_object','机器人候选；整帧脚边支持来自ROI之外，裁剪移除了背景证据。'),
 37:('foot_misclassified','增强ROI把鞋部附近误报skateboard0.339。'),
 38:('missed_known_container','喝水16嘴边容器在三种输入下都未过阈值；整帧还输出无关train框。'),
 39:('missed_known_container','喝水34嘴边容器仍未检出；ROI检出/误分类前景椅子，不能修复RGB把喝水判挥手的问题。'),
 40:('missed_paper','捡东西2手边纸片在三种输入下都未检出；COCO80没有通用纸张类别。'),
}


def main() -> None:
    rows=json.loads((OUT/'visual-index.json').read_text())
    assert set(NOTES)=={r['number']for r in rows}
    result=[]
    for row in rows:
        category,note=NOTES[row['number']]
        result.append(dict(**row,reviewer='assistant visual inspection of rendered three-way image and contact sheets',
                           observation_category=category,note=note,object_box_ground_truth=False))
    (OUT/'visual-review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# 逐图核对记录','',
      '已查看40组对照图。以下为助手视觉核对，未绘制全量物体真值框，不计算检测mAP。',
      '候选出现的位置、语义与动作支持分开记录；图号对应index.html。','',
      '| 图号 | 视频/帧/人物 | 观察 |','|---|---|---|']
    for row in result:
        lines.append(f'| [{row["number"]:03d}]({row["image"]}) | {row["source"]} / {row["frame"]} / T{row["track_id"]:02d} | {row["note"]} |')
    (OUT/'visual-review.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
