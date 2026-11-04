"""Editable native PPT and draw.io generated from the selected release configuration."""
from pathlib import Path
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
import zipfile

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
RELEASE = ROOT/'reports/person-release-20261024'
spec = importlib.util.spec_from_file_location('flow_scene', ROOT/'reports/algorithm-flow-20261023/scripts/build_flow.py')
flow = importlib.util.module_from_spec(spec); sys.modules[spec.name] = flow; spec.loader.exec_module(flow)
flow.OUT = OUT
flow.BASE = '逐人动作识别_整体算法与训练来源_20261024'


def main():
    lock = json.loads((RELEASE/'selection-lock.json').read_text())
    c = lock['config']; a, tau = c['alpha'], c['threshold']
    training = json.loads((RELEASE/'training-history.json').read_text())
    changes = {
        'subtitle': '现场微调模型 · 逐人双分支融合 · 并发片段保存',
        'version': '2026.10.24 · 正式逐人流程',
        'enhance': '保留原图亮度 L → 增强\nRetinexformer\n公开 NTIRE 预训练',
        'people': 'YOLO26m 人体框＋人物跟踪\n现场暗场＋清理展品亮场框微调',
        'rgb': '逐人 ROI → VideoMamba-Tiny\n现场逐人＋线上 other 微调\nRGB 概率 P_RGB',
        'gate_text': '原图亮度\nL < 30？',
        'skeleton': 'RTMO-s → 同人 ST-GCN\n现场暗场姿态／现场骨架动作',
        'fusion': f'同人加权融合\n{a:g} P_RGB + {1-a:g} P_pose',
        'decision': f'取最高类别\n动作 ≥ {tau:g} → 保留\n否则显示「其他」',
        'footnote': f'L：增强前原图亮度（0–255）；无合格骨架：{c["no_pose"]}。不使用物体先验；模型来源见下一页。',
    }
    for node in flow.NODES:
        if node.key in changes:
            node.text = changes[node.key]
        if node.key in ('enhance','people','rgb','skeleton'):
            node.size = 14.5
    flow.NOTES = f'''当前正式方案：原图亮度门30；RGB权重{a:g}、动作阈值{tau:g}，按冻结验证集选择后锁定。
先量原图整帧亮度再增强。每人15%外扩ROI，等比例填边400×224；1秒采样32帧，每0.5秒提交。
逐帧L>=30才提取姿态，32帧窗口平均L>=30才启用融合。回传增强切片原图L未知，跳过亮度门。
同一人物RGB与其骨架融合，禁止跨人取最大。无合格骨架策略为{c['no_pose']}。低于阈值只显示其他。
不做骨架平滑。YOLO仅定位人体，不以物体先验判断动作；人物ID是短期跟踪ID。
Retinexformer保留公开NTIRE预训练；此前现场增强微调损害动作效果，所以未采用，不能称全链模型都经现场训练。
实时≥3卡前两张交替增强/检测/姿态，其余GPU独立动作进程；离线第一卡前端，单卡共享。单卡拥挤实时性能需按实测说明。
每人+动作保存完整画面，前后各2秒，15秒重叠分段。保存链延迟1.2秒对齐异步事件并重绘标签。
输入断流、起始缓存不足、模型漏识、过载和提前停止可能造成上下文不足，元数据明确记录。
准确率和稳定性边界见person-release-20261024，不能把训练内回放当新场景泛化。
'''
    OUT.mkdir(exist_ok=True)
    ppt = flow.make_ppt(); drawio = flow.make_drawio()
    selected = '本轮逐人现场适配' if c['model']=='adapted' else '现场历史微调（新适配未获选）'
    records = [
        ('Retinexformer', '公开预训练；本轮未做现场微调', 'NTIRE 低照增强公开权重；保留已有版本，历史现场微调识别效果退化'),
        ('YOLO26m · 人体检测', '现场微调；已清理人形展品误标', '1626 暗场帧／1741 人体框＋72 亮场帧／89 真人框；原图和增强视图'),
        ('RTMO-s · 17点姿态', '现场暗场微调；推理不做平滑', '现场暗场人体框＋辅助 COCO17 姿态序列标注；训练前做质量筛选，非独立人工精标'),
        ('VideoMamba-Tiny · 逐人 RGB', '现场逐人 ROI 微调', '全部现场视频＋两批线上 other；535 去重来源，按来源组隔离；人物和时间段审核标签'),
        ('ST-GCN · 同人骨架动作', selected, ('同一冻结来源划分；最新检测／姿态输出＋逐人时间段标签；仅训练集更新，验证选轮' if c['model']=='adapted' else '历史现场 real-actiondata-all 骨架缓存微调；本轮逐人适配已对比，验证选择保留历史权重')),
    ]
    deck = flow.Presentation(ppt)
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    nodes = [flow.Node('title2',64,35,1472,65,'模型训练状态与数据来源','none','none',28,'text',bold=True,align='left')]
    for i,(model,status,data) in enumerate(records):
        y=145+i*124
        nodes += [flow.Node(f'm{i}',64,y,350,103,model,flow.BLUE_BG,flow.BLUE,18),
                  flow.Node(f's{i}',440,y,365,103,status,flow.TEAL_BG,flow.TEAL,17),
                  flow.Node(f'd{i}',831,y,705,103,data,flow.GRAY_BG,flow.LINE,16,align='left')]
    nodes.append(flow.Node('note2',64,790,1472,80,
        f'参数：L门30 · RGB权重{a:g} · 动作阈值{tau:g}；ST-GCN适配候选选第{training["selected_epoch"]}轮。\n训练／验证／测试沿用417／60／58个来源；历史初始化见过部分旧视频，结果不代表全新场景泛化。',
        'none','none',14,'text',color=flow.MUTED,align='left'))
    for n in nodes:
        flow.native_node(slide,n)
    slide.notes_slide.notes_text_frame.text=flow.NOTES
    deck.save(ppt)
    tree=ET.parse(drawio); root=tree.getroot()
    page=ET.SubElement(root,'diagram',id='training-provenance',name='模型训练与数据来源')
    graph=ET.SubElement(page,'mxGraphModel',page='1',pageWidth='1600',pageHeight='900')
    cells=ET.SubElement(graph,'root');ET.SubElement(cells,'mxCell',id='0');ET.SubElement(cells,'mxCell',id='1',parent='0')
    for n in nodes:
        cell=ET.SubElement(cells,'mxCell',id=n.key,value=n.text,vertex='1',parent='1',
            style=f'rounded=1;whiteSpace=wrap;html=0;fillColor={n.fill};strokeColor={n.stroke};fontSize={n.size/.72:g};fontFamily={flow.FONT};align={n.align};')
        ET.SubElement(cell,'mxGeometry',x=str(n.x),y=str(n.y),width=str(n.w),height=str(n.h),**{'as':'geometry'})
    ET.indent(root,space='  ')
    tree.write(drawio,encoding='utf-8',xml_declaration=True)
    (ROOT/'platform/模型运行流程_v2.drawio').write_bytes(drawio.read_bytes())
    with zipfile.ZipFile(ppt) as z:
        assert not any(n.startswith('ppt/media/') for n in z.namelist())
    assert len(flow.Presentation(ppt).slides)==2
    (OUT/'editable-qa.json').write_text(json.dumps(dict(slides=2,picture_objects=0,
        native_shapes=True,config=c,selection_lock_sha256=__import__('hashlib').sha256((RELEASE/'selection-lock.json').read_bytes()).hexdigest()),ensure_ascii=False,indent=2)+'\n')
    (OUT/'讲解备注.md').write_text(flow.NOTES)
    print(ppt,drawio)


if __name__=='__main__':
    main()
