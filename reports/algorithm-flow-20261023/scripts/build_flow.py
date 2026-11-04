"""Build the editable current algorithm diagram from one shared scene description."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
import zipfile

OUT = Path(__file__).resolve().parents[1]
REPO = OUT.parents[1]
# Reuse the locally installed meeting export tools; no operator dependencies change.
sys.path.insert(0, str(REPO / "reports/meeting-20261015/.tools"))
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

BASE = "逐人动作识别_整体算法流程_20261023"
FONT = "Noto Sans CJK SC"
INK, MUTED, LINE = "#172B43", "#58697D", "#D9E1EB"
BLUE, BLUE_BG = "#2463B6", "#EBF2FB"
TEAL, TEAL_BG = "#00867B", "#E9F5F2"
AMBER, AMBER_BG = "#AB701C", "#FBF3E3"
GRAY_BG = "#F4F6F9"


@dataclass
class Node:
    key: str
    x: float
    y: float
    w: float
    h: float
    text: str
    fill: str = BLUE_BG
    stroke: str = BLUE
    size: float = 18
    kind: str = "round"
    color: str = INK
    bold: bool = False
    align: str = "center"


@dataclass
class Edge:
    source: str
    target: str
    points: list[tuple[float, float]]
    color: str = MUTED


NODES = [
    Node("scope", 64, 335, 1472, 310, "", "#FAFBFD", LINE),
    Node("title", 64, 37, 1472, 65, "逐人动作识别 · 整体算法流程", "none", "none", 29, "text", bold=True, align="left"),
    Node("subtitle", 64, 101, 1110, 42, "照度门 → 同人双分支融合 → 逐人显示与并发保存", "none", "none", 16, "text", color=MUTED, align="left"),
    Node("version", 1190, 101, 346, 42, "2026.10.23 · 独立候选方案", "none", "none", 13, "text", color=TEAL, align="right"),
    Node("input", 64, 175, 220, 90, "视频输入\n文件 / 实时流", GRAY_BG, MUTED),
    Node("enhance", 340, 175, 350, 90, "保留原图亮度 L\nRetinexformer 画面增强", GRAY_BG, MUTED),
    Node("people", 750, 175, 350, 90, "YOLO 人体检测＋跟踪\n分配人物 ID", GRAY_BG, MUTED),
    Node("window", 1160, 175, 376, 90, "建立逐人时序窗口\n最近 1 秒 / 32 帧", GRAY_BG, MUTED),
    Node("scope_label", 150, 346, 625, 38, "逐人执行 · 两条分支对应同一人物 ID", "none", "none", 16, "text", color=BLUE, bold=True, align="left"),
    Node("rgb", 150, 430, 300, 120, "逐人 ROI 裁剪\nVideoMamba-Tiny\nRGB 概率 P_RGB"),
    Node("gate", 535, 425, 210, 130, "", AMBER_BG, AMBER, 19, "diamond"),
    Node("gate_text", 540, 450, 200, 80, "原图亮度\nL < 75？", "none", "none", 18, "text"),
    Node("rgb_only", 835, 365, 290, 85, "仅用本人的 RGB\nP = P_RGB"),
    Node("skeleton", 835, 520, 290, 90, "同人骨架动作识别\nRTMO-s → ST-GCN", TEAL_BG, TEAL, 17),
    Node("fusion", 1205, 520, 285, 90, "同人加权融合\n0.7 P_RGB + 0.3 P_pose", TEAL_BG, TEAL, 16),
    Node("decision", 1205, 360, 285, 100, "取最高类别\n动作 ≥ 0.75 → 保留\n否则显示「其他」", GRAY_BG, MUTED, 16),
    Node("yes", 665, 385, 150, 32, "是 · 暗场", "#FAFBFD", "none", 14, "text", color=BLUE),
    Node("no", 665, 542, 150, 32, "否 · 亮场", "#FAFBFD", "none", 14, "text", color=TEAL),
    Node("display", 90, 725, 650, 95, "逐人可视化 → 实时输出\n人物框＋框下动作；骨架分支激活时绘制骨架", BLUE_BG, BLUE, 17),
    Node("record", 860, 725, 650, 95, "按人物 ID＋动作保存完整画面\n前后各 2 秒 · 15 秒分段 · 多人片段允许重叠", TEAL_BG, TEAL, 17),
    Node("record_label", 1050, 682, 270, 32, "仅非「其他」动作触发", "#FFFFFF", "none", 13, "text", color=TEAL),
    Node("footnote", 64, 842, 1472, 39, "L：增强前原图窗口平均亮度（0–255）；骨架缺失时使用 soft「其他」票。YOLO 仅用于人体定位。", "none", "none", 12, "text", color=MUTED, align="left"),
]

EDGES = [
    Edge("input", "enhance", [(284, 220), (340, 220)]),
    Edge("enhance", "people", [(690, 220), (750, 220)]),
    Edge("people", "window", [(1100, 220), (1160, 220)]),
    Edge("window", "rgb", [(1348, 265), (1348, 300), (100, 300), (100, 490), (150, 490)]),
    Edge("rgb", "gate", [(450, 490), (535, 490)]),
    Edge("gate", "rgb_only", [(640, 425), (640, 407.5), (835, 407.5)], BLUE),
    Edge("gate", "skeleton", [(640, 555), (640, 565), (835, 565)], TEAL),
    Edge("skeleton", "fusion", [(1125, 565), (1205, 565)], TEAL),
    Edge("rgb_only", "decision", [(1125, 407.5), (1205, 407.5)], BLUE),
    Edge("fusion", "decision", [(1347.5, 520), (1347.5, 460)], TEAL),
    Edge("decision", "display", [(1490, 407.5), (1560, 407.5), (1560, 675), (415, 675), (415, 725)]),
    Edge("decision", "record", [(1490, 407.5), (1560, 407.5), (1560, 675), (1185, 675), (1185, 725)]),
]

NOTES = """本页对应 2026-10-23 独立逐人候选，尚未替换平台算子包。
图按算法依赖组织；实际执行会交错人体检测、姿态关联、缓存和异步推理，并非逐框串行。
输入先缩至最长边720，保留增强前原图亮度，然后用Retinexformer增强。
YOLO26m仅检测人体，短期跟踪分配ID。RGB输入是每人的15%外扩ROI，等比例填边为400×224。
每人最近约1秒采样32帧；VideoMamba-Tiny使用逐人ROI微调权重，骨架使用同人RTMO-s与ST-GCN。
亮度门使用原图全画面的窗口平均像素亮度，不是人物ROI亮度，也不是lux。
L<75时仅用本人的RGB；否则同人完整概率向量按0.7/0.3融合，禁止跨人物取最高骨架。
亮场无合格骨架时，骨架向量为other独热票；不是回退为纯RGB。已增强回传片段缺少原图时跳过亮度门。
取最高类别后，非other动作达到0.75才显示动作及概率；否则只显示其他。暖机/过期结果显示识别中。
骨架只在分支有效且有当前姿态时绘制；画面没有全局动作底栏，不使用物体先验。
单卡共用设备，多卡可将增强/检测/姿态与逐人动作推理解耦；逻辑与阈值相同。
直播与录制独立，保存侧等待1.2秒对齐异步结果，不阻塞直播。其他不触发保存。
人物ID＋动作分别保存完整画面。前后各2秒，超过15秒以重叠片段续存；同时不同动作各自保留尾部。
跟踪ID不是持久身份。启动缓存不足、断流、漏识或提前停止可能缺少动作上下文，元数据会标记。
本页只展示关键算法，不展开重连、队列、硬件分配及平台部署状态；细节见相邻README与实时核验报告。
"""


def unit(value: float) -> int:
    return Inches(value / 100)


def color(value: str) -> RGBColor:
    return RGBColor.from_string(value.lstrip("#"))


def native_node(slide, node: Node):
    if node.kind == "text":
        shape = slide.shapes.add_textbox(unit(node.x), unit(node.y), unit(node.w), unit(node.h))
    else:
        kind = MSO_AUTO_SHAPE_TYPE.DIAMOND if node.kind == "diamond" else MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE
        shape = slide.shapes.add_shape(kind, unit(node.x), unit(node.y), unit(node.w), unit(node.h))
        if node.kind != "diamond":
            shape.adjustments[0] = .1
    shape.name = node.key + " · " + node.text.replace("\n", " / ")
    if node.fill == "none":
        shape.fill.background()
    else:
        shape.fill.solid()
        shape.fill.fore_color.rgb = color(node.fill)
    if node.stroke == "none":
        shape.line.fill.background()
    else:
        shape.line.color.rgb = color(node.stroke)
        shape.line.width = Pt(1.35)
    shape._element.spPr.append(OxmlElement("a:effectLst"))
    # LibreOffice also reads the inherited Office effect style, even with effectLst.
    for effect in shape._element.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}effectRef"):
        effect.set("idx", "0")
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = unit(5)
    tf.margin_top = tf.margin_bottom = unit(2)
    for i, line in enumerate(node.text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}[node.align]
        p.space_before = p.space_after = Pt(0)
        p.line_spacing = 1.1
        run = p.add_run()
        run.text = line
        run.font.name = FONT
        run.font.size = Pt(node.size)
        run.font.bold = node.bold
        run.font.color.rgb = color(node.color)
        ea = OxmlElement("a:ea")
        ea.set("typeface", FONT)
        run.font._rPr.append(ea)
    return shape


def attach(connector, shape, point: tuple[float, float], start: bool) -> None:
    px, py = unit(point[0]), unit(point[1])
    distances = [abs(py-shape.top), abs(px-shape.left),
                 abs(py-shape.top-shape.height), abs(px-shape.left-shape.width)]
    tag = OxmlElement("a:stCxn" if start else "a:endCxn")
    tag.set("id", str(shape.shape_id))
    tag.set("idx", str(distances.index(min(distances))))
    connector._element.nvCxnSpPr.cNvCxnSpPr.append(tag)


def make_ppt() -> Path:
    deck = Presentation()
    deck.slide_width, deck.slide_height = unit(1600), unit(900)
    deck.core_properties.title = "逐人动作识别：整体算法流程（2026-10-23）"
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color("#FFFFFF")
    shapes = {n.key: native_node(slide, n) for n in NODES}
    # Edges are native connectors; insert them above the background and below labels.
    for i, edge in enumerate(EDGES):
        for j, (a, b) in enumerate(zip(edge.points, edge.points[1:])):
            conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, unit(a[0]), unit(a[1]), unit(b[0]), unit(b[1]))
            conn.name = f"连线 {i + 1}.{j + 1} · {edge.source} → {edge.target}"
            conn.line.color.rgb = color(edge.color)
            conn.line.width = Pt(1.5)
            conn._element.spPr.append(OxmlElement("a:effectLst"))
            for effect in conn._element.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}effectRef"):
                effect.set("idx", "0")
            if j == 0:
                attach(conn, shapes[edge.source], a, True)
            if j == len(edge.points)-2:
                attach(conn, shapes[edge.target], b, False)
                arrow = OxmlElement("a:tailEnd")
                arrow.set("type", "triangle")
                conn.line._get_or_add_ln().append(arrow)
            tree = slide.shapes._spTree
            tree.remove(conn._element)
            tree.insert(3, conn._element)
    slide.notes_slide.notes_text_frame.text = NOTES
    path = OUT / f"{BASE}_可编辑.pptx"
    deck.save(path)
    return path


def make_drawio() -> Path:
    mx = ET.Element("mxfile", host="app.diagrams.net", version="24.7.17")
    page = ET.SubElement(mx, "diagram", id="person-action-20261023", name="逐人动作识别 · 当前候选")
    model = ET.SubElement(page, "mxGraphModel", grid="0", page="1", pageScale="1", pageWidth="1600", pageHeight="900", math="0", shadow="0")
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")
    for n in NODES:
        shape = "text" if n.kind == "text" else "shape=rhombus" if n.kind == "diamond" else "rounded=1;arcSize=10"
        style = (f"{shape};html=0;whiteSpace=wrap;fillColor={n.fill};strokeColor={n.stroke};"
                 f"strokeWidth=1.875;fontFamily={FONT};fontSize={n.size/0.72:.3f};fontColor={n.color};"
                 f"fontStyle={int(n.bold)};align={n.align};verticalAlign=middle;spacing=5;")
        cell = ET.SubElement(root, "mxCell", id=n.key, value=n.text, style=style, vertex="1", parent="1")
        ET.SubElement(cell, "mxGeometry", x=str(n.x), y=str(n.y), width=str(n.w), height=str(n.h), **{"as": "geometry"})
    by_key = {n.key: n for n in NODES}
    for i, e in enumerate(EDGES):
        s, t = by_key[e.source], by_key[e.target]
        a, b = e.points[0], e.points[-1]
        style = (f"edgeStyle=none;html=0;rounded=0;endArrow=block;endFill=1;strokeWidth=2;strokeColor={e.color};"
                 f"exitX={(a[0]-s.x)/s.w};exitY={(a[1]-s.y)/s.h};exitPerimeter=0;"
                 f"entryX={(b[0]-t.x)/t.w};entryY={(b[1]-t.y)/t.h};entryPerimeter=0;")
        cell = ET.Element("mxCell", id=f"edge{i}", edge="1", parent="1", source=e.source, target=e.target, style=style)
        geo = ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
        points = ET.SubElement(geo, "Array", **{"as": "points"})
        for x, y in e.points[1:-1]:
            ET.SubElement(points, "mxPoint", x=str(x), y=str(y))
        root.insert(3, cell)
    ET.indent(mx, space="  ")
    path = OUT / f"{BASE}.drawio"
    data = ET.tostring(mx, encoding="utf-8", xml_declaration=True)
    path.write_bytes(data)
    (REPO / "platform/模型运行流程_v2.drawio").write_bytes(data)
    return path


def verify(ppt: Path, drawio: Path) -> None:
    deck = Presentation(ppt)
    slide = deck.slides[0]
    with zipfile.ZipFile(ppt) as archive:
        assert not any(name.startswith("ppt/media/") for name in archive.namelist())
    assert len(deck.slides) == 1
    texts = "\n".join(s.text for s in slide.shapes if s.has_text_frame)
    assert all(v in texts for v in ("0.75", "75", "0.7", "0.3", "逐人 ROI", "VideoMamba", "ST-GCN", "2 秒", "15 秒"))
    root = ET.parse(drawio).getroot()
    assert len(root.findall("diagram")) == 1
    ids = {cell.get("id") for cell in root.iter("mxCell")}
    for cell in root.iter("mxCell"):
        if cell.get("edge"):
            assert cell.get("source") in ids and cell.get("target") in ids
    # Exercise native text editing and save/reopen without changing the delivered slide.
    edit = next(s for s in slide.shapes if s.name.startswith("title ·"))
    edit.text = "可编辑性核验"
    check = OUT / "render-check/edit-check.pptx"
    check.parent.mkdir(exist_ok=True)
    deck.save(check)
    assert "可编辑性核验" in [s.text for s in Presentation(check).slides[0].shapes if s.has_text_frame]
    check.unlink()
    evidence = dict(slides=1, picture_objects=0,
                    text_objects=sum(bool(n.text) for n in NODES),
                    connector_segments=sum(len(e.points)-1 for e in EDGES),
                    drawio_edges=len(EDGES), native_edit_roundtrip=True,
                    files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ppt, drawio)})
    (OUT / "evidence/editable-qa.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    (OUT / "evidence").mkdir(parents=True, exist_ok=True)
    (OUT / "render-check").mkdir(exist_ok=True)
    (OUT / "讲解备注.md").write_text("# 流程图讲解备注\n\n"+NOTES)
    verify(make_ppt(), make_drawio())
