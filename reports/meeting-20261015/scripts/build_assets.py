"""Build meeting diagrams and evidence figures from preserved local measurements."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parents[1]
REPO = OUT.parents[1]
sys.path.insert(0, str(OUT / ".tools"))
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
font_manager.fontManager.addfont(FONT)
font_manager.fontManager.addfont(BOLD)
plt.rcParams.update({
    "font.family": font_manager.FontProperties(fname=FONT).get_name(),
    "font.size": 15, "axes.unicode_minus": False, "svg.fonttype": "none",
    "pdf.fonttype": 42, "savefig.facecolor": "white",
})
INK, MUTED, LINE = "#172B43", "#58697D", "#D9E1EB"
BLUE, BLUE_BG = "#2463B6", "#EBF2FB"
TEAL, TEAL_BG = "#00867B", "#E9F5F2"
RED, RED_BG = "#C54F38", "#FCF0EB"
AMBER, AMBER_BG = "#AB701C", "#FBF3E3"
GRAY_BG = "#F4F6F9"
LABELS = ["wave", "throw", "chase", "fall", "fight", "talk", "drink", "pick_up", "shake_hands", "other"]
ZH = dict(zip(LABELS, ["挥手", "投掷", "追逐", "跌倒", "打架", "交谈", "喝水", "捡东西", "握手", "其他"]))
ROWS = [json.loads(s) for s in (OUT / "sources/historical-pn-840.jsonl").read_text().splitlines()]
FIGURES: list[tuple[str, str, str]] = []


def prediction(vector):
    i = int(np.argmax(vector))
    return LABELS[i] if vector[i] >= 0.65 else "other"


def outcome(row, key):
    return prediction(row["rgb"] if row["lum"] < 75 else row[key])


def canvas(number, title, subtitle, badge=""):
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 1600), ylim=(900, 0))
    ax.axis("off")
    ax.add_patch(Rectangle((64, 53), 8, 54, color=TEAL, linewidth=0))
    ax.text(92, 73, title, fontsize=27, weight="bold", color=INK, va="center")
    ax.text(92, 122, subtitle, fontsize=14, color=MUTED, va="center")
    ax.text(1535, 67, f"{number:02d}", fontsize=20, color=TEAL, ha="right", va="center")
    if badge:
        ax.text(1535, 122, badge, fontsize=12, color=MUTED, ha="right", va="center")
    return fig, ax


def text(ax, x, y, value, size=16, color=INK, weight="normal", ha="left"):
    return ax.text(x, y, value, fontsize=size, color=color, weight=weight,
                   ha=ha, va="center", linespacing=1.55)


def panel(ax, x, y, w, h, fill=GRAY_BG, edge="none", radius=14):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                              facecolor=fill, edgecolor=edge, linewidth=1.4))


def footer(ax, value):
    ax.plot([64, 1536], [835, 835], color=LINE, lw=1)
    text(ax, 64, 862, value, size=10.5, color=MUTED)


def axes_at(fig, x, y, w, h):
    return fig.add_axes([x / 1600, 1 - (y + h) / 900, w / 1600, h / 900])


def clean_plot(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=12)
    ax.set_axisbelow(True)


def save(fig, name, title, notes):
    fig.savefig(OUT / "png" / f"{name}.png", dpi=180)
    fig.savefig(OUT / "svg" / f"{name}.svg")
    fig.savefig(OUT / "pdf" / f"{name}.pdf")
    FIGURES.append((name, title, notes))
    plt.close(fig)


class Diagram:
    """One node/edge representation drives both the rendered figure and draw.io XML."""
    def __init__(self, ax, name):
        self.ax, self.name = ax, name
        self.nodes, self.edges = {}, []

    def node(self, key, x, y, w, h, label, fill=BLUE_BG, stroke=BLUE, diamond=False, size=16):
        self.nodes[key] = dict(x=x, y=y, w=w, h=h, label=label, fill=fill,
                               stroke=stroke, diamond=diamond, size=size)
        if diamond:
            self.ax.add_patch(Polygon([(x+w/2,y), (x+w,y+h/2), (x+w/2,y+h), (x,y+h/2)],
                                     facecolor=fill, edgecolor=stroke, lw=1.6))
        else:
            panel(self.ax, x, y, w, h, fill, stroke)
        text(self.ax, x+w/2, y+h/2, label, size=size, ha="center")

    def edge(self, source, target, points, label="", label_xy=None, color=MUTED):
        self.edges.append(dict(source=source, target=target, points=points, label=label,
                               label_xy=label_xy, color=color))
        xs, ys = zip(*points)
        self.ax.plot(xs[:-1], ys[:-1], lw=1.6, color=color, solid_capstyle="round")
        self.ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                         mutation_scale=15, linewidth=1.6, color=color))
        if label:
            self.ax.text(*label_xy, label, fontsize=12, color=color, ha="center", va="center",
                         bbox=dict(facecolor="white", edgecolor="none", pad=2))

    def xml(self, mxfile, title, subtitle, note):
        page = ET.SubElement(mxfile, "diagram", id=self.name, name=title)
        model = ET.SubElement(page, "mxGraphModel", grid="0", page="1", pageScale="1",
                             pageWidth="1600", pageHeight="900", math="0", shadow="0")
        root = ET.SubElement(model, "root")
        ET.SubElement(root, "mxCell", id="0")
        ET.SubElement(root, "mxCell", id="1", parent="0")
        for key, label, x, y, w, h, size in [
            ("title", title, 64, 45, 1450, 55, 34),
            ("subtitle", subtitle, 64, 104, 1450, 50, 19),
            ("note", note, 64, 818, 1472, 65, 17),
        ]:
            c = ET.SubElement(root, "mxCell", id=key, value=label, vertex="1", parent="1",
                              style=f"text;html=0;whiteSpace=wrap;fontFamily=Noto Sans CJK SC;fontSize={size};align=left;fontColor={INK};")
            ET.SubElement(c, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), **{"as":"geometry"})
        for key,n in self.nodes.items():
            shape = "rhombus" if n["diamond"] else "rectangle;rounded=1;arcSize=10"
            c = ET.SubElement(root, "mxCell", id=key, value=n["label"], vertex="1", parent="1",
                style=f"shape={shape};whiteSpace=wrap;html=0;fontFamily=Noto Sans CJK SC;fontSize={n['size']*1.35:.1f};fillColor={n['fill']};strokeColor={n['stroke']};fontColor={INK};strokeWidth=2;")
            ET.SubElement(c,"mxGeometry",x=str(n['x']),y=str(n['y']),width=str(n['w']),height=str(n['h']),**{"as":"geometry"})
        for i,e in enumerate(self.edges):
            s,t=self.nodes[e['source']],self.nodes[e['target']]
            start,end=e['points'][0],e['points'][-1]
            style=(f"edgeStyle=none;rounded=0;html=0;endArrow=block;endFill=1;strokeWidth=2;"
                   f"fontSize=16;fontFamily=Noto Sans CJK SC;labelBackgroundColor=#ffffff;strokeColor={e['color']};"
                   f"exitX={(start[0]-s['x'])/s['w']};exitY={(start[1]-s['y'])/s['h']};exitPerimeter=0;"
                   f"entryX={(end[0]-t['x'])/t['w']};entryY={(end[1]-t['y'])/t['h']};entryPerimeter=0;")
            c=ET.SubElement(root,"mxCell",id=f"edge{i}",edge="1",parent="1",source=e['source'],target=e['target'],style=style)
            g=ET.SubElement(c,"mxGeometry",relative="1",**{"as":"geometry"})
            a=ET.SubElement(g,"Array",**{"as":"points"})
            for x,y in e['points'][1:-1]:ET.SubElement(a,"mxPoint",x=str(x),y=str(y))
            if e['label']:
                x,y=e['label_xy']
                c=ET.SubElement(root,"mxCell",id=f"label{i}",value=e['label'],vertex="1",parent="1",
                    style=f"text;html=0;align=center;fontSize=16;fontColor={e['color']};fillColor=#ffffff;")
                ET.SubElement(c,"mxGeometry",x=str(x-85),y=str(y-15),width="170",height="30",**{"as":"geometry"})


def flows():
    mxfile=ET.Element("mxfile",host="app.diagrams.net",version="24.7.17")
    title="当前默认方案：照度门 + 加权融合"
    subtitle="暗场仅用 RGB；亮场融合 RGB 与骨架，再统一判定动作"
    fig,ax=canvas(1,title,subtitle,"当前代码 · 默认参数")
    d=Diagram(ax,"alpha-with-luma")
    d.node("input",64,355,145,100,"视频输入",GRAY_BG,MUTED)
    d.node("prep",255,335,230,140,"长边 ≤720\n记录原图亮度\n低照度增强",GRAY_BG,MUTED)
    d.node("rgb",530,350,190,110,"RGB 动作识别\nVideoMamba")
    d.node("gate",765,325,210,160,"原图亮度\nL < 75？",AMBER_BG,AMBER,True)
    d.node("dark",1030,210,255,95,"仅用 RGB 结果")
    d.node("skel",1030,440,255,100,"骨架动作识别\nRTMO-s + ST-GCN",TEAL_BG,TEAL)
    d.node("fuse",1030,590,255,95,"加权融合\n0.7 RGB + 0.3 骨架",TEAL_BG,TEAL)
    d.node("result",1340,340,205,130,"取最高分结果\n具名动作 ≥0.65\n否则输出「其他」",GRAY_BG,MUTED,size=15)
    d.edge("input","prep",[(209,405),(255,405)])
    d.edge("prep","rgb",[(485,405),(530,405)])
    d.edge("rgb","gate",[(720,405),(765,405)])
    d.edge("gate","dark",[(870,325),(870,257),(1030,257)],"是 · 暗场",(950,240),BLUE)
    d.edge("gate","skel",[(870,485),(870,490),(1030,490)],"否 · 亮场",(950,472),TEAL)
    d.edge("skel","fuse",[(1157,540),(1157,590)],color=TEAL)
    d.edge("dark","result",[(1285,257),(1310,257),(1310,385),(1340,385)],color=BLUE)
    d.edge("fuse","result",[(1285,637),(1310,637),(1310,430),(1340,430)],color=TEAL)
    panel(ax,64,741,1480,68)
    text(ax,88,775,"共同输入：最近 1 秒采样为 32 帧；识别使用增强后、超分前画面。",15)
    note="L 是增强前原图的窗口平均像素亮度（0–255），不是 lux；骨架无人时按 soft 策略投「其他」票。"
    footer(ax,note)
    d.xml(mxfile,title,subtitle,note)
    save(fig,"01_照度门与加权融合",title,"按当前 v2 代码绘制。先保存缩放后、增强前帧的平均亮度；RGB 使用增强帧。L<75 时仅 RGB；否则融合，alpha=0.7。骨架无人时 soft，不是硬否决。骨架加载失败时退回 RGB。流程省略出流、去噪、片段保存等不影响分类决策的步骤。")

    title="三部曲方案：照度门 → 取高分 → 先验判断"
    subtitle="与默认方案共用照度门；亮场改用分支竞争，并对指定动作追加必要条件"
    fig,ax=canvas(2,title,subtitle,"可选模式 · max_prior")
    d=Diagram(ax,"three-step-with-luma")
    d.node("prep",64,205,340,105,"视频输入 → 记录原图亮度 → 增强\n最近 1 秒采样为 32 帧",GRAY_BG,MUTED,size=15)
    d.node("gate",480,185,210,145,"原图亮度\nL < 75？",AMBER_BG,AMBER,True)
    d.node("dark",775,210,245,95,"仅用 RGB 结果")
    d.node("branches",64,430,290,115,"RGB 与骨架分别识别\n骨架最多 6 人",BLUE_BG,BLUE)
    d.node("candidate",405,430,285,115,"骨架取最高合格候选\n非「其他」且分数 >0.65\n无候选则沿用 RGB",TEAL_BG,TEAL,size=14)
    d.node("winner",745,430,285,115,"两分支取置信度更高者\n低于 0.65 →「其他」",RED_BG,RED,size=15)
    d.node("need",1120,405,240,165,"属于需要\n先验的动作？",AMBER_BG,AMBER,True)
    d.node("prior",1080,625,320,105,"物体 / 骨架先验\n满足：保留动作\n不满足：改为「其他」",RED_BG,RED,size=15)
    d.node("result",585,710,340,80,"最终阈值判定\n输出动作或「其他」",GRAY_BG,MUTED,size=16)
    d.edge("prep","gate",[(404,257),(480,257)])
    d.edge("gate","dark",[(690,257),(775,257)],"是 · 暗场",(732,234),BLUE)
    d.edge("gate","branches",[(585,330),(585,370),(209,370),(209,430)],"否 · 亮场",(400,370),TEAL)
    d.edge("branches","candidate",[(354,487),(405,487)])
    d.edge("candidate","winner",[(690,487),(745,487)])
    d.edge("winner","need",[(1030,487),(1120,487)])
    d.edge("need","prior",[(1240,570),(1240,625)],"是",(1260,597),RED)
    d.edge("need","result",[(1360,487),(1485,487),(1485,775),(925,775)],"否 · 直接保留",(1455,558),TEAL)
    d.edge("prior","result",[(1080,678),(985,678),(985,735),(925,735)],color=RED)
    d.edge("dark","result",[(1020,257),(1545,257),(1545,808),(755,808),(755,790)],color=BLUE)
    note="先验覆盖：喝水、捡东西（物体 + 骨架）；挥手、握手（骨架几何）。其他动作没有先验校验。"
    footer(ax,note)
    d.xml(mxfile,title,subtitle,note)
    save(fig,"02_三部曲方案",title,"按当前 max_prior 代码绘制，不采用旧会议记录中过时的增强后亮度门或挥手过脖规则。骨架候选来自合格人员，非 other 且置信度>0.65；无候选则沿用 RGB。最高分<0.65时直接 other。挥手当前规则为手腕或肘高于躯干中线；握手为两人手腕接近。检测器不可用时物体先验放行，正常运行但检测不到目标则可能否决。最终拒识阈值0.65。")
    ET.indent(mxfile,space="  ")
    raw=ET.tostring(mxfile,encoding="utf-8",xml_declaration=True)
    (OUT/"模型运行流程_最新双方案.drawio").write_bytes(raw)
    # Historical meeting materials must not overwrite the current per-person diagram.
    for i,page in enumerate(mxfile.findall('diagram'),1):
        single=ET.Element('mxfile',host='app.diagrams.net',version='24.7.17')
        single.append(page)
        (OUT/f'0{i}_流程图.drawio').write_bytes(ET.tostring(single,encoding='utf-8',xml_declaration=True))


def comparison():
    title="三部曲同时增加负样本触发，并漏掉正确动作"
    fig,ax=canvas(3,title,"同一批历史 pn 回放：原图亮度门 75、动作阈值 0.65；加权融合 α=0.7","历史 840 配置")
    p=[r for r in ROWS if r['suite']=='P'];c=[r for r in ROWS if r['suite']=='C']
    counts=[sum(outcome(r,k)!='other' for r in p) for k in ['a0.70','mp']]
    assert counts==[1,150]
    text(ax,85,205,"平台回传负样本 P · 越低越好",20,weight="bold")
    a=axes_at(fig,125,285,520,385)
    bars=a.bar([0,1],np.array(counts)/len(p)*100,color=[TEAL,RED],width=.55)
    a.set_ylim(0,40);a.set_yticks([0,10,20,30,40]);a.set_ylabel("触发窗口比例（%）",fontsize=13)
    a.set_xticks([0,1],["照度门 + 融合","三部曲"]);a.grid(axis='y',color=LINE)
    clean_plot(a)
    for b,n in zip(bars,counts):
        a.text(b.get_x()+b.get_width()/2,b.get_height()+1.4,f"{n}/460\n{n/460:.1%}",ha='center',fontsize=18,weight='bold',color=INK)
    text(ax,810,205,"现场正样本 C · 越高越好",20,weight="bold")
    text(ax,810,250,"总命中：融合 18/18    三部曲 7/18",16)
    a=axes_at(fig,835,330,650,340)
    names=['wave','chase','drink','pick_up'];x=np.arange(4);width=.32
    ns=[sum(r['expect']==n for r in c) for n in names]
    groups=[]
    for offset,key,color,label in [(-width/2,'a0.70',TEAL,'照度门 + 融合'),(width/2,'mp',RED,'三部曲')]:
        values=[sum(outcome(r,key)==n for r in c if r['expect']==n) for n in names]
        groups.append(values)
        bars=a.bar(x+offset,[v/n*100 for v,n in zip(values,ns)],width,color=color,label=label)
        for b,v,n in zip(bars,values,ns):a.text(b.get_x()+width/2,b.get_height()+3,f'{v}/{n}',ha='center',fontsize=13,color=color,weight='bold')
    assert groups==[[4,3,6,5],[4,3,0,0]]
    a.set_ylim(0,120);a.set_yticks([0,50,100]);a.set_ylabel('正确命中比例（%）',fontsize=13)
    a.set_xticks(x,[ZH[n] for n in names]);a.grid(axis='y',color=LINE);clean_plot(a)
    a.legend(loc='upper left',bbox_to_anchor=(0,1.2),frameon=False,ncol=2,fontsize=12)
    panel(ax,75,726,1450,73,RED_BG)
    text(ax,100,762,"正确的喝水、捡东西窗口共 11 个，全部被物体先验否决；误报也没有被整体拦住。",17,color=RED,weight='bold')
    footer(ax,"历史旧工具记录，含舍入字段及训练内、重叠窗口；用于解释失效机制，不是泛化精度或新 720 包验收成绩。")
    save(fig,"03_两方案效果对比",title,"来源 sources/historical-pn-840.jsonl 与 historical-pn-lum-sweep.txt。P=96段回传片段的460个窗口，C=4段现场视频的18个窗口。历史pn工具有舍入与无人策略漂移，精确部署口径须用修复后的schema2重导。P用于pn训练且含标注歧义，C也属于训练内；不能解释为独立测试泛化成绩。此图保留同次历史对照，不与后续镜像默认alpha的2/460混用。")


def objects():
    title="画面不暗，物体先验仍会漏掉真实动作"
    fig,ax=canvas(4,title,"同一现场帧：原图与增强图都只检出人；蓝框为模型输出，橙虚线为人工观察区","本次单帧诊断")
    records=json.loads((OUT/'evidence/object-detections.json').read_text())['records']
    for x,stem,action,roi,crop,n in [
        (65,'喝水16','喝水',(364,112,405,151),(335,85,435,185),6),
        (845,'捡东西2','捡东西',(445,286,501,316),(416,260,526,335),5),
    ]:
        text(ax,x,199,f'{stem} · 原始画面',20,weight='bold')
        img=np.array(Image.open(OUT/'sources'/f'{stem}-raw.jpg'))
        a=axes_at(fig,x,235,690,388);a.imshow(img);a.axis('off')
        raw=next(r for r in records if r['name']==stem and r['kind']=='raw')
        enh=next(r for r in records if r['name']==stem and r['kind']=='enhanced')
        assert raw['non_person']==enh['non_person']==0
        for det in raw['detections']:
            x1,y1,x2,y2,conf,cls=det
            a.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor=BLUE,lw=2))
            a.text(x1,y1-7,f'person {conf:.2f}',fontsize=10,color='white',bbox=dict(facecolor=BLUE,pad=2,edgecolor='none'))
        x1,y1,x2,y2=roi
        a.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='#F0AA30',lw=2.3,linestyle='--'))
        b=axes_at(fig,x,657,176,140);b.imshow(img[crop[1]:crop[3],crop[0]:crop[2]],interpolation='nearest');b.axis('off')
        text(ax,x+193,675,"局部放大 · 人工观察区",13,color=AMBER)
        text(ax,x+193,716,"原图 / 增强图：所需物体均未检出",14,weight='bold')
        text(ax,x+193,763,f'历史回放：{n}/{n} 个{action}窗口被先验否决',14,color=RED)
    footer(ax,"诊断：包内 YOLO26m，输入 640，阈值 0.25，CPU；仅各 1 帧的原图/增强图，不据此估计整体召回率。")
    save(fig,"04_物体漏检现场图",title,"使用既有审查中保存的840×472 JPEG，在当前包的YoloOnnxDetector预处理和后处理下运行CPU ONNX诊断。显式CPU session仅用于功能观察，绕过生产GPU性能加载门。四张图均只检测出person，分数约0.934–0.940。橙色虚线是人工观察区域，不是模型框或标注真值；放大图不生成或补画物体。本次没有执行动作模型，6/6与5/5否决来自独立标注的历史逐窗回放。")


def false_positives():
    title="取高分放大骨架误判，先验覆盖也不完整"
    fig,ax=canvas(5,title,"三部曲的 150 个负样本触发窗口：149 个由骨架分支胜出，110 个没有适用先验","历史 pn 回放")
    p=[r for r in ROWS if r['suite']=='P' and outcome(r,'mp')!='other']
    counts=Counter(outcome(r,'mp') for r in p)
    assert len(p)==150 and sum(r['branch']=='skel' for r in p)==149
    keys=['chase','throw','fall','shake_hands','wave','pick_up']
    assert sum(counts[k] for k in ['chase','throw','fall'])==110
    text(ax,75,204,"触发类别及先验路径",20,weight='bold')
    a=axes_at(fig,140,282,610,395)
    bars=a.barh(np.arange(6),[counts[k] for k in keys],height=.58,color=[RED]*3+[AMBER]*3)
    a.set_yticks(np.arange(6),[ZH[k] for k in keys]);a.invert_yaxis();a.set_xlim(0,105)
    a.set_xlabel('触发窗口数',fontsize=13);a.grid(axis='x',color=LINE);clean_plot(a)
    for b,key in zip(bars,keys):a.text(b.get_width()+2,b.get_y()+b.get_height()/2,str(counts[key]),va='center',fontsize=16,weight='bold',color=INK)
    panel(ax,75,725,690,67,GRAY_BG)
    text(ax,95,758,"红色：110 个无先验    棕色：40 个经先验放行",14)
    example=next(r for r in ROWS if r['name']=='20261012_094823_drinking_water_0029@t0.5' and r['suite']=='P')
    text(ax,845,204,"一个真实日志窗口的分支竞争",20,weight='bold')
    panel(ax,845,265,690,102,BLUE_BG)
    text(ax,870,300,f"RGB：其他 {example['rgb'][9]:.1%}",20,color=BLUE,weight='bold')
    text(ax,870,341,f"RGB 对「追逐」仅给出 {example['rgb'][2]:.2%}",14)
    panel(ax,845,395,690,102,RED_BG)
    text(ax,870,430,f"骨架候选：追逐 {max(example['mp']):.1%}",20,color=RED,weight='bold')
    text(ax,870,471,"三部曲取更高分 → 追逐；该动作没有先验",14)
    panel(ax,845,530,690,130,TEAL_BG)
    text(ax,870,568,"同窗加权融合 → 其他",20,color=TEAL,weight='bold')
    text(ax,870,617,f"融合后追逐分数 {example['a0.70'][2]:.1%}，低于 0.65",15)
    text(ax,845,713,"分支分数高，并不能保证动作判断正确。",17,weight='bold')
    text(ax,845,760,"案例：P0029 / t0.5；融合值取自日志，非另行推算。",11,color=MUTED)
    footer(ax,"历史 pn / 长边 840；两模式的骨架汇总方式不同。P 为回传负样本记录，含训练内数据及标注歧义。")
    save(fig,"05_取高分与先验失效",title,"统计来自同一历史pn JSONL，避免沿用旧权重177个全挥手的结论。当前这次历史pn三部曲150个触发分布为追逐91、投掷18、跌倒1、握手20、挥手10、捡东西10；骨架胜出149、RGB胜出1。前三类没有先验，共110；后三类各自通过先验，共40。例子为P0029 t0.5，RGB other .8403，骨架候选chase .904451，融合chase .248。alpha骨架采用两槽汇总，三部曲在最多六人中选候选，因此不能把候选.904直接代入alpha公式复算.248。")


def brightness():
    title="照度门能避开暗场，却修复不了这些亮场错误"
    fig,ax=canvas(6,title,"C 与 P 的历史窗口亮度全部高于 75：两套方案都会继续使用骨架分支","增强前原图亮度")
    a=axes_at(fig,135,285,1320,300)
    rng=np.random.default_rng(20261015)
    a.axvspan(0,75,color=BLUE_BG);a.axvspan(75,180,color=TEAL_BG)
    a.axvline(75,color=AMBER,lw=2,ls='--')
    for y,suite,color in [(1,'C',TEAL),(0,'P',RED)]:
        vals=np.array([r['lum'] for r in ROWS if r['suite']==suite])
        assert vals.min()>=75
        a.scatter(vals,y+rng.uniform(-.12,.12,len(vals)),s=23 if suite=='C' else 12,
                  color=color,alpha=.8 if suite=='C' else .4,edgecolors='none')
        a.text(177,y,f'{vals.min():.1f}–{vals.max():.1f}',ha='right',va='center',fontsize=14,color=color,weight='bold')
    a.set_xlim(0,180);a.set_ylim(-.6,1.6);a.set_xticks([0,30,60,75,90,120,150,180])
    a.set_yticks([1,0],['现场正样本 C\n18 窗口','平台回传 P\n460 窗口'])
    a.set_xlabel('窗口平均像素亮度 L（0–255；不是 lux）',fontsize=14,labelpad=15)
    a.text(37.5,1.37,'仅 RGB',ha='center',fontsize=17,color=BLUE)
    a.text(110,1.37,'允许骨架参与',ha='center',fontsize=17,color=TEAL)
    a.text(76,1.05,'门限 75',fontsize=12,color=AMBER)
    clean_plot(a);a.spines['left'].set_visible(False)
    panel(ax,75,675,400,116,AMBER_BG)
    text(ax,100,710,"0 / 478",29,color=AMBER,weight='bold')
    text(ax,100,758,"C、P 窗口触发暗场旁路",14)
    panel(ax,510,675,1025,116,GRAY_BG)
    text(ax,535,715,"照度门没有介入这些窗口，无法阻止取高分误判或物体漏检否决。",16,weight='bold')
    text(ax,535,758,"需要分别改善分支决策和目标可见性；不能把问题都归因于亮度。",15)
    footer(ax,"来源：历史 pn / 840 的原图亮度记录，保留 0.1 精度；每个点对应一个窗口，纵向抖动仅用于避免重叠。")
    save(fig,"06_照度门作用边界",title,"历史C原图窗口平均亮度135.0–154.1，P为135.5–149.6；门限75，478个窗口均不走暗场RGB旁路。该图只说明为什么增加门75不能修复这两套数据上的三部曲问题，不建议据此把门限盲目调到150。亮度值是图像平均BT.601 luma，而非照度仪lux；不能推导现场真实照度。")


def package():
    from build_editable_pptx import build
    build(FIGURES)
    thumbs=[]
    for name,_,_ in FIGURES:
        with Image.open(OUT/'png'/f'{name}.png') as img:
            thumbs.append(img.convert('RGB').resize((800,450),Image.Resampling.LANCZOS))
    contact=Image.new('RGB',(1600,1350),'#d9e1eb')
    for i,img in enumerate(thumbs):contact.paste(img,((i%2)*800,(i//2)*450))
    contact.save(OUT/'总览.jpg',quality=92)
    summary={}
    for suite in 'ABCP':
        sub=[r for r in ROWS if r['suite']==suite]
        summary[suite]=dict(n=len(sub),luma_min=min(r['lum'] for r in sub),luma_max=max(r['lum'] for r in sub))
        for key in ['a0.70','mp']:
            pred=[outcome(r,key) for r in sub]
            summary[suite][key]=dict(triggered=sum(p!='other' for p in pred),correct=sum(p==r['expect'] for p,r in zip(pred,sub)),classes=dict(Counter(pred)))
    (OUT/'evidence/chart-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'evidence/chart-data.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['suite','name','expected','raw_luma','alpha_luma75','three_step_luma75','winning_branch','prior_note'])
        for r in ROWS:w.writerow([r['suite'],r['name'],r['expect'],r['lum'],outcome(r,'a0.70'),outcome(r,'mp'),r['branch'],r['note']])
    (OUT/'讲解备注.md').write_text('# 会议图说明\n\n'+'\n\n'.join(f'## {i+1}. {title}\n\n{notes}' for i,(_,title,notes) in enumerate(FIGURES))+'\n')


if __name__=='__main__':
    flows();comparison();objects();false_positives();brightness();package()
    print('Built',len(FIGURES),'figures, two-page draw.io, and PPTX.')
