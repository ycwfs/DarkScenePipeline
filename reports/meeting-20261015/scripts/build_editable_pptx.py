"""Native PowerPoint shapes/connectors/charts; only field photographs are raster images."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import re
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OUT / '.tools'))
from pptx import Presentation
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import (XL_CHART_TYPE, XL_DATA_LABEL_POSITION, XL_MARKER_STYLE,
                             XL_TICK_MARK, XL_TICK_LABEL_POSITION)
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

INK, MUTED, LINE = '#172B43', '#58697D', '#D9E1EB'
BLUE, BLUE_BG = '#2463B6', '#EBF2FB'
TEAL, TEAL_BG = '#00867B', '#E9F5F2'
RED, RED_BG = '#C54F38', '#FCF0EB'
AMBER, AMBER_BG = '#AB701C', '#FBF3E3'
GRAY_BG = '#F4F6F9'
FONT = 'Noto Sans CJK SC'
BASE = '暗光行为识别_方案与三部曲问题_20261015'
LABELS = ['wave','throw','chase','fall','fight','talk','drink','pick_up','shake_hands','other']
ZH = dict(zip(LABELS,['挥手','投掷','追逐','跌倒','打架','交谈','喝水','捡东西','握手','其他']))
ROWS = [json.loads(s) for s in (OUT/'sources/historical-pn-840.jsonl').read_text().splitlines()]


def emu(value):
    return Inches(value / 100)


def rgb(value):
    return RGBColor.from_string(value.lstrip('#'))


def font_style(font, size=16, color=INK, bold=False):
    font.name = FONT
    font.size = Pt(size)
    font.bold = bold
    font.color.rgb = rgb(color)
    # Set the East Asian face explicitly so Chinese text remains editable across viewers.
    props = font._rPr
    ea = props.find('{http://schemas.openxmlformats.org/drawingml/2006/main}ea')
    if ea is None:
        ea = OxmlElement('a:ea')
        props.append(ea)
    ea.set('typeface', FONT)


def fill_line(shape, fill=None, stroke=None, width=1.3):
    # Override the stock Office theme's shadow, including for native text boxes.
    props=shape._element.spPr
    if props.find('{http://schemas.openxmlformats.org/drawingml/2006/main}effectLst') is None:
        props.append(OxmlElement('a:effectLst'))
    if fill:
        shape.fill.solid(); shape.fill.fore_color.rgb = rgb(fill)
    else:
        shape.fill.background()
    if stroke:
        shape.line.color.rgb = rgb(stroke); shape.line.width = Pt(width)
    else:
        shape.line.fill.background()


def set_text(shape, value, size=16, color=INK, bold=False, align='left', margin=3):
    tf = shape.text_frame
    tf.clear(); tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = emu(margin)
    tf.margin_top = tf.margin_bottom = emu(2)
    for i, line in enumerate(value.split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {'left': PP_ALIGN.LEFT, 'center': PP_ALIGN.CENTER, 'right': PP_ALIGN.RIGHT}[align]
        p.space_before = p.space_after = Pt(0)
        p.line_spacing = 1.12
        run = p.add_run(); run.text = line
        font_style(run.font, size, color, bold)


def textbox(slide, x, y, w, h, value, size=16, color=INK, bold=False, align='left', name=None, fill=None):
    shape = slide.shapes.add_textbox(emu(x), emu(y), emu(w), emu(h))
    shape.name = name or ('文字 · ' + value.replace('\n',' / ')[:45])
    fill_line(shape, fill)
    set_text(shape, value, size, color, bold, align)
    return shape


def box(slide, x, y, w, h, fill=GRAY_BG, stroke=None, label='', size=16,
        diamond=False, name=None, align='center', color=INK):
    kind = MSO_AUTO_SHAPE_TYPE.DIAMOND if diamond else MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE
    shape = slide.shapes.add_shape(kind, emu(x), emu(y), emu(w), emu(h))
    if not diamond:
        shape.adjustments[0] = .12
    shape.name = name or ('流程框 · '+label.replace('\n',' / '))
    fill_line(shape, fill, stroke)
    if label:
        set_text(shape, label, size, color, False, align, 7)
    return shape


def line(slide, start, end, color=LINE, width=1.3, arrow=False, name='连线'):
    shape = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
        emu(start[0]), emu(start[1]), emu(end[0]), emu(end[1]))
    shape.name = name
    shape.line.color.rgb = rgb(color); shape.line.width = Pt(width)
    shape._element.spPr.append(OxmlElement('a:effectLst'))
    if arrow:
        end_tag = OxmlElement('a:tailEnd')
        end_tag.set('type','triangle'); end_tag.set('w','med'); end_tag.set('len','med')
        shape.line._get_or_add_ln().append(end_tag)
    return shape


def header(deck, number, title, subtitle, badge=''):
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    slide.background.fill.solid(); slide.background.fill.fore_color.rgb = rgb('#FFFFFF')
    s = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE,emu(64),emu(53),emu(8),emu(54))
    fill_line(s,TEAL);s.name='标题装饰条'
    textbox(slide,92,43,1370,65,title,27,bold=True,name='标题')
    textbox(slide,92,99,1235,44,subtitle,14,MUTED,name='副标题')
    textbox(slide,1465,40,75,45,f'{number:02d}',20,TEAL,align='right',name='页码')
    textbox(slide,1330,103,207,34,badge,12,MUTED,align='right',name='数据口径标记')
    return slide


def footer(slide, value):
    line(slide,(64,835),(1536,835))
    textbox(slide,64,840,1472,47,value,10.5,MUTED,name='口径与来源')


def prediction(p):
    i=int(np.argmax(p));return LABELS[i] if p[i]>=.65 else 'other'


def outcome(row,key):
    return prediction(row['rgb'] if row['lum']<75 else row[key])


def parse_style(raw):
    return dict(s.split('=',1) for s in raw.split(';') if '=' in s)


def attach(connector, node, start, point):
    """Keep endpoint-to-node relationships in editable native connector XML."""
    x,y,w,h=node.left,node.top,node.width,node.height
    px,py=emu(point[0]),emu(point[1])
    distances=[abs(py-y),abs(px-x),abs(py-y-h),abs(px-x-w)]
    idx=int(np.argmin(distances))
    props=connector._element.find('{http://schemas.openxmlformats.org/presentationml/2006/main}nvCxnSpPr/{http://schemas.openxmlformats.org/presentationml/2006/main}cNvCxnSpPr')
    link=OxmlElement('a:stCxn' if start else 'a:endCxn')
    link.set('id',str(node.shape_id));link.set('idx',str(idx));props.append(link)


def flow_slides(deck):
    pages=ET.parse(OUT/'模型运行流程_最新双方案.drawio').getroot().findall('diagram')
    for number,page in enumerate(pages,1):
        cells={c.get('id'):c for c in page.findall('.//mxCell')}
        slide=header(deck,number,cells['title'].get('value'),cells['subtitle'].get('value'),
                     '当前代码 · 默认参数' if number==1 else '可选模式 · max_prior')
        nodes={};specs={}
        for key,c in cells.items():
            if c.get('vertex')!='1' or key in {'title','subtitle','note'} or key.startswith('label'):
                continue
            g=c.find('mxGeometry');st=parse_style(c.get('style',''))
            x,y,w,h=[float(g.get(k)) for k in ['x','y','width','height']]
            value='需要先验\n判断？' if key=='need' else c.get('value')
            nodes[key]=box(slide,x,y,w,h,st['fillColor'],st['strokeColor'],value,
                           float(st['fontSize'])/1.35,st.get('shape')=='rhombus',name=f'流程框 · {key}')
            specs[key]=(x,y,w,h)
        for key,c in cells.items():
            if c.get('edge')!='1':continue
            source,target=c.get('source'),c.get('target');st=parse_style(c.get('style',''))
            x,y,w,h=specs[source];a=(x+float(st['exitX'])*w,y+float(st['exitY'])*h)
            x,y,w,h=specs[target];b=(x+float(st['entryX'])*w,y+float(st['entryY'])*h)
            points=[a]+[(float(p.get('x')),float(p.get('y'))) for p in c.findall('.//mxPoint')]+[b]
            points=[p for i,p in enumerate(points) if i==0 or p!=points[i-1]]
            for i,(u,v) in enumerate(zip(points,points[1:])):
                segment=line(slide,u,v,st.get('strokeColor',MUTED),1.5,i==len(points)-2,
                             f'连接 {source} → {target} · {i+1}')
                if i==0:attach(segment,nodes[source],True,u)
                if i==len(points)-2:attach(segment,nodes[target],False,v)
                # Place lines behind boxes; every segment stays a native editable connector.
                slide.shapes._spTree.remove(segment._element)
                slide.shapes._spTree.insert(2,segment._element)
        for key,c in cells.items():
            if not key.startswith('label'):continue
            g=c.find('mxGeometry');st=parse_style(c.get('style',''))
            x,y,w,h=[float(g.get(k)) for k in ['x','y','width','height']]
            value=c.get('value').replace(' · ','·')
            nw=len(value)*18+16
            x+= (w-nw)/2;w=nw;y+=4;h-=8
            if number==2 and key=='label2':y-=24
            if number==2 and key=='label7':x=1350
            label=textbox(slide,x,y,w,h,value,12,st.get('fontColor',MUTED),align='center',name='分支条件 · '+c.get('value'))
            label.text_frame.word_wrap=False
        if number==1:
            box(slide,64,741,1480,68,GRAY_BG,name='输入说明背景')
            textbox(slide,88,748,1430,54,'共同输入：最近 1 秒采样为 32 帧；识别使用增强后、超分前画面。',15)
        footer(slide,cells['note'].get('value'))


def chart_style(chart, maximum, unit, number_format='0', grid=True):
    chart.has_title=False;chart.has_legend=False
    font_style(chart.font,12,MUTED)
    chart.chart_style=10
    for axis in [chart.category_axis,chart.value_axis]:
        font_style(axis.tick_labels.font,12,MUTED)
        axis.major_tick_mark=XL_TICK_MARK.NONE;axis.minor_tick_mark=XL_TICK_MARK.NONE
        axis.format.line.color.rgb=rgb(LINE)
    axis=chart.value_axis
    axis.minimum_scale=0;axis.maximum_scale=maximum;axis.major_unit=unit
    axis.tick_labels.number_format=number_format
    axis.has_major_gridlines=grid
    if grid:axis.major_gridlines.format.line.color.rgb=rgb(LINE)
    chart.plots[0].gap_width=90


def series_fill(series,color):
    series.format.fill.solid();series.format.fill.fore_color.rgb=rgb(color)
    series.format.line.fill.background()


def labels(chart,fmt='0',size=13):
    plot=chart.plots[0];plot.has_data_labels=True
    dl=plot.data_labels;dl.show_value=True;dl.position=XL_DATA_LABEL_POSITION.OUTSIDE_END
    dl.number_format=fmt;font_style(dl.font,size,INK,True)


def comparison(deck):
    slide=header(deck,3,'三部曲同时增加负样本触发，并漏掉正确动作',
        '同一批历史 pn 回放：原图亮度门 75、动作阈值 0.65；加权融合 α=0.7','历史 840 配置')
    textbox(slide,85,180,660,48,'平台回传负样本 P · 越低越好',20,bold=True)
    textbox(slide,90,235,650,38,'触发窗口：融合 1/460；三部曲 150/460',15)
    data=CategoryChartData();data.categories=['照度门 + 融合','三部曲']
    p=[r for r in ROWS if r['suite']=='P']
    values=[sum(outcome(r,k)!='other' for r in p)/len(p) for k in ['a0.70','mp']]
    data.add_series('负样本触发比例',values)
    frame=slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED,emu(80),emu(283),emu(660),emu(400),data)
    frame.name='可编辑图表 · P负样本触发率';chart=frame.chart
    chart_style(chart,.4,.1,'0%');labels(chart,'0.0%',17)
    for point,color in zip(chart.series[0].points,[TEAL,RED]):
        point.format.fill.solid();point.format.fill.fore_color.rgb=rgb(color);point.format.line.fill.background()
    textbox(slide,810,180,680,48,'现场正样本 C · 越高越好',20,bold=True)
    textbox(slide,810,235,680,38,'总命中：融合 18/18；三部曲 7/18',15)
    box(slide,850,286,20,15,TEAL,name='图例色块 · 融合')
    textbox(slide,878,273,260,40,'照度门 + 融合',12)
    box(slide,1150,286,20,15,RED,name='图例色块 · 三部曲')
    textbox(slide,1178,273,180,40,'三部曲',12)
    c=[r for r in ROWS if r['suite']=='C'];names=['wave','chase','drink','pick_up']
    ns=[sum(r['expect']==n for r in c) for n in names]
    data=CategoryChartData();data.categories=[f'{ZH[n]}\n(n={den})' for n,den in zip(names,ns)]
    for key,title in [('a0.70','照度门 + 融合'),('mp','三部曲')]:
        data.add_series(title,[sum(outcome(r,key)==n for r in c if r['expect']==n)/den for n,den in zip(names,ns)])
    frame=slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED,emu(795),emu(320),emu(720),emu(365),data)
    frame.name='可编辑图表 · C正样本命中率';chart=frame.chart
    chart_style(chart,1.2,.5,'0%');labels(chart,'0%',12)
    for series,color in zip(chart.series,[TEAL,RED]):series_fill(series,color)
    box(slide,75,726,1450,73,RED_BG,name='结论背景')
    textbox(slide,100,738,1400,49,'正确的喝水、捡东西窗口共 11 个，全部被物体先验否决；误报也没有被整体拦住。',17,RED,True)
    footer(slide,'历史旧工具记录，含舍入字段及训练内、重叠窗口；用于解释失效机制，不是泛化精度或新 720 包验收成绩。')


def photo(slide,path,x,y,w,h,name,crop=None):
    picture=slide.shapes.add_picture(str(path),emu(x),emu(y),width=emu(w),height=emu(h))
    picture.name=name
    if crop:
        with Image.open(path) as image:iw,ih=image.size
        x1,y1,x2,y2=crop
        picture.crop_left=x1/iw;picture.crop_right=(iw-x2)/iw
        picture.crop_top=y1/ih;picture.crop_bottom=(ih-y2)/ih
    return picture


def outline(slide,x,y,w,h,color,name,dashed=False):
    from pptx.enum.dml import MSO_LINE_DASH_STYLE
    s=slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE,emu(x),emu(y),emu(w),emu(h))
    s.name=name;fill_line(s,None,color,1.8)
    if dashed:s.line.dash_style=MSO_LINE_DASH_STYLE.DASH
    return s


def objects(deck):
    slide=header(deck,4,'画面不暗，物体先验仍会漏掉真实动作',
        '同一现场帧：原图与增强图都只检出人；蓝框为模型输出，橙虚线为人工观察区','本次单帧诊断')
    records=json.loads((OUT/'evidence/object-detections.json').read_text())['records']
    for x,stem,action,roi,crop,n in [
        (65,'喝水16','喝水',(364,112,405,151),(335,85,435,185),6),
        (845,'捡东西2','捡东西',(445,286,501,316),(416,260,526,335),5),
    ]:
        textbox(slide,x,175,690,48,f'{stem} · 原始画面',20,bold=True)
        path=OUT/'sources'/f'{stem}-raw.jpg'
        h=690*472/840;photo(slide,path,x,235,690,h,'现场照片 · '+stem)
        record=next(r for r in records if r['name']==stem and r['kind']=='raw')
        scale=690/840
        for x1,y1,x2,y2,conf,cls in record['detections']:
            outline(slide,x+x1*scale,235+y1*scale,(x2-x1)*scale,(y2-y1)*scale,BLUE,'模型检测框 · '+stem)
            textbox(slide,x+x1*scale,235+y1*scale-23,110,23,f'person {conf:.2f}',10,'#FFFFFF',fill=BLUE,name='检测置信度 · '+stem)
        x1,y1,x2,y2=roi
        outline(slide,x+x1*scale,235+y1*scale,(x2-x1)*scale,(y2-y1)*scale,'#F0AA30','人工观察区 · '+stem,True)
        cw,ch=crop[2]-crop[0],crop[3]-crop[1];s=min(176/cw,140/ch)
        photo(slide,path,x+(176-cw*s)/2,657+(140-ch*s)/2,cw*s,ch*s,'可重新裁剪的局部图 · '+stem,crop)
        textbox(slide,x+193,650,500,45,'局部放大 · 人工观察区',13,AMBER)
        textbox(slide,x+193,694,500,48,'原图 / 增强图：所需物体均未检出',14,bold=True)
        textbox(slide,x+193,742,500,46,f'历史回放：{n}/{n} 个{action}窗口被先验否决',14,RED)
    footer(slide,'诊断：包内 YOLO26m，输入 640，阈值 0.25，CPU；仅各 1 帧的原图/增强图，不据此估计整体召回率。')


def false_positives(deck):
    slide=header(deck,5,'取高分放大骨架误判，先验覆盖也不完整',
        '三部曲的 150 个负样本触发窗口：149 个由骨架分支胜出，110 个没有适用先验','历史 pn 回放')
    textbox(slide,75,180,690,48,'触发类别及先验路径',20,bold=True)
    keys=['pick_up','wave','shake_hands','fall','throw','chase']
    counts=Counter(outcome(r,'mp') for r in ROWS if r['suite']=='P' and outcome(r,'mp')!='other')
    data=CategoryChartData();data.categories=[ZH[k] for k in keys]
    data.add_series('触发窗口数',[counts[k] for k in keys])
    frame=slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED,emu(75),emu(270),emu(695),emu(420),data)
    frame.name='可编辑图表 · 三部曲负样本触发类别';chart=frame.chart
    chart_style(chart,105,20);labels(chart,'0',15)
    for point,key in zip(chart.series[0].points,keys):
        point.format.fill.solid();point.format.fill.fore_color.rgb=rgb(RED if key in {'chase','throw','fall'} else AMBER)
        point.format.line.fill.background()
    textbox(slide,250,680,360,35,'触发窗口数',12,MUTED,align='center')
    box(slide,75,725,690,67,GRAY_BG,name='图例背景')
    textbox(slide,95,736,650,46,'红色：110 个无先验    棕色：40 个经先验放行',14)
    r=next(r for r in ROWS if r['suite']=='P' and r['name']=='20261012_094823_drinking_water_0029@t0.5')
    textbox(slide,845,180,690,48,'一个真实日志窗口的分支竞争',20,bold=True)
    for y,h,bg,title,detail,color in [
        (265,102,BLUE_BG,f"RGB：其他 {r['rgb'][9]:.1%}",f"RGB 对「追逐」仅给出 {r['rgb'][2]:.2%}",BLUE),
        (395,102,RED_BG,f"骨架候选：追逐 {max(r['mp']):.1%}",'三部曲取更高分 → 追逐；该动作没有先验',RED),
        (530,130,TEAL_BG,'同窗加权融合 → 其他',f"融合后追逐分数 {r['a0.70'][2]:.1%}，低于 0.65",TEAL),
    ]:
        box(slide,845,y,690,h,bg,name='分支说明背景')
        textbox(slide,870,y+10,640,48,title,20,color,True)
        textbox(slide,870,y+56,640,h-61,detail,14)
    textbox(slide,845,688,690,48,'分支分数高，并不能保证动作判断正确。',17,bold=True)
    textbox(slide,845,740,690,38,'案例：P0029 / t0.5；融合值取自日志，非另行推算。',11,MUTED)
    footer(slide,'历史 pn / 长边 840；两模式的骨架汇总方式不同。P 为回传负样本记录，含训练内数据及标注歧义。')


def brightness(deck):
    slide=header(deck,6,'照度门能避开暗场，却修复不了这些亮场错误',
        'C 与 P 的历史窗口亮度全部高于 75：两套方案都会继续使用骨架分支','增强前原图亮度')
    textbox(slide,200,210,600,42,'照度门：L <75 → 仅 RGB',17,BLUE)
    textbox(slide,850,210,650,42,'L ≥75 → 允许骨架参与',17,TEAL)
    bounds=[]
    for suite in ['P','C']:
        values=[r['lum'] for r in ROWS if r['suite']==suite]
        bounds.append((min(values),max(values)))
    data=CategoryChartData();data.categories=['平台回传 P（460 窗口）','现场正样本 C（18 窗口）']
    data.add_series('最小亮度（透明起点）',[lo for lo,hi in bounds])
    data.add_series('最大值减最小值（范围）',[hi-lo for lo,hi in bounds])
    frame=slide.shapes.add_chart(XL_CHART_TYPE.BAR_STACKED,emu(200),emu(275),emu(1300),emu(335),data)
    frame.name='可编辑图表 · C/P原始亮度区间';chart=frame.chart
    chart_style(chart,180,30,grid=False)
    chart.plots[0].gap_width=300
    chart.category_axis.tick_label_position=XL_TICK_LABEL_POSITION.NONE
    chart.series[0].format.fill.background();chart.series[0].format.line.fill.background()
    chart.series[1].format.line.fill.background()
    for point,color in zip(chart.series[1].points,[RED,TEAL]):
        point.format.fill.solid();point.format.fill.fore_color.rgb=rgb(color)
        point.format.line.fill.background()
    # Fixed plot layout keeps the native threshold annotation aligned to the value axis.
    plot=chart._chartSpace.chart.plotArea
    layout=plot.find('{http://schemas.openxmlformats.org/drawingml/2006/chart}layout')
    if layout is None:
        layout=OxmlElement('c:layout')
        plot.insert(0,layout)
    manual=OxmlElement('c:manualLayout');layout.append(manual)
    target=OxmlElement('c:layoutTarget');target.set('val','inner');manual.append(target)
    for axis in ['x','y','w','h']:
        mode=OxmlElement(f'c:{axis}Mode');mode.set('val','factor');manual.append(mode)
    for name,val in [('x',.025),('y',.03),('w',.945),('h',.84)]:
        child=OxmlElement('c:'+name);child.set('val',str(val));manual.append(child)
    gate_x=200+1300*(.025+.945*75/180)
    gate=line(slide,(gate_x,290),(gate_x,562),AMBER,1.8,name='可编辑门限线 · L=75')
    from pptx.enum.dml import MSO_LINE_DASH_STYLE
    gate.line.dash_style=MSO_LINE_DASH_STYLE.DASH
    textbox(slide,65,330,135,75,'现场正样本 C\n18 窗口',13,MUTED,align='right')
    textbox(slide,65,470,135,75,'平台回传 P\n460 窗口',13,MUTED,align='right')
    textbox(slide,1320,340,190,35,'135.0–154.1',14,TEAL,True,align='right')
    textbox(slide,1320,482,190,35,'135.5–149.6',14,RED,True,align='right')
    textbox(slide,gate_x+8,288,150,35,'门限 75',12,AMBER)
    textbox(slide,400,612,1000,40,'窗口平均像素亮度 L（0–255；不是 lux）',14,MUTED,align='center')
    box(slide,75,675,400,116,AMBER_BG,name='旁路统计背景')
    textbox(slide,100,680,350,58,'0 / 478',29,AMBER,True)
    textbox(slide,100,735,350,45,'C、P 窗口触发暗场旁路',14)
    box(slide,510,675,1025,116,GRAY_BG,name='结论背景')
    textbox(slide,535,684,975,52,'照度门没有介入这些窗口，无法阻止取高分误判或物体漏检否决。',16,bold=True)
    textbox(slide,535,738,975,42,'需要分别改善分支决策和目标可见性；不能把问题都归因于亮度。',15)
    footer(slide,'来源：历史 pn / 840 的原图亮度记录；色条为各套件窗口亮度的最小值至最大值，不表示样本密度。')


def build(figures=None):
    deck=Presentation();deck.slide_width=Inches(16);deck.slide_height=Inches(9)
    flow_slides(deck);comparison(deck);objects(deck);false_positives(deck);brightness(deck)
    if figures is None:
        sections=re.split(r'\n## \d+\. ',(OUT/'讲解备注.md').read_text())[1:]
        notes=[s.strip() for s in sections]
    else:
        notes=[title+'\n\n'+note for _,title,note in figures]
    assert len(notes)==len(deck.slides)==6
    for slide,note in zip(deck.slides,notes):
        slide.notes_slide.notes_text_frame.text=note+'\n\n编辑说明：文字/流程框/连线均为原生对象；统计图可右键编辑数据。现场照片保留为图片，检测框与标注可单独调整。'
        for shape in slide.shapes:
            for effect in shape._element.xpath('.//a:effectRef'):
                effect.set('idx','0')
    original=OUT/(BASE+'.pptx')
    backup=OUT/'sources/previous-image-only.pptx'
    if original.exists() and not backup.exists():shutil.copy2(original,backup)
    editable=OUT/(BASE+'_可编辑.pptx');deck.save(editable)
    shutil.copy2(editable,original)
    print(f'Built native editable PPTX: {editable}')
    return editable


if __name__=='__main__':
    build()
