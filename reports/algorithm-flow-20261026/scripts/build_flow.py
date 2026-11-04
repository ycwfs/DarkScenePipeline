"""Native editable diagrams generated from deployed defaults and model provenance."""
from pathlib import Path
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
import zipfile
import hashlib

OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parents[1]
spec=importlib.util.spec_from_file_location('native_flow',ROOT/'reports/algorithm-flow-20261023/scripts/build_flow.py')
f=importlib.util.module_from_spec(spec);sys.modules[spec.name]=f;spec.loader.exec_module(f)
sys.path.insert(0,str(ROOT/'platform'))
from package_v2 import parser_of, validate_contract
op=ROOT/'platform/dark-demo-v2';validate_contract(op)
a=parser_of(op).parse_args(['--video_path','demo','--session_json','session.json'])
m=json.loads((op/'models.json').read_text());c=m['defaults']
for key in ('lum_gate','reco_min_conf','skel_fusion_alpha','object_min_conf','person_object_prior'):
    assert getattr(a,key)==c[key],key
alpha=c['skel_fusion_alpha'];pose=1-alpha;gate=c['lum_gate'];tau=c['reco_min_conf'];obj=c['object_min_conf']
f.OUT=OUT;f.BASE='逐人动作识别_物体先验与默认参数_20261026'
N,E=f.Node,f.Edge
f.NODES=[
 N('scope',55,315,1490,270,'','#FAFBFD',f.LINE),
 N('title',55,28,1490,70,'逐人动作识别 · 当前整体方案','none','none',28,'text',bold=True,align='left'),
 N('defaults',55,102,1490,42,f'默认：RGB {alpha:g} ／ 骨架 {pose:.3f}    ·    原图照度门 {gate:g}    ·    动作阈值 {tau:g}    ·    物体阈值 {obj:g}','none','none',17,'text',color=f.TEAL,align='left'),
 N('input',55,177,215,85,'视频输入\n文件 / 实时流',f.GRAY_BG,f.MUTED,18),
 N('enhance',325,177,345,85,'保留增强前亮度 L → 增强\nRetinexformer · 公开预训练\n处理长边 720',f.GRAY_BG,f.MUTED,15),
 N('people',725,177,365,85,'YOLO26m 人体检测＋跟踪\n现场微调 · 每人独立 ID',f.GRAY_BG,f.MUTED,17),
 N('window',1145,177,400,85,'逐人时序窗口\n最近 1 秒 / 32 帧 · 每 0.5 秒判断',f.GRAY_BG,f.MUTED,16),
 N('rgb',100,385,290,115,'逐人 ROI · 现场微调\nVideoMamba-Tiny\nRGB 概率 P_RGB',f.BLUE_BG,f.BLUE,16),
 N('gate',465,385,185,115,'',f.AMBER_BG,f.AMBER,17,'diamond'),
 N('gate_text',465,398,185,90,f'原图亮度\nL < {gate:g}？','none','none',17,'text'),
 N('rgb_only',755,332,290,75,'只用本人的 RGB\n跳过骨架及物体先验',f.BLUE_BG,f.BLUE,16),
 N('skeleton',755,472,290,90,'RTMO-s → 同人 ST-GCN\n现场姿态 / 骨架动作微调',f.TEAL_BG,f.TEAL,15.5),
 N('fusion',1110,472,385,90,f'同人融合\n{alpha:g} P_RGB + {pose:.3f} P_pose\n无合格骨架：soft 投「其他」票',f.TEAL_BG,f.TEAL,15),
 N('decision',1110,332,385,90,f'最高类别动作分数 ≥ {tau:g}：保留\n低于阈值：其他',f.GRAY_BG,f.MUTED,17),
 N('yes',640,340,110,30,'是 · 暗场','none','none',13,'text',color=f.BLUE),
 N('no',640,514,110,30,'否','none','none',13,'text',color=f.TEAL),
 N('object_gate',1240,620,270,135,'',f.AMBER_BG,f.AMBER,16,'diamond'),
 N('object_gate_text',1250,640,250,95,f'喝水 / 捡物\n且原图 L ≥ {gate:g}？','none','none',17,'text'),
 N('object_model',775,635,390,110,'现场七类 YOLO26m · 独立物体模型\n720 增强图 → 同人周围 / 局部裁剪\n窗口均匀取 5 帧',f.TEAL_BG,f.TEAL,15.5),
 N('support',305,635,395,110,f'任一帧检出兼容物体 ≥ {obj:g}\n有：保留动作；有效观测无物体：其他\n亮度未知 / 观测不可用：不否决',f.BLUE_BG,f.BLUE,15),
 N('object_yes',1168,652,72,32,'是','none','none',13,'text',color=f.TEAL),
 N('object_no',1490,733,72,32,'否','none','none',13,'text',color=f.MUTED),
 N('output',305,792,1240,75,'每人框下显示最终动作 → 实时输出；仅非「其他」触发保存\n按人物 ID＋动作保存完整画面 · 前后各 2 秒 · 最长 15 秒分段 · 多人片段可重叠',f.GRAY_BG,f.MUTED,16),
 N('note',55,875,1490,22,'L 为增强前原图亮度（0–255），不是 lux。骨架与物体逐帧也受照度门约束；不做推理平滑。模型来源见第 2 页。','none','none',11,'text',color=f.MUTED,align='left'),
]
f.EDGES=[
 E('input','enhance',[(270,219.5),(325,219.5)]),E('enhance','people',[(670,219.5),(725,219.5)]),E('people','window',[(1090,219.5),(1145,219.5)]),
 E('window','rgb',[(1345,262),(1345,292),(245,292),(245,385)]),
 E('rgb','gate',[(390,442.5),(465,442.5)]),
 E('gate','rgb_only',[(557.5,385),(557.5,369.5),(755,369.5)],f.BLUE),
 E('gate','skeleton',[(557.5,500),(557.5,517),(755,517)],f.TEAL),
 E('skeleton','fusion',[(1045,517),(1110,517)],f.TEAL),
 E('rgb_only','decision',[(1045,369.5),(1110,369.5)],f.BLUE),
 E('fusion','decision',[(1302.5,472),(1302.5,422)],f.TEAL),
 E('decision','object_gate',[(1495,377),(1570,377),(1570,600),(1375,600),(1375,620)]),
 E('object_gate','object_model',[(1240,687.5),(1165,687.5)],f.TEAL),
 E('object_model','support',[(775,687.5),(700,687.5)],f.TEAL),
 E('support','output',[(502.5,745),(502.5,792)]),
 E('object_gate','output',[(1510,687.5),(1570,687.5),(1570,829.5),(1545,829.5)]),
]
f.NOTES=f'''2026-10-26 用户接受四参数默认，并授权物体先验接入平台逐人实时流程及更新交付包。
默认 RGB={alpha:g}、骨架={pose:.3f}、lum_gate={gate:g}、reco_min_conf={tau:g}、object_min_conf={obj:g}、person_object_prior=true。
图按依赖组织；RTMO实际在前端提取，同人骨架经时间采样后输入ST-GCN。原图逐帧亮度及窗口平均亮度双重约束。
RGB为该人15%外扩ROI、400×224等比例填边；1秒32帧、每0.5秒提交。原图L<{gate:g}直接RGB。
无合格骨架按soft把{pose:.3f}权重给other；历史已增强回传切片的原L未知，保留原骨架规则，但默认物体先验弃权。
物体先验只检查喝水和捡物，不使用历史max_prior的跨分支取最大和几何规则。person_object_prior=false可回滚无先验。
物体模型是独立现场七类YOLO26m，非人体YOLO；5个采样位置0/8/16/23/31，重复源帧去重。
使用720增强图；原分辨率坐标上30%外扩人体框及可信头/手/腿局部区域，再准确映射到增强图裁剪、640letterbox。
每帧使用准确缓存的轨迹和姿态。与别人人体框IoU>0.4时视为绑定不确定，该观测弃权；全无有效观测不否决。
喝水允许瓶/杯；捡物允许瓶/杯/手机/相机/纸/便携物。一帧同人ROI内检测中心且分数≥{obj:g}足够；不要求连续全程检出。
有效亮场观测均无兼容物体则改其他；物体推理错误按实际故障上报，不能伪装成未检出或关闭先验。
过滤后的label统一用于绘制、事件、并发片段；action_label/action_confidence保留过滤前值，object_prior记录原因与证据。
输出按人物ID+动作保存完整画面，前后各2秒，最长15秒重叠续段；保存缓冲1.2秒对齐异步事件。上下文完整不代表识别完整。
模型与算法未再训练；四参数用旧现场与新9段反馈校准，优先压误报，有实测漏报代价。新唯一真喝水仍被先验过滤。
单卡共享GPU；实时≥3卡前2卡前端，其余动作+物体；双卡一前端一动作；离线第1卡前端。实体平台长期性能需上线核验。
'''
OUT.mkdir(exist_ok=True);ppt=f.make_ppt();drawio=f.make_drawio()
records=[
 ('Retinexformer · 增强','公开 NTIRE 预训练','未采用此前效果退化的现场增强微调；保留公开权重'),
 ('YOLO26m · 人体检测','现场微调 · 已剔除展品误标','1626 暗场帧 / 1741 人体框＋72 亮场帧 / 89 真人框；原图与增强视图'),
 ('RTMO-s · 17 点姿态','现场暗场微调 · 推理不平滑','现场暗场人体框＋辅助 COCO17 骨架序列；训练前质量筛选，非独立人工精标'),
 ('VideoMamba-Tiny · RGB','现场逐人 ROI 微调','全部旧现场＋两批线上 other；535 去重来源，人物 / 时间段审核；按来源组划分'),
 ('ST-GCN · 骨架动作','历史现场微调权重','real-actiondata-all 骨架缓存；新逐人适配已比较，验证选择保留历史模型'),
 ('YOLO26m · 七类物体','现场物体 ROI 微调 · 独立权重','瓶 / 杯 / 手机 / 相机 / 纸 / 便携物 / 展品；319 训练图、31 验证、47 测试；杯无标注，相机仅 3 个训练框'),
]
nodes=[N('title2',55,30,1490,65,'模型训练状态与数据来源','none','none',28,'text',bold=True,align='left'),N('sub2',55,100,1490,42,'模型不变；本轮将物体候选接入逐人实时流程，采用四参数联合校准默认值。','none','none',16,'text',color=f.MUTED,align='left')]
for i,(model,status,data) in enumerate(records):
 y=160+102*i
 nodes.extend([N(f'm{i}',55,y,360,86,model,f.BLUE_BG,f.BLUE,17),N(f's{i}',438,y,370,86,status,f.TEAL_BG,f.TEAL,16),N(f'd{i}',831,y,714,86,data,f.GRAY_BG,f.LINE,15,align='left')])
nodes.append(N('footer2',55,790,1490,90,'10 月 25 日新 9 段多人录像未参与模型训练，后续用于误报反馈及阈值校准。\n本组默认优先减少误报，存在召回损失；并非独立新场景泛化成绩。现场包六个主模型的文件及哈希见 models.json。','none','none',15,'text',color=f.MUTED,align='left'))
deck=f.Presentation(ppt);deck.core_properties.title='逐人动作识别：物体先验与默认参数（2026-10-26）'
slide=deck.slides.add_slide(deck.slide_layouts[6])
for n in nodes:f.native_node(slide,n)
slide.notes_slide.notes_text_frame.text=f.NOTES;deck.save(ppt)
tree=ET.parse(drawio);root=tree.getroot();root.find('diagram').set('name','当前算法与默认参数')
page=ET.SubElement(root,'diagram',id='training-20261026',name='模型训练与数据来源');graph=ET.SubElement(page,'mxGraphModel',page='1',pageWidth='1600',pageHeight='900');cells=ET.SubElement(graph,'root');ET.SubElement(cells,'mxCell',id='0');ET.SubElement(cells,'mxCell',id='1',parent='0')
for n in nodes:
 cell=ET.SubElement(cells,'mxCell',id=n.key,value=n.text,vertex='1',parent='1',style=f'rounded=1;whiteSpace=wrap;html=0;fillColor={n.fill};strokeColor={n.stroke};fontSize={n.size/.72:g};fontFamily={f.FONT};fontColor={n.color};align={n.align};')
 ET.SubElement(cell,'mxGeometry',x=str(n.x),y=str(n.y),width=str(n.w),height=str(n.h),**{'as':'geometry'})
ET.indent(root,space='  ');tree.write(drawio,encoding='utf-8',xml_declaration=True)
(ROOT/'platform/模型运行流程_v2.drawio').write_bytes(drawio.read_bytes())
with zipfile.ZipFile(ppt) as z:assert not any(n.startswith('ppt/media/') for n in z.namelist())
assert len(f.Presentation(ppt).slides)==2
for page in root.findall('diagram'):
 ids={x.get('id') for x in page.iter('mxCell')}
 for cell in page.iter('mxCell'):
  if cell.get('edge'):assert cell.get('source') in ids and cell.get('target') in ids
texts='\n'.join(s.text for sl in deck.slides for s in sl.shapes if s.has_text_frame)
assert all(str(x) in texts for x in (.972,.028,.925,.581,35))
(OUT/'讲解备注.md').write_text(f.NOTES)
(OUT/'editable-qa.json').write_text(json.dumps(dict(slides=2,picture_objects=0,native_shapes=True,defaults=c,models_sha256=hashlib.sha256((op/'models.json').read_bytes()).hexdigest()),ensure_ascii=False,indent=2)+'\n')
print(ppt);print(drawio)
