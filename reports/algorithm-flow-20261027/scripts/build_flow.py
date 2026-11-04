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
for key in ('lum_gate','reco_min_conf','skel_fusion_alpha','object_min_conf','person_object_prior','wave_pose_min_frames','running_pose_min_speed'):
    assert getattr(a,key)==c[key],key
alpha=c['skel_fusion_alpha'];pose=1-alpha;gate=c['lum_gate'];tau=c['reco_min_conf'];obj=c['object_min_conf']
f.OUT=OUT;f.BASE='逐人动作识别_完整先验方案_20261027'
N,E=f.Node,f.Edge
f.NODES=[
 N('scope',55,315,1490,270,'','#FAFBFD',f.LINE),
 N('title',55,28,1490,70,'逐人动作识别 · 完整先验方案','none','none',28,'text',bold=True,align='left'),
 N('defaults',55,102,1490,42,f'默认：RGB {alpha:g} ／ 骨架 {pose:.3f}    ·    原图照度门 {gate:g}    ·    动作阈值 {tau:g}    ·    物体阈值 {obj:g}','none','none',17,'text',color=f.TEAL,align='left'),
 N('prior_defaults',55,137,1490,27,f'新增几何先验：挥手至少 {c["wave_pose_min_frames"]} 帧    ·    跑动至少 {c["running_pose_min_speed"]:g} 人体框高 / 秒','none','none',14,'text',color=f.TEAL,align='left'),
 N('input',55,177,215,85,'视频输入\n文件 / 实时流',f.GRAY_BG,f.MUTED,18),
 N('enhance',325,177,345,85,'保留增强前亮度 L → 增强\nRetinexformer · 公开预训练\n处理长边 720',f.GRAY_BG,f.MUTED,15),
 N('people',725,177,365,85,'YOLO26m 人体检测＋跟踪\n现场微调 · 每人独立 ID',f.GRAY_BG,f.MUTED,17),
 N('window',1145,177,400,85,'逐人时序窗口\n最近 1 秒 / 32 帧 · 每 0.5 秒提交',f.GRAY_BG,f.MUTED,16),
 N('rgb',100,385,290,115,'逐人 ROI · 现场微调\nVideoMamba-Tiny\nRGB 概率 P_RGB',f.BLUE_BG,f.BLUE,16),
 N('gate',465,385,185,115,'',f.AMBER_BG,f.AMBER,17,'diamond'),
 N('gate_text',465,398,185,90,f'原图亮度\nL < {gate:g}？','none','none',17,'text'),
 N('rgb_only',755,332,290,75,'只用本人的 RGB\n跳过骨架与三类先验',f.BLUE_BG,f.BLUE,16),
 N('skeleton',755,472,290,90,'RTMO-s → 同人 ST-GCN\n现场姿态 / 骨架动作微调',f.TEAL_BG,f.TEAL,15.5),
 N('fusion',1110,472,385,90,f'同人融合\n{alpha:g} P_RGB + {pose:.3f} P_pose\n无合格骨架：soft 投「其他」票',f.TEAL_BG,f.TEAL,15),
 N('decision',1110,332,385,90,f'最高类别动作分数 ≥ {tau:g}：保留\n低于阈值：其他',f.GRAY_BG,f.MUTED,17),
 N('yes',640,340,110,30,'是 · 暗场','none','none',13,'text',color=f.BLUE),
 N('no',640,514,110,30,'否','none','none',13,'text',color=f.TEAL),
 N('prior_title',55,602,1490,32,'亮场按动作检查先验 · 低亮 / 原图亮度未知：跳过','none','none',16,'text',bold=True,align='left'),
 N('objects',55,653,480,112,f'喝水 / 捡物 · 现场七类 YOLO26m\n720 增强图裁剪 · 5 帧中任一兼容物体 ≥ {obj:g}\n有效观测无支持：其他；观测不可用：保留',f.BLUE_BG,f.BLUE,15),
 N('wave',565,653,480,112,f'挥手 · 同一只腕高于双肩\n1 秒窗口累计 ≥ {c["wave_pose_min_frames"]} 个不同源帧\n不足：其他（腕 / 双肩置信度 ≥ 0.5）',f.TEAL_BG,f.TEAL,16),
 N('running',1075,653,470,112,f'跑动 · 共同可靠关节平均移动速度\n≥ {c["running_pose_min_speed"]:g} 人体框高 / 秒：保留\n低于：其他；可靠姿态不足：保留',f.AMBER_BG,f.AMBER,16),
 N('output',55,802,1490,65,'人体框＋骨架＋青色物体框（类别/置信度/人物 ID）→ 实时输出\n非其他按人物 ID＋动作保存完整画面 · 前后各 2 秒 · 最长 15 秒 · 多人可重叠',f.GRAY_BG,f.MUTED,16),
 N('note',55,875,1490,22,'L 为增强前原图亮度（0–255），不是 lux。骨架与物体逐帧也受照度门约束；不做推理平滑。挥手 / 跑动不新增模型推理；模型来源见第 2 页。','none','none',11,'text',color=f.MUTED,align='left'),
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
 E('decision','objects',[(1495,377),(1570,377),(1570,637),(295,637),(295,653)]),
 E('decision','wave',[(1495,377),(1570,377),(1570,637),(805,637),(805,653)]),
 E('decision','running',[(1495,377),(1570,377),(1570,637),(1310,637),(1310,653)]),
 E('objects','output',[(295,765),(295,802)]),E('wave','output',[(805,765),(805,802)]),E('running','output',[(1310,765),(1310,802)]),
]
f.NOTES=f'''挥手：wave_pose_min_frames={c['wave_pose_min_frames']}；同一腕高于双肩累计3个不同源帧，亮场缺姿态拒绝，不要求连续。
跑动：running_pose_min_speed={c['running_pose_min_speed']}人体框高/秒；共同可靠关节约0.2秒间隔平均速度除框高，至少5关节、3对、跨度0.3秒；缺可靠速度放行。
两项在原图低亮/未知/骨架关闭时跳过，0可关闭；不做推理平滑。
2026-10-27 用户正式确认加入挥手与跑动先验，优先压低误报；六模型和四参数不变。
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
过滤后的label经物体→挥手→跑动检查后统一用于绘制、事件、并发片段；action_label/action_confidence保留过滤前值，object_prior记录原因与证据。
物体框显示检测类别/置信度/人物ID；~标记短时光流跟随，最多0.6秒，无法对齐/断流/低亮时隐藏，不用于动作决策。
输出按人物ID+动作保存完整画面，前后各2秒，最长15秒重叠续段；保存缓冲1.2秒对齐异步事件。上下文完整不代表识别完整。
模型与算法未再训练；四参数用旧现场与新9段反馈校准，优先压误报，有实测漏报代价。新唯一真喝水仍被先验过滤。
平台默认3卡，实时前2卡前端、其余动作+物体；双卡一前端一动作；离线第1卡前端。模型输入长边720，平台默认bicubic×2显示放大至长边1440。
2026-10-28核对：平台正式包full_priors_20261027；Box当前默认包Orin4GB_candidate_20261028，原六模型、四参数及三类先验一致。
Box默认单卡，128帧uint8人物缓存、异步公平轮转、同人待处理窗口合并、batch2；sr=off直接输出长边720。过载时合并窗口，因此并非每0.5秒的提交都必定完成推理。
Box在ARM64的auto后端使用TensorRT：人体/物体FP16、RTMO FP32且TF32关闭；姿态ONNX仅作等价ArgMax兼容改写。x86 auto仍使用ORT CUDA；增强仍为PyTorch，CUDA Graph默认关闭。
Box未训练或替换轻量模型。后端数值及任务调度会造成部分窗口结果差异，不承诺与平台逐窗一致；单RTX3090测试已完成，Orin Nano 4GB真机与JetPack尚未验收。
本次仅同步图和提交入口，平台/Box运行代码、归档和原0.3.0镜像不变；详见reports/delivery-consistency-20261028与reports/box-single-gpu-20261028。
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
nodes=[N('title2',55,30,1490,65,'模型训练状态与数据来源','none','none',28,'text',bold=True,align='left'),N('sub2',55,100,1490,42,'2026-10-28 核对：平台与 Box 共用以下模型及判断规则；Box 单卡优化版仍待 Orin 4GB 真机验收。','none','none',16,'text',color=f.MUTED,align='left')]
for i,(model,status,data) in enumerate(records):
 y=160+102*i
 nodes.extend([N(f'm{i}',55,y,360,86,model,f.BLUE_BG,f.BLUE,17),N(f's{i}',438,y,370,86,status,f.TEAL_BG,f.TEAL,16),N(f'd{i}',831,y,714,86,data,f.GRAY_BG,f.LINE,15,align='left')])
nodes.append(N('footer2',55,790,1490,90,'新 9 段多人录像未参与训练，后续用于误报反馈 / 阈值校准；优先压误报，仍有召回损失。\n平台：默认 3 卡 / ORT / 1440 显示放大；Box：1 卡 / ARM TensorRT / 720 直接输出；模型输入均为 720。\n相同规则不保证逐窗结果完全相同；Box 调度与后端差异见备注，Orin 4GB 尚未真机验收。','none','none',14,'text',color=f.MUTED,align='left'))
deck=f.Presentation(ppt);deck.core_properties.title='逐人动作识别：完整先验与默认参数（2026-10-27）'
slide=deck.slides.add_slide(deck.slide_layouts[6])
for n in nodes:f.native_node(slide,n)
slide.notes_slide.notes_text_frame.text=f.NOTES;deck.save(ppt)
tree=ET.parse(drawio);root=tree.getroot();root.find('diagram').set('name','当前算法与默认参数')
page=ET.SubElement(root,'diagram',id='training-20261027',name='模型训练与数据来源');graph=ET.SubElement(page,'mxGraphModel',page='1',pageWidth='1600',pageHeight='900');cells=ET.SubElement(graph,'root');ET.SubElement(cells,'mxCell',id='0');ET.SubElement(cells,'mxCell',id='1',parent='0')
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
assert all(str(x) in texts for x in (.972,.028,.925,.581,35,c['wave_pose_min_frames'],c['running_pose_min_speed']))
(OUT/'讲解备注.md').write_text(f.NOTES)
(OUT/'editable-qa.json').write_text(json.dumps(dict(slides=2,picture_objects=0,native_shapes=True,defaults=c,models_sha256=hashlib.sha256((op/'models.json').read_bytes()).hexdigest(),synchronized='2026-10-28',platform_release='dark-demo-v2_full_priors_20261027',box_release='NVIDIA-Box_Orin4GB_candidate_20261028',physical_orin_tested=False),ensure_ascii=False,indent=2)+'\n')
print(ppt);print(drawio)
