import json
from pathlib import Path
import numpy as np
from run import OUT,greedy_match
rows=json.loads((OUT/'matched/records.json').read_text())
aux=[r for r in rows if r['cohort']=='auxiliary44']; hard=[r for r in rows if r['cohort']=='quartile24']
def agg(rr,c,det='formal'):
 g=sum(len(r['gt_boxes_native']) for r in rr); m=sum(r['conditions'][c][det]['matched'] for r in rr);p=sum(len(r['conditions'][c][det]['pred_boxes_native']) for r in rr);u=sum(r['conditions'][c][det]['unmatched'] for r in rr);ii=[z for r in rr for z in r['conditions'][c][det]['ious']]
 return g,m,p,u,float(np.mean(ii)) if ii else float('nan')
lines=['# 统计附录：暗场增强与人体检测（扩展版）','', '辅助集和困难参考集均按帧配对；24帧困难集的参考框是视觉诊断框，不是穷尽人工GT。未匹配预测不能直接当误检。','', '## 正式检测器：44帧辅助集', '', '| 输入 | 命中 | 召回 | 预测 | 未匹配预测 | 平均IoU |','|---|---:|---:|---:|---:|---:|']
for c,l in [('raw_native','Raw native'),('raw720','Raw 720'),('ntire720','NTIRE 720'),('ft720','Onsite FT 720')]:
 g,m,p,u,i=agg(aux,c);lines.append(f'| {l} | {m}/{g} | {m/g:.4f} | {p} | {u} | {i:.4f} |')
lines += ['', '## 正式检测器：24帧困难参考集', '', '| 输入 | 命中 | 召回 | 预测 | 未匹配预测 | 平均IoU |','|---|---:|---:|---:|---:|---:|']
for c,l in [('raw_native','Raw native'),('raw720','Raw 720'),('ntire720','NTIRE 720'),('ft720','Onsite FT 720'),('raw1280','Raw 1280'),('ntire1280','NTIRE 1280'),('ft1280','Onsite FT 1280')]:
 g,m,p,u,i=agg(hard,c);lines.append(f'| {l} | {m}/{g} | {m/g:.4f} | {p} | {u} | {i:.4f} |')
lines += ['', '## 原始检测器对照：44帧', '', '| 输入 | 命中 | 召回 | 预测 | 未匹配预测 | 平均IoU |','|---|---:|---:|---:|---:|---:|']
for c,l in [('raw_native','Raw native'),('raw720','Raw 720'),('ntire720','NTIRE 720'),('ft720','Onsite FT 720')]:
 g,m,p,u,i=agg(aux,c,'original');lines.append(f'| {l} | {m}/{g} | {m/g:.4f} | {p} | {u} | {i:.4f} |')
lines += ['', '## 阈值扫描：44帧辅助集', '', '阈值扫描在统一CPU FP32输出上从保存的0.10候选重算，正式后处理包含宽度>8、高度>20、NMS=0.75。','', '| 输入 | 0.10 | 0.25 | 0.35 | 0.50 |','|---|---:|---:|---:|---:|']
for c,l in [('raw720','Raw720'),('ntire720','NTIRE720'),('ft720','Onsite FT720')]:
 vals=[]
 for t in [.1,.25,.35,.5]:
  m=p=un=0
  for r in aux:
   pred=np.asarray(r['conditions'][c]['formal']['low_conf_boxes_native'],np.float32)
   scores=np.asarray(r['conditions'][c]['formal']['low_conf_scores'])
   # scores and boxes are the .1 person-class output; use threshold then one-to-one match.
   keep=scores>=t; pred=pred[keep];gt=np.asarray(r['gt_boxes_native'],np.float32)
   ma=greedy_match(gt,pred)
   m+=len(ma);p+=len(pred);un+=len(pred)-len(ma)
  vals.append(f'{m}/44; +{un} unmatched')
 lines.append('| '+l+' | '+' | '.join(vals)+' |')
lines += ['', '## 人物尺寸', '', '44帧辅助框平均原图高度349.9 px、720高度130.9 px；24帧困难参考框720高度均值127.3 px，中位120.9 px，最小60 px。640 letterbox后的困难参考高度均值113.2 px、中位107.5 px、最小53.3 px。', '', '## 统计限制', '', '- 相邻帧相关，未作独立样本显著性检验；本报告是配对描述统计。', '- 困难集参考框来自历史候选和视觉复核，非用户确认、非穷尽GT；空框帧可能是漏标或无真人。', '- 原始检测器只作模型对照；生产使用正式清理检测器。', '- CPU FP32增强结果用于输入效果对照，不能外推GPU FP16速度。']
(OUT/'stats-appendix.md').write_text('\n'.join(lines)+'\n')
