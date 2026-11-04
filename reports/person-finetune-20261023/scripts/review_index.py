"""Local HTML index for annotation evidence and before/after demonstration videos."""
import html
import json
from pathlib import Path

OUT=Path(__file__).resolve().parents[1]


def main() -> None:
    annotations=json.loads((OUT/'annotations.json').read_text())['records']
    lines=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>逐人动作标注核对</title>',
           '<style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:16px;line-height:1.6}video{width:720px;max-width:100%}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px;text-align:left}code{font-size:13px}.pending{background:#fff4df}</style>',
           '<h1>逐人动作标注核对</h1><p>这里展示带编号的轨迹和人工核对结果，不使用模型预测生成真值。'+
           '无稳定轨迹的视频仍记录在数据覆盖表中；下面只列已核对轨迹。不确定轨迹暂不参与训练和测试。</p>',
           '<table><tr><th>源视频/人物编号</th><th>状态与动作时间段（秒）</th><th>备注</th></tr>']
    for row in sorted(annotations,key=lambda r:(r['source'],r['track_id'])):
        url=f'data/{row["video_id"]}/review/tracks.mp4'
        intervals='；'.join(f'{s["start"]:.2f}–{s["end"]:.2f}: {s["label"]}'for s in row['segments']) or '未定，已排除'
        cls=' class="pending"'if row['status']!='reviewed'else ''
        lines.append(f'<tr{cls}><td><a href="{url}">{html.escape(row["source"])}</a><br>T{row["track_id"]:02d}</td>'+
                     f'<td>{row["status"]}<br>{html.escape(intervals)}</td><td>{html.escape(row["note"])}</td></tr>')
    lines.append('</table><p>时间边界为视觉核对的近似位置。过渡窗口按 protocol.json 中固定规则保守过滤。</p></html>')
    (OUT/'annotations-review.html').write_text('\n'.join(lines))
    resultpath=OUT/'demo/sample-results.json'
    if resultpath.exists():
        results=json.loads(resultpath.read_text())
        comparison_path=OUT/'demo-comparison.json'
        comparisons={r['name']:r for r in json.loads(comparison_path.read_text())}if comparison_path.exists()else {}
        lines=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>逐人微调结果演示</title>',
               '<style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:16px;line-height:1.6}video{width:960px;max-width:100%}section{padding:16px 0;border-top:1px solid #ccc}</style>',
               '<h1>逐人 VideoMamba 微调结果</h1><p>每个人使用自己的 RGB 裁剪和骨架，框下显示融合动作；阈值0.75，低分显示其他。'+
               '对照视频左侧为旧模型，右侧为逐人微调模型；新旧输入及骨架完全对齐。'+
               '这些是演示视频，包含训练内样本。数值对比与局限请查看 <a href="analysis/analysis-report.md">配对评估报告</a>。</p>',
               '<p>仍有不足：喝水测试命中11/11→8/11，三个错误窗来自喝水34；'+
               '骨架融合还会压低正确动作，检测框也可能框住影子或展品。画面中的T编号是短期轨迹，不是身份真值。</p>']
        for row in results:
            comparison=comparisons.get(row['name'])
            primary=comparison['comparison']if comparison else 'demo/'+row['video']
            poster=''if comparison else f' poster="demo/{row["still"]}"'
            lines.append(f'<section><h2>{html.escape(row["name"])}</h2><p>{html.escape(row["source"])}</p>'+
                         f'<video controls preload="metadata" src="{html.escape(primary,quote=True)}"{poster}></video>'+
                         f'<p><a href="demo/{row["video"]}">仅看微调后视频（原尺寸）</a> · '+
                         f'<a href="demo/events/{row["name"]}.json">逐人预测明细</a></p></section>')
        lines.append('</html>');(OUT/'demo-index.html').write_text('\n'.join(lines))


if __name__=='__main__':main()
