"""Local review page for actual recorded stream, clips, and four offline probes."""
import html
import json
from pathlib import Path

OUT=Path(__file__).resolve().parents[1]
def esc(s):return html.escape(str(s),quote=True)
sections=['<h1>逐人物体先验 · 最终包验证</h1>',
    '<p>默认 RGB 0.972 / 骨架 0.028，原图门 35，动作 0.925，物体 0.581。原 0.3.0 镜像。</p>',
    '<p>此页视频来自真实运行，无人工纠正动作。完整推流录制含预热及收尾，不添加黑白帧号条；此页验证有限回放，不是独立准确率评估。</p>',
    '<h2>外部接收的完整输出流</h2><video controls preload="metadata" src="full-output-stream.mp4"></video>',
    '<p><a href="input-multiaction-5min.mp4">五分钟输入</a> · <a href="input-manifest.json">29段素材时间索引</a> · <a href="verification.json">核验结果</a> · <a href="../algorithm-flow-20261026/README.md">可编辑流程图</a></p>',
    '<h2>4 段原镜像单卡完整输出</h2>']
for directory in sorted((OUT/'functional').iterdir()):
    video=directory/'output.mp4'
    if not video.exists():continue
    rel=video.relative_to(OUT)
    sections.append(f'<details><summary>{esc(directory.name)}</summary><video controls preload="none" src="{esc(rel)}"></video><p><a href="{esc((directory/"events.json").relative_to(OUT))}">完整逐人事件</a></p></details>')
sections.append('<h2>实时算子实际保存的片段</h2>')
for path in sorted((OUT/'live/clips').rglob('*.json')):
    if not path.with_suffix('.mp4').exists():continue
    c=json.loads(path.read_text());video=path.with_suffix('.mp4')
    assert video.exists(),video
    title=f'T{c["person_id"]} · {c["action_key"]} · {c["start_source_time"]:.2f}–{c["end_source_time"]:.2f}s · 上下文完整={c["context_complete"]}'
    sections.append(f'<details><summary>{esc(title)}</summary><video controls preload="none" src="{esc(video.relative_to(OUT))}"></video><p><a href="{esc(path.relative_to(OUT))}">保存元数据</a></p></details>')
(OUT/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>逐人物体先验交付验证</title><style>body{font:16px system-ui;max-width:1100px;margin:36px auto;padding:0 20px;color:#182d42;background:#f5f7fa}video{display:block;width:100%;max-height:650px;background:#111;margin:14px 0}details{padding:14px;background:white;border:1px solid #d7e2ea;margin:12px 0;border-radius:8px}summary{cursor:pointer}a{color:#126b8a}</style>'+''.join(sections)+'</html>')
