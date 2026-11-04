"""Build a per-recording review page and verify completed operator outputs."""
import argparse
from collections import Counter
import concurrent.futures
import csv
import html
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.parse import quote

from verify_inputs import decode

REPORT = Path(__file__).resolve().parents[1]


def save(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def local(paths: list) -> Path:
    for value in paths:
        if isinstance(value, str) and Path(value).is_file():
            return Path(value)
    raise ValueError(f'No local artifact: {paths}')


def link(path: Path) -> str:
    return quote(os.path.relpath(path, REPORT), safe='/')


def verify(item: dict, source: dict, number: int) -> dict:
    video = local(item['output_video'])
    events_path = local(item['events_json'])
    rows = json.loads(events_path.read_text())
    metadata = json.loads(subprocess.check_output(['ffprobe', '-v', 'error',
        '-show_streams', '-show_format', '-of', 'json', str(video)], text=True))
    stream = next(s for s in metadata['streams'] if s['codec_type']=='video')
    decoded = decode(str(video))
    clips = [local(paths) for paths in item['clip_videos']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        decoded_clips = list(pool.map(decode, map(str, clips)))
    clip_rows = []
    for path, decoded_clip in zip(clips, decoded_clips):
        data = json.loads(path.with_suffix('.json').read_text())
        clip_rows.append(dict(path=str(path), metadata=data, decode=decoded_clip))
    duration = float(metadata['format']['duration'])
    expected_duration = source['frames'] / (float(source['fps'].split('/')[0])/float(source['fps'].split('/')[1]))
    checks = dict(
        output_decode=decoded['exit_code']==0,
        frames_complete=decoded['frames']==source['frames']==item['stats']['frames'],
        duration_matches=abs(duration-expected_duration)<.12,
        events_count=len(rows)==item['events_total'],
        person_schema=all(r.get('schema_version')==3 and r.get('recognition_scope')=='person' for r in rows),
        clips_decode=all(r['exit_code']==0 for r in decoded_clips),
        clip_count=len(clips)==item['clips'].get('clips_saved',0),
        no_write_errors=not any(item['clips'].get(k,0) for k in ('write_errors','clips_unfinished','clips_failed','clips_forced')))
    preview = REPORT/'previews'/f'{number:02d}.jpg'
    preview.parent.mkdir(exist_ok=True)
    subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','2','-ss',str(min(30.,duration/2)),
        '-i',str(video),'-frames:v','1','-vf','scale=960:-2','-y',str(preview)],check=True)
    result = dict(number=number, source=source, output=str(video), events=str(events_path),
        preview=str(preview), duration=duration, width=stream['width'], height=stream['height'],
        frames=decoded['frames'], output_decode=decoded, checks=checks, passed=all(checks.values()),
        events_total=len(rows), label_counts=dict(Counter(r['label'] for r in rows)),
        clips_total=len(clips), clips=clip_rows, stats=item['stats'])
    return result


def render(results: list, inputs: list, finished: bool) -> None:
    n=len(results)
    cards=[]
    for r in results:
        name=html.escape(Path(r['source']['absolute_path']).name)
        counts='、'.join(f'{html.escape(k)} {v}' for k,v in r['label_counts'].items())
        clips=''.join(f'<li><a href="{link(Path(c["path"]))}">{html.escape(Path(c["path"]).name)}</a>'
            f' · 原片 {c["metadata"].get("start_source_time",0):.2f}–{c["metadata"].get("end_source_time",0):.2f} 秒'
            f' · {"上下文完整" if c["metadata"].get("context_complete") else "上下文不足，见元数据"}'
            f' · <a href="{link(Path(c["path"]).with_suffix(".json"))}">元数据</a></li>' for c in r['clips'])
        cards.append(f'''<article id="v{r['number']}"><h2>{r['number']:02d} · {name}</h2>
<video controls preload="none" poster="{link(Path(r['preview']))}" src="{link(Path(r['output']))}"></video>
<p>{r['duration']/60:.2f} 分钟 · {r['frames']:,} 帧 · {r['width']}×{r['height']} · 核验{'通过' if r['passed'] else '有异常'}</p>
<p><a href="{link(Path(r['output']))}">完整结果视频</a> · <a href="{link(Path(r['events']))}">逐人事件 JSON</a> ·
<a href="{link(Path(r['source']['absolute_path']))}">原始 AVI</a></p>
<p>人物窗口预测：{counts or '无有效动作窗口'}。这是预测次数，不是真值或独立动作次数。</p>
<details><summary>自动保存的动作片段：{r['clips_total']} 段</summary><ul>{clips or '<li>无非 other 动作片段</li>'}</ul></details></article>''')
    state='全部推理完成' if finished and n==len(inputs) else '正在推理／核验'
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>现场多人录像 · 逐段推理结果</title><style>
body{{font:16px/1.65 system-ui,sans-serif;background:#f3f5f8;color:#172338;margin:0 auto;padding:30px;max-width:1100px}}
article{{background:white;border:1px solid #d7dfe9;border-radius:10px;padding:22px;margin:25px 0}}h1{{font-size:28px}}
h2{{font-size:18px;overflow-wrap:anywhere}}video{{width:100%;background:#111}}a{{color:#145cb4}}li{{overflow-wrap:anywhere}}
.status{{padding:15px;background:#e5edf8;border-radius:8px}}</style>
<h1>现场多人录像 · 逐段推理结果</h1><p class="status">{state}：已核验 {n}/{len(inputs)} 段；原始视频总长 {sum(r['duration'] for r in inputs)/60:.2f} 分钟。</p>
<p>tau075 正式包／原 0.3.0 镜像。动作阈值 0.75，亮度门 30，RGB 权重 0.95；处理长边 720，输出放大 2 倍。
各人物独立识别和融合，框旁显示动作；自动片段前后各 2 秒。离线逐帧处理，保持源帧率，无音频、无测流黑白条。</p>
<p>本批未提供逐人真值标注，预测中的非 other 动作仍需人工核对。<a href="summary.csv">结果索引 CSV</a> · <a href="verification.json">完整核验记录</a></p>
{''.join(cards)}<p>更新于 {time.strftime('%Y-%m-%d %H:%M:%S %Z')}。处理期间可刷新本页查看已完成结果。</p></html>'''
    (REPORT/'index.html').write_text(page)
    with (REPORT/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['number','source','seconds','frames','events','clips','verification','output','events_json']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in results:
            writer.writerow(dict(number=r['number'],source=r['source']['absolute_path'],seconds=r['duration'],
                frames=r['frames'],events=r['events_total'],clips=r['clips_total'],verification=r['passed'],output=r['output'],events_json=r['events']))
    save(REPORT/'verification.json',dict(finished=finished,files_verified=n,files_total=len(inputs),
        passed=n==len(inputs) and all(r['passed'] for r in results),items=results))


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    inputs=json.loads((REPORT/'input-manifest.json').read_text())
    by_path={r['absolute_path']:r for r in inputs}
    previous=REPORT/'verification.json'
    results=json.loads(previous.read_text())['items'] if previous.exists() else []
    seen={r['source']['absolute_path'] for r in results}
    while True:
        path=REPORT/'completed-items.jsonl'
        items=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
        for item in items:
            if item['file'] not in seen:
                result=verify(item,by_path[item['file']],item.get('batch_number',len(results)+1))
                results.append(result);seen.add(item['file'])
                results.sort(key=lambda r:r['number'])
                print('VERIFIED',len(results),item['file'],result['checks'],flush=True)
                render(results,inputs,False)
        finished=(REPORT/'completion.json').exists()
        render(results,inputs,finished)
        if finished or not args.watch:
            break
        time.sleep(10)
    if finished and (len(results)!=len(inputs) or not all(r['passed'] for r in results)):
        raise SystemExit(1)


if __name__=='__main__':
    main()
