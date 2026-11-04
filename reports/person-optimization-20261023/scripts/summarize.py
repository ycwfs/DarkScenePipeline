"""Aggregate measured optimization runs without treating dropped work as acceleration."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]


def read(path):
    return json.loads(path.read_text())


def quantile(values, q):
    return float(np.percentile(values,q)) if values else None


def live_summary(folder):
    summaries = []
    for path in sorted(folder.glob('*gpu/*/result.json')):
        if path.parent.name.startswith('warmup'):
            continue
        result = read(path)
        events, frames = read(path.parent/'events.json'), read(path.parent/'frames.json')
        ages = [r['decision_age'] for r in events]
        groups = {}
        for row in events:
            groups.setdefault((row['track_id'],row['history_generation']),[]).append(row)
        gaps, counts, update_gaps = [], [], []
        for rows in groups.values():
            by_source = sorted(rows,key=lambda r:r['timestamp'])
            by_completed = sorted(rows,key=lambda r:r['completed'])
            gaps.extend(b['timestamp']-a['timestamp'] for a,b in zip(by_source,by_source[1:]))
            update_gaps.extend(b['completed']-a['completed'] for a,b in zip(by_completed,by_completed[1:]))
            counts.append(len(rows))
        person_frames = sum(f['people'] for f in frames)
        classified_frames = sum(f['classified'] for f in frames)
        ready_person_frames = sum(len(f.get('ready_ids', [])) for f in frames)
        row = dict(run=str(path.parent.relative_to(OUT)),name=result['name'],cards=len(result['devices']),
            backend='process' if result.get('process_workers') else 'thread/inline',
            source_duration=result['source_duration'],elapsed=result['elapsed'],
            frames=result['frames'],source_frames=result['source_frames_offered'],
            unique_source_indices=len(set(result['source_indices'])),
            source_dropped=result['dropped_source_frames'],effective_fps=result['frames']/result['source_duration'],
            drained_processing_fps=result['processing_fps'],submitted=result['submitted'],
            completed=result['completed'],coalesced=result['coalesced'],
            decision_age_p50=quantile(ages,50),decision_age_p95=quantile(ages,95),
            decision_age_p99=quantile(ages,99),decision_age_max=max(ages)if ages else None,
            late_decisions=sum(a>1 for a in ages),frame_age_p95=quantile([f['output_age']for f in frames],95),
            frame_age_max=max(f['output_age']for f in frames),person_window_gap_p95=quantile(gaps,95),
            person_window_gap_max=max(gaps)if gaps else None,person_publish_gap_p95=quantile(update_gaps,95),
            person_publish_gap_max=max(update_gaps)if update_gaps else None,
            classified_person_frame_fraction=classified_frames/max(person_frames,1),
            ready_person_frames=ready_person_frames,
            ready_person_classified_fraction=sum(f.get('classified_ready',0)for f in frames)/max(ready_person_frames,1),
            mean_candidate_people=float(np.mean([f['people']for f in frames])),
            max_candidate_people=max(f['people']for f in frames),clip_boundary_resets=result['clip_boundary_resets'])
        assert row['frames']==row['unique_source_indices']
        assert row['frames']+row['source_dropped']==row['source_frames']
        assert row['submitted']==row['completed']+row['coalesced']
        summaries.append(row)
    return summaries


def main():
    base = read(ROOT/'reports/person-performance-20261023/summary.json')
    summaries = []
    for name,folder in [('thread',OUT/'runs'),('process',OUT/'process-runs')]:
        if not (folder/'complete.json').exists():
            continue
        runs = read(folder/'results.json')
        for cards in sorted({r['cards']for r in runs}):
            for sample in base:
                rows = [r for r in runs if r['cards']==cards and r['name']==sample['name']]
                if not rows:
                    continue
                fps = [r['processing_fps']for r in rows]
                calls = [call for r in rows for w in r['action_workers']for call in w['calls']]
                summaries.append(dict(backend=name,cards=cards,name=sample['name'],repeats=len(rows),
                    baseline_fps=sample['fps_mean'],fps_mean=float(np.mean(fps)),
                    fps_sd=float(np.std(fps,ddof=1))if len(rows)>1 else None,fps_min=min(fps),fps_max=max(fps),
                    speedup=float(np.mean(fps))/sample['fps_mean'],predictions=rows[0]['completed'],
                    action_batch_p95=quantile([c['total_seconds']for c in calls],95),
                    action_batch_max=max(c['total_seconds']for c in calls),
                    uploaded_frames=sum(w['stats'].get('uploaded_frames',0)for r in rows for w in r['action_workers']),
                    reused_frames=sum(w['stats'].get('reused_frames',0)for r in rows for w in r['action_workers'])))
    (OUT/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+'\n')
    if summaries:
        with (OUT/'summary.csv').open('w')as stream:
            writer=csv.DictWriter(stream,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    live = []
    for folder in (OUT/'live',OUT/'live-extra',OUT/'soak'):
        if (folder/'complete.json').exists():
            live.extend(live_summary(folder))
    (OUT/'live-summary.json').write_text(json.dumps(live,ensure_ascii=False,indent=2)+'\n')
    (OUT/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(11,4.8),constrained_layout=True)
    labels=['Drink16','Wave3','Wave28','Crowd0060','Crowd0054','Feedback181324']
    x=np.arange(len(base));width=.2
    series=[('Original 1 GPU',[b['fps_mean']for b in base])]
    for backend,cards,title in [('thread',1,'Optimized 1 GPU'),('thread',2,'Optimized 2 GPUs'),('process',3,'Optimized 3 GPUs')]:
        values=[next((r['fps_mean']for r in summaries if r['backend']==backend and r['cards']==cards and r['name']==b['name']),np.nan)for b in base]
        series.append((title,values))
    for i,(title,values)in enumerate(series):
        ax.bar(x+(i-1.5)*width,values,width,label=title)
    ax.axhline(15,color='#a73737',ls='--',label='15 fps requirement')
    ax.set(xticks=x,xticklabels=labels,ylabel='Full-output processing frames/s',title='Same inputs and models; no frames or action windows dropped')
    ax.legend(fontsize=9,ncol=2)
    for ext in ('png','pdf'):fig.savefig(OUT/f'figures/throughput.{ext}',dpi=180)
    plt.close(fig)
    profile=read(OUT/'profile.json')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for mode,label in [('baseline','CPU normalization'),('uint8','uint8 upload + GPU normalization')]:
        values=[np.mean([r['total_ms']for r in profile if r['mode']==mode and r['batch']==b])for b in (1,2,4,8,16)]
        axes[0].plot([1,2,4,8,16],values,'o-',label=label)
    axes[0].set(xlabel='Person windows per batch',ylabel='RGB call wall time (ms)',title='Same-weight action microbenchmark')
    axes[0].legend(fontsize=8)
    if live:
        selected=[r for r in live if r['source_duration']<120]
        for cards in sorted({r['cards']for r in selected}):
            rows=[r for r in selected if r['cards']==cards]
            axes[1].scatter([r['effective_fps']for r in rows],[r['decision_age_max']for r in rows],label=f'{cards} GPU',s=55)
        axes[1].axvline(15,color='#a73737',ls='--');axes[1].axhline(1,color='#a73737',ls='--')
        axes[1].set(xlabel='Unique consumed source frames/s',ylabel='Maximum decision age (s)',title='Native-paced bounded-input replay')
        axes[1].legend(fontsize=8)
    for ext in ('png','pdf'):fig.savefig(OUT/f'figures/action-and-live.{ext}',dpi=180)
    plt.close(fig)
    print(json.dumps(summaries,ensure_ascii=False,indent=2))
    print('LIVE',json.dumps(live,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
