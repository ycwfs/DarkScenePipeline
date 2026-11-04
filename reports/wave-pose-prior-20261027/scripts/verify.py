"""Full decoding, exact paired upstream checks and wave-only decision changes."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
from common import OUT, ROOT, save, digest


def read_rows(path):return [json.loads(line)for line in path.read_text().splitlines()]


def decode(path):
    result=subprocess.run(['ffmpeg','-nostdin','-v','error','-xerror','-i',str(path),'-f','null','-'],capture_output=True,text=True)
    assert result.returncode==0 and not result.stderr,(path,result.stderr)
    result=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,nb_frames,duration','-of','json',str(path)],capture_output=True,text=True,check=True)
    return dict(path=str(path.relative_to(OUT)),sha256=digest(path),stream=json.loads(result.stdout)['streams'][0],decode_passed=True)


def main():
    assert all((OUT/name).exists()for name in ['video-runs-complete.json','old-video-runs-complete.json','wave-short-runs-complete.json'])
    sources=[];videos=[];checks=Counter()
    for scope in ['long-videos','short-regression','wave-shorts','old-videos']:
        for directory in sorted((OUT/scope).glob('*')):
            if not directory.is_dir():continue
            s=json.loads((directory/'summary.json').read_text())
            assert s['frames']==s['source']['frames']==s['output_frames']
            a=read_rows(directory/'baseline-events.jsonl');b=read_rows(directory/'candidate-events.jsonl')
            assert len(a)==len(b)==s['paired_checks']
            for x,y in zip(a,b):
                for key in ['rgb_probs','skeleton_probs','fused_probs','track_id','history_generation','source_frames','frame_index','bbox','action_label','action_confidence','object_prior']:
                    assert x[key]==y[key],(directory,key)
                if x['label']!='wave':assert x['label']==y['label']
                else:
                    assert y['wave_sweep']['3_total']==y['label']
                    e=y['wave_prior'];assert len({r['frame']for r in e['samples']})==len(e['samples'])
                    if e['applied']:
                        assert (y['label']=='wave')==(max(e['counts'])>=3)
                        assert e['counts']==[sum(r['above_both'][j]for r in e['samples'])for j in range(2)]
                    else:assert y['label']=='wave'
                    checks['wave_windows']+=1
                    checks['wave_vetoes']+=int(y['label']=='other')
                checks['paired_events']+=1
            for variant in ['baseline','candidate']:
                clips=s[variant]['clips']
                for key in ['clips_failed','write_errors','frames_dropped','clips_forced','clips_unfinished','clips_abandoned']:
                    assert clips.get(key,0)==0,(directory,variant,key,clips.get(key))
                files=list((directory/f'{variant}-clips').rglob('*.mp4'))
                assert len(files)==clips['clips_saved']
                videos.extend(files);videos.append(directory/f'{variant}.mp4')
            sources.append(dict(directory=str(directory.relative_to(OUT)),frames=s['frames'],paired_checks=len(a)))
    integrated=json.loads((OUT/'integrated-candidate-check.json').read_text());assert integrated['passed']
    videos.extend((OUT/'integrated-candidate').rglob('*.mp4'))
    bright=json.loads((OUT/'integrated-bright-candidate-check.json').read_text())
    assert bright['passed'] and bright['wave_vetoes']==2
    videos.extend((OUT/'integrated-bright-candidate').rglob('*.mp4'))
    with ThreadPoolExecutor(max_workers=3)as pool:decoded=list(pool.map(decode,videos))
    lookup={s['directory']:s for s in sources}
    for v in decoded:
        path=OUT/v['path']
        if path.name in ['baseline.mp4','candidate.mp4']:
            assert int(v['stream']['nb_frames'])==lookup[str(path.parent.relative_to(OUT))]['frames']
        assert v['stream']['codec_name']=='h264'
    save(OUT/'verification.json',dict(passed=True,source_count=len(sources),sources=sources,checks=dict(checks),
        videos=decoded,video_count=len(decoded),integrated_candidate_passed=True,
        integrated_bright_candidate_passed=True,
        independent_bright_probability_reproducibility_within_2e_4=bright['probability_reproducibility_within_2e_4'],
        scope='Finite offline paired evidence and complete decode; no new streaming, NFS or infinite-duration claim.'))
    print(dict(passed=True,sources=len(sources),videos=len(decoded),checks=dict(checks)))


if __name__=='__main__':main()
