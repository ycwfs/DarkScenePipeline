"""Validate complete offline artifacts and candidate compatibility without changing release."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import subprocess
from pathlib import Path

from common import OUT,ROOT,save


def digest(path):
    h=hashlib.sha256()
    with path.open('rb')as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify_video(path):
    result=subprocess.run(['ffmpeg','-nostdin','-v','error','-xerror','-i',str(path),'-f','null','-'],capture_output=True,text=True)
    assert result.returncode==0 and not result.stderr,(path,result.stderr)
    p=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,nb_frames,duration,avg_frame_rate','-of','json',str(path)],capture_output=True,text=True,check=True)
    return dict(path=str(path.relative_to(OUT)),sha256=digest(path),decode_passed=True,stream=json.loads(p.stdout)['streams'][0])


def main():
    videos=[];summary=[]
    for scope,n in [(scope,n)for scope in ['long-videos','short-regression']for n in [1,7,9]]:
        p=OUT/scope/f'new-{n:02}'
        s=json.loads((p/'summary.json').read_text());assert s['frames']==s['output_frames']==s['source']['frames']
        for variant in ['baseline','candidate']:
            clips=s[variant]['clips']
            for key in ['clips_failed','write_errors','frames_dropped','clips_forced','clips_unfinished','clips_abandoned']:
                assert clips.get(key,0)==0,(n,variant,key,clips.get(key))
            assert len(list((p/f'{variant}-clips').rglob('*.mp4')))==clips['clips_saved']
            videos.append(p/f'{variant}.mp4');videos.extend((p/f'{variant}-clips').rglob('*.mp4'))
        a=[json.loads(l)for l in (p/'baseline-events.jsonl').read_text().splitlines()]
        b=[json.loads(l)for l in (p/'candidate-events.jsonl').read_text().splitlines()]
        assert len(a)==len(b)==s['paired_checks']
        for x,y in zip(a,b):
            for key in ['rgb_probs','skeleton_probs','fused_probs','track_id','history_generation','source_frames','frame_index','action_label','action_confidence']:
                assert x[key]==y[key],(n,key)
        summary.append(dict(scope=scope,number=n,frames=s['frames'],paired_checks=len(a)))
    with ThreadPoolExecutor(max_workers=3)as pool:checks=list(pool.map(verify_video,videos))
    for c in checks:
        if c['path'].endswith(('/baseline.mp4','/candidate.mp4')):
            scope=c['path'].split('/')[0];n=int(c['path'].split('/')[1][-2:])
            s=next(r for r in summary if r['number']==n and r['scope']==scope)
            assert int(c['stream']['nb_frames'])==s['frames'],c
    baseline=ROOT/'platform/dark-demo-v2/ckpts/yolo26m_objects_20261026.onnx'
    assert digest(baseline)=='7c9bc11befb2bc74890767fe769e0efb692f061b379473286fe8bf3b8e5848b5'
    exported=json.loads((OUT/'export-verification.json').read_text())
    assert digest(OUT/'models/yolo26m_coco80_candidate.onnx')==exported['onnx_sha256']
    assert exported['passed']
    save(OUT/'verification.json',dict(passed=True,complete_sources=summary,video_count=len(checks),videos=checks,
        baseline_model_unchanged=True,candidate_onnx_matches_verified=True,
        scope='Complete finite offline paired inference and video/clip integrity; not streaming/NFS/infinite-duration verification.'))
    print(json.dumps(dict(passed=True,videos=len(checks),sources=summary),indent=2))


if __name__=='__main__':main()
