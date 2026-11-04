"""Wave-specific diagnostics: retained and rejected full-video triggers, all included."""
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
from common import OUT, ROOT, save, digest


def run(v,gpu):
    cmd=['docker','run','--rm','--network','none','--gpus',f'device={gpu}','--ipc=host',
        '-v',f'{ROOT}:{ROOT}','-v',f'{ROOT}/platform/dark-demo-v2:/app:ro',
        '--entrypoint','/opt/conda/envs/darkpipe/bin/python','darkpipe-operator:0.3.0',
        str(OUT/'scripts/paired_video.py'),'--number',str(v['number']),'--wave-short',v['id']]
    with (OUT/f'wave-short-{v["id"]}.log').open('w')as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    print('FINISHED',v['id'],flush=True)


def main():
    sources=json.loads((ROOT/'reports/manypeople-inference-20261025/input-manifest.json').read_text())
    rows=[]
    for n,start,tag in [(1,18,'child-retained'),(1,261,'wave-veto'),(7,90,'standing-veto'),(9,133,'raised-retained')]:
        source=next(v for v in sources if v['number']==n)
        sid=f'new-{n:02}-{tag}';path=OUT/'inputs'/f'{sid}.mkv';path.parent.mkdir(exist_ok=True)
        assert not path.exists()
        subprocess.run(['ffmpeg','-nostdin','-v','error','-ss',str(start),'-i',source['absolute_path'],
            '-t','12','-an','-c:v','ffv1','-threads','2',str(path)],check=True)
        probe=subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,avg_frame_rate,codec_name','-of','json',str(path)],capture_output=True,text=True,check=True)
        stream=json.loads(probe.stdout)['streams'][0]
        numerator,denominator=map(int,stream['avg_frame_rate'].split('/'));fps=numerator/denominator
        assert int(stream['nb_read_frames'])==round(12*fps)
        rows.append(dict(source,id=sid,absolute_path=str(path),source_offset=start,duration=12.,fps=fps,
            frames=int(stream['nb_read_frames']),sha256=digest(path),original_sha256=source['sha256'],
            original_bytes=source['bytes'],bytes=path.stat().st_size,codec=stream['codec_name']))
    save(OUT/'wave-short-inputs.json',dict(basis='Post-baseline diagnostic excerpts around retained AND rejected wave triggers; not independent test or threshold selection.',sources=rows))
    def shard(items,gpu):
        for v in items:run(v,gpu)
    with ThreadPoolExecutor(max_workers=3)as pool:
        jobs=[pool.submit(shard,rows[i::3],i+3)for i in range(3)]
        for job in jobs:job.result()
    save(OUT/'wave-short-runs-complete.json',dict(passed=True,sources=[v['id']for v in rows]))


if __name__=='__main__':main()
