import argparse
import json
import subprocess
import sys
from common import OUT,ROOT,WAVE,save,digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');a=parser.parse_args()
    if not a.run:
        source=json.loads((WAVE/'long-videos/new-01/summary.json').read_text())['source']
        selected=[('child-away',18,10),('adult-walk',190,12),('overlap-walk',225,12),('child-across',294,12),('slow-walk',348,10)]
        inputs=[]
        for name,start,seconds in selected:
            path=OUT/'inputs'/f'{name}.mkv';path.parent.mkdir(exist_ok=True)
            subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-ss',str(start),'-i',source['absolute_path'],'-t',str(seconds),'-an','-c:v','ffv1','-threads','2',str(path)],check=True)
            inputs.append(dict(id=name,absolute_path=str(path),start=start,seconds=seconds,frames=seconds*25,processed=False,
                sha256=digest(path),original=source['absolute_path'],scope='Feedback short excerpts; overlap full-video episodes, restarted tracking'))
        save(OUT/'short-inputs.json',inputs)
        old=json.loads((WAVE/'protocol.json').read_text())['old_sources']
        wanted=['追逐38.mp4','追逐40.mp4','追逐24.mp4','追逐30.mp4','other186.mp4']
        old=[v for v in old if any(v['source'].endswith('/'+name)for name in wanted)]
        save(OUT/'short-old-selection.json',old)
        print('Prepared',len(inputs),'new excerpts and',len(old),'old sources')
    else:
        for v in json.loads((OUT/'short-inputs.json').read_text()):
            subprocess.run([sys.executable,str(OUT/'scripts/run_candidate.py'),'--excerpt',v['id']],check=True)
        for v in json.loads((OUT/'short-old-selection.json').read_text()):
            subprocess.run([sys.executable,str(OUT/'scripts/run_candidate.py'),'--source-id',v['id']],check=True)

if __name__=='__main__':main()
