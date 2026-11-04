import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; OUT=Path(__file__).resolve().parents[1]
sources=json.loads((ROOT/'reports/four-parameter-tuning-20261026/protocol.json').read_text())['sources']
lookup={s['source']:s for s in sources}; byid={s['id']:s for s in sources}
selected=[]
for n,name in enumerate(['喝水','喝水16','喝水17','捡东西14','捡东西12','挥手3','喝水4','other/other123','喝水27'],1):
 s=lookup['real-actiondata-all/'+name+'.mp4']
 selected.append(dict(id=f'old-{n:02d}',title=name,source=s['source'],input=s['source'],source_id=s['id'],
   split=s['split'],group='old',start=0,seconds=s['seconds'],historical_people_mode=s['people_mode']))
for sid,start,seconds in [('new-01',140,25),('new-07',10,30),('new-09',94,30)]:
 s=byid[sid]; source=Path(s['source']); dest=OUT/'inputs'/f'{sid}.mkv'
 if not dest.exists():
  subprocess.run(['ffmpeg','-v','error','-nostdin','-ss',str(start),'-i',str(source),'-t',str(seconds),
   '-map','0:v:0','-an','-c:v','ffv1','-threads','2',str(dest)],check=True)
 selected.append(dict(id=sid,title=f'{sid} {start}–{start+seconds}s',source=str(source.relative_to(ROOT)),
  input=str(dest.relative_to(ROOT)),source_id=sid,split=s['split'],group='new',start=start,seconds=seconds))
for s in selected:
 p=ROOT/s['input']; h=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 s['input_sha256']=h.hexdigest()
(OUT/'inputs.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2)+'\n')
print([(s['id'],s['title'],s['seconds'])for s in selected])
