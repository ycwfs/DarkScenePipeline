"""Persist manually reviewed intervals; no model outputs are used as labels."""
import json
from pathlib import Path

OUT=Path(__file__).resolve().parents[1]
data=json.loads((OUT/'annotations.json').read_text())
rows=[r for page in json.loads((OUT/'review/wave-index.json').read_text()) for r in page['rows']]
# Explicit intervals from the numbered temporal sheets reviewed by the assistant.
actions={
('d0ab8993',1):[(.8,3.4)],('c99b19d7',1):[(.3,4.7)],
('97c76cf3',1):[(.8,4.)],('fc9a731d',1):[(.35,1.6)],('fc9a731d',3):[(1.7,5.1)],
('6849b046',1):[(.2,1.52)],('6849b046',2):[(1.52,3.1)],('9e9349dd',1):[(.1,2.1)],
('541cb241',1):[(.3,2.)],('5014b0b7',1):[(.9,4.2)],('fcde48c0',1):[(.1,1.95)],
('7c2f7e8b',1):[(0,1.3)],('6e54fd92',1):[(.15,2.1)],('c76c09cc',1):[(.8,2.6)],
('7ebba3f1',1):[(.4,2.3)],('f019d7af',1):[(.55,2.2)],('56f605e0',1):[(.3,2.25)],
('92b9a428',1):[(.35,2.15)],('b576f320',2):[(.3,5.15)],('1fa7b05d',1):[(.3,2.2)],
('d460cd44',1):[(.1,1.4)],('e46d679b',1):[(.95,2.5)],('220cc822',1):[(.4,1.95)],
('6c195037',2):[(1.,2.3)],('684c5ecb',1):[(1.1,6.1)],('1a5e5b4e',1):[(.2,2.)],
('530259c7',1):[(.3,2.2)],('0483b0eb',1):[(.4,1.7)],('922a3b6d',1):[(.4,1.8)],
('911066db',1):[(.4,1.65)],('6f2be3a3',1):[(.35,2.)],('6e5e1844',1):[(.35,1.95)],
('40fd7c59',1):[(.2,1.85)],('b9755309',1):[(.25,1.5)],('ba0ba142',1):[(.25,2.15)],
('c938ebc5',1):[(1.,4.9)],('e97748cc',1):[(.3,1.6)],('df7aa782',1):[(.4,2.15)],
('015b1467',1):[(.45,1.95)],('bbbaa4e6',1):[(.9,4.)],('344e4ea8',1):[(0,5.7)],
('cc074151',1):[(.5,2.5)],('cdb9e33c',1):[(.4,2.35)],
('6d3442ab',1):[(1.7,4.1),(4.85,6.1),(6.85,8.1)],
('6d3442ab',2):[(8.,9.2),(11.2,13.6),(15.7,16.7),(18.4,20.4),(21.6,22.4),(23.5,24.2)],
('6d3442ab',4):[(27.35,28.5)]}
others={('d0ab8993',2),('5014b0b7',4),('56f605e0',2),('56f605e0',3),
('92b9a428',2),('b576f320',1),('d460cd44',3),('684c5ecb',2),('6c195037',1)}
uncertain={('1e8fc538',2):'Static dark detection; cannot reliably distinguish fixture/person.',
('1e8fc538',6):'Static dark detection; cannot reliably distinguish fixture/person.',
('92ea11e6',5):'Static dark detection; actual actor elsewhere; uncertain entity.',
('5014b0b7',2):'Track changes from background bystander to foreground actor; mixed identity.',
('1fa7b05d',2):'Distant bystander hand near face; action unclear, pending user confirmation.',
('d460cd44',2):'Distant bystander hand near face; action unclear, pending user confirmation.'}
keys={(r['video_id'],r['track_id'])for r in rows}
data['records']=[r for r in data['records']if(r['video_id'],r['track_id'])not in keys]
for row in rows:
 key=(row['video_id'][:8],row['track_id']); duration=row['duration']
 manifest=json.loads((OUT/'data'/row['video_id']/'manifest.json').read_text())
 samples=manifest['review'][str(row['track_id'])]
 start=samples[0]['timestamp']; end=min(duration,samples[-1]['timestamp']+.24)
 r={k:row[k]for k in ['video_id','track_id','source']}
 r.update(reviewer='assistant visual review',review_evidence='review/wave-index.json',
          status='reviewed',segments=[],note='Numbered track temporal sheets; uncertain transition margins remain unlabeled.')
 if key in actions:
  spans=actions[key]
  r['segments']=[dict(start=max(start,a),end=min(end,b),label='wave')for a,b in spans if min(end,b)>max(start,a)]
  # Only visible neutral walking/standing intervals, with conservative boundary gaps.
  prev=start
  for a,b in spans:
   if a-.25>prev:r['segments'].append(dict(start=prev,end=a-.25,label='other'))
   prev=b+.25
  if end>prev:r['segments'].append(dict(start=prev,end=end,label='other'))
 elif key in others:
  r['segments']=[dict(start=start,end=end,label='other')]
  r['note']='Visible inactive bystander, target distinct from the waving person.'
  if key==('6c195037',1):
   r.update(entity_type='detector_false_positive',note='Numbered full-scene review confirms T01 is an office chair; actual actor is T02.')
 elif key in uncertain:
  r.update(status='uncertain',note=uncertain[key])
 else:raise ValueError(f'Missing explicit visual annotation: {key}')
 r['segments'].sort(key=lambda s:s['start'])
 data['records'].append(r)
# One historically ambiguous negative reviewed separately, before inference.
m=json.loads(next((OUT/'data').glob('ab4913f4*/manifest.json')).read_text())
vid=m['video']['id']; tids=sorted({w['track_id']for w in m['windows']})
data['records']=[r for r in data['records']if r['video_id']!=vid]
for tid in tids:
 r=dict(video_id=vid,track_id=tid,source=m['video']['source'],reviewer='assistant visual review',
        review_evidence='review/negative-0104-0.jpg and negative-0104-1.jpg',status='reviewed',
        segments=[dict(start=0,end=m['video']['seconds'],label='other')],
        note='Standing, walking, holding phone, or stretching; no clearly visible named target action.')
 if tid==3:r.update(status='uncertain',segments=[],note='Hand/object stays near mouth; possible drink; pending user confirmation.')
 if tid==11:r.update(entity_type='detector_false_positive',note='Static service robot misdetected as human.')
 data['records'].append(r)
(OUT/'annotations.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
print('Saved',len(data['records']),'track annotations')
