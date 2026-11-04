"""Add pre-existing wave veto examples for functional path coverage, not selection."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parents[1]
p=OUT/'inputs.json';rows=json.loads(p.read_text())
selected={'new-01-wave-veto','new-07-standing-veto'}
rows=[r for r in rows if r['id']not in selected]
(OUT/'initial-demo-selection.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
for s in json.loads((ROOT/'reports/wave-pose-prior-20261027/wave-short-inputs.json').read_text())['sources']:
 if s['id']not in selected:continue
 rows.append(dict(id=s['id'],title=s['id'],input=str(Path(s['absolute_path']).relative_to(ROOT)),source='data-10-25-manypeople/'+s['path'],group='new',processed=False,start=s['source_offset'],seconds=s['duration'],input_sha256=s['sha256'],selection='Added after initial21 demonstrations to exercise wave-veto path absent from those21; fixed historical diagnostic excerpt, not independent test'))
p.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
