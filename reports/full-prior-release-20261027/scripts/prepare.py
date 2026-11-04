"""Freeze real demo inputs and a five-minute network source before inference."""
from pathlib import Path
import hashlib,json,shutil,subprocess
ROOT=Path(__file__).resolve().parents[3]; OUT=Path(__file__).resolve().parents[1]
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(cmd):subprocess.run(cmd,check=True)
(OUT/'inputs').mkdir(exist_ok=True);(OUT/'transport').mkdir(exist_ok=True)
old=json.loads((ROOT/'reports/person-release-20261024/demo-selection.json').read_text())
selected=[]
for n,r in enumerate(old['records']):
 selected.append(dict(id=f'lighting-{n+1:02d}',title=Path(r['source']).stem,input=r['source'],source=r['source'],group='old',processed=r['processed'],seconds=r['seconds'],historical_luma=r['raw_mean_luma'],historical_people_mode=r['people_mode'],band=r['band'],selection=r['selection'],input_sha256=r['sha256']))
sources=json.loads((ROOT/'reports/four-parameter-tuning-20261026/protocol.json').read_text())['sources'];byid={r['id']:r for r in sources};bypath={r['source']:r for r in sources}
for name in ['挥手22','挥手32','追逐20','喝水27']:
 source='real-actiondata-all/'+name+'.mp4';r=bypath[source]
 selected.append(dict(id='rule-'+name,title=name,input=source,source=source,group='old',processed=False,seconds=r['seconds'],historical_luma=r['raw_mean_luma'],historical_people_mode=r['people_mode'],selection='Previously reviewed examples exercising bright priors; no outcome filtering in this release',input_sha256=sha(ROOT/source)))
segments=[]
for name,sid,start,seconds in [('child','new-01',10,30),('walking','new-01',185,30),('phone','new-07',10,30),('camera','new-09',94,30)]:
 r=byid[sid];source=Path(r['source']);dest=OUT/'inputs'/f'{name}.mkv'
 if not dest.exists():run(['ffmpeg','-nostdin','-v','error','-ss',str(start),'-i',str(source),'-t',str(seconds),'-an','-c:v','ffv1','-threads','2',str(dest)])
 selected.append(dict(id='crowd-'+name,title=f'{sid} {start}–{start+seconds}s',input=str(dest.relative_to(ROOT)),source=str(source.relative_to(ROOT)),group='new',processed=False,start=start,seconds=seconds,selection='Prior-reviewed child running / adult walking / phone / camera context, fixed before this release inference',input_sha256=sha(dest)))
 segments.append(dest)
save(OUT/'inputs.json',selected);save(OUT/'coverage-limitations.json',old['missing'])
# Encode all transport parts identically; never modify image brightness or insert markers.
parts=[(ROOT/'reports/person-object-release-20261026/input-multiaction-5min.mp4',180.)]+[(p,30.)for p in segments]
encoded=[];manifest=[];offset=0
for i,(source,seconds) in enumerate(parts):
 dest=OUT/'transport'/f'part-{i}.mp4'
 if not dest.exists():run(['ffmpeg','-nostdin','-v','error','-i',str(source),'-t',str(seconds),'-vf','scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=25,setsar=1','-an','-c:v','libx264','-preset','veryfast','-crf','18','-threads','2','-pix_fmt','yuv420p','-g','25','-bf','0','-video_track_timescale','12800',str(dest)])
 encoded.append(dest);manifest.append(dict(source=str(source.relative_to(ROOT)),start=offset,seconds=seconds,sha256=sha(source)));offset+=seconds
playlist=OUT/'transport/parts.txt';playlist.write_text(''.join("file '"+p.name+"'\n"for p in encoded))
final=OUT/'input-live-300s.mp4'
if not final.exists():run(['ffmpeg','-nostdin','-v','error','-f','concat','-safe','0','-i',str(playlist),'-c','copy','-movflags','+faststart',str(final)])
for name,seconds in [('preroll',75),('tail',30)]:
 dest=OUT/'transport'/f'{name}.mp4'
 if not dest.exists():run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i',f'color=black:s=1920x1080:r=25:d={seconds}','-an','-c:v','libx264','-preset','veryfast','-crf','18','-threads','2','-pix_fmt','yuv420p','-g','25','-bf','0','-video_track_timescale','12800',str(dest)])
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,r_frame_rate:format=duration','-of','json',str(final)]));assert int(probe['streams'][0]['nb_read_frames'])==7500
save(OUT/'input-manifest.json',dict(input=final.name,frames=7500,seconds=300.,fps=25,sha256=sha(final),segments=manifest,selection='Existing fixed raw-scene montage first180s + four prior-reviewed crowd excerpts, no outcome selection this release',caveat='Hard scene cuts, continuous tracking without oracle resets; historical sources, not independent accuracy benchmark',markers=False))
print('PREPARED',len(selected),'demos and300s stream')
