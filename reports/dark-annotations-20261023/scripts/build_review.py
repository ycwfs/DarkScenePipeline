"""Make offline/file:// review assets, contextual MP4s and coverage evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scan import OUT, ROOT, downscale, save

EDGES=[(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
PALETTE=[(100,235,60),(255,200,40),(200,120,255),(60,150,255),(230,220,70),(150,70,245)]
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def gamma(image: np.ndarray) -> np.ndarray:
    return cv2.LUT(image,np.rint((np.arange(256)/255.)**.45*255).astype(np.uint8))


def overlay(image: np.ndarray, frame: dict, video: dict) -> np.ndarray:
    out=image.copy();scale=np.array(out.shape[1::-1])/np.array([video['width'],video['height']])
    for p in frame['people']:
        color=PALETTE[(int(p['person_id'][1:])-1)%len(PALETTE)]
        x1,y1,x2,y2=np.rint(np.array(p['bbox_xyxy'])*np.tile(scale,2)).astype(int)
        cv2.rectangle(out,(x1,y1),(x2,y2),color,2)
        cv2.putText(out,p['person_id']+' ?',(x1,max(18,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1,cv2.LINE_AA)
        points=np.array(p['keypoints']);points[:,:2]*=scale
        valid=p['keypoint_usable']
        for a,b in EDGES:
            if valid[a] and valid[b]:
                cv2.line(out,tuple(np.rint(points[a,:2]).astype(int)),tuple(np.rint(points[b,:2]).astype(int)),color,2,cv2.LINE_AA)
        for j,point in enumerate(points):
            if valid[j]:
                cv2.circle(out,tuple(np.rint(point[:2]).astype(int)),2,color,-1,cv2.LINE_AA)
    return out


def build_one(path: Path, make_video=True) -> dict:
    data=json.loads(path.read_text());video=data['video'];frames=data['frames']
    directory=path.parent
    for frame in frames:
        gamma_path=directory/'gamma'/f'{frame["frame"]:06d}.jpg'
        gamma_path.parent.mkdir(exist_ok=True)
        if not gamma_path.exists():
            raw=cv2.imread(str(OUT/frame['image']))
            if not cv2.imwrite(str(gamma_path),gamma(raw),[cv2.IMWRITE_JPEG_QUALITY,92]):
                raise IOError(gamma_path)
        frame['gamma_image']=gamma_path.relative_to(OUT).as_posix()
    (directory/'review.js').write_text('window.receiveAnnotation('+json.dumps(data,ensure_ascii=False,separators=(',',':'))+');\n')
    name=Path(video['source']).stem+'_'+video['id'][:8]
    dest=OUT/'videos'/f'{name}_review.mp4';dest.parent.mkdir(exist_ok=True)
    check_path=directory/'video-check.json'
    if make_video and not check_path.exists():
        annotated={f['frame']:f for f in frames}
        scanned=json.loads((directory/'scan.json').read_text())['frames']
        cap=cv2.VideoCapture(str(ROOT/video['source']))
        writer=None;count=0
        pending=dest.with_suffix('.pending.mp4')
        try:
            while True:
                ok,raw=cap.read()
                if not ok:
                    break
                left=downscale(raw,720)
                record=annotated.get(count)
                if record:
                    right=overlay(gamma(left),record,video)
                    note=f'{len(record["people"])} proposals; REVIEW REQUIRED' if record['people'] else 'UNRESOLVED - NOT A NEGATIVE'
                else:
                    right=left.copy();note='L >= 75: CONTEXT ONLY / NOT ANNOTATED'
                top=np.full((32,left.shape[1]*2,3),(26,31,32),np.uint8)
                bottom=top.copy()
                cv2.putText(top,'ORIGINAL',(10,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(235,235,235),1,cv2.LINE_AA)
                cv2.putText(top,'GAMMA + BOX / COCO17 DRAFT',(left.shape[1]+10,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(235,235,235),1,cv2.LINE_AA)
                f=scanned[count]
                text=f'{video["id"][:8]}  f{count:06d}  t={f["timestamp"]:.3f}s  L={f["luma"]:.3f} | {note}'
                cv2.putText(bottom,text,(10,22),cv2.FONT_HERSHEY_SIMPLEX,.53,(235,235,235),1,cv2.LINE_AA)
                canvas=np.concatenate([top,np.concatenate([left,right],axis=1),bottom],axis=0)
                if writer is None:
                    command=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24',
                        '-s',f'{canvas.shape[1]}x{canvas.shape[0]}','-r',str(video['fps']),'-i','pipe:0',
                        '-an','-c:v','libx264','-preset','veryfast','-crf','22','-pix_fmt','yuv420p',
                        '-threads','2','-movflags','+faststart',str(pending)]
                    writer=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
                writer.stdin.write(canvas.tobytes());count+=1
            if writer:
                writer.stdin.close();stderr=writer.stderr.read();rc=writer.wait()
                if rc:
                    raise RuntimeError(stderr.decode(errors='replace'))
                assert count==video['frames'],(video['source'],count,video['frames'])
                pending.replace(dest)
            else:
                raise ValueError(f'No decoded frames in {video["source"]}')
        finally:
            cap.release()
            if writer and writer.poll() is None:
                writer.terminate();writer.wait()
        # Decode the actual deliverable fully, independently of the writer's frame count.
        verify=cv2.VideoCapture(str(dest));decoded=0
        while True:
            ok,image=verify.read()
            if not ok:break
            decoded+=1
        verify.release()
        assert decoded==count,(str(dest),decoded,count)
        save(check_path,dict(path=dest.relative_to(OUT).as_posix(),frames=decoded,fps=video['fps'],
            complete_decode=True,source_frames=video['frames'],dark_annotated_frames=len(frames),
            note='All decoded source frames retained; review MP4 uses nominal constant FPS. AVI source PTS is shown separately.'))
    chosen=sorted(set(np.linspace(0,len(frames)-1,min(6,len(frames)),dtype=int).tolist()))
    contact=Image.new('RGB',(960, (len(chosen)+1)//2*294),(244,243,238))
    font=ImageFont.truetype(FONT,14)
    draw=ImageDraw.Draw(contact)
    for slot,index in enumerate(chosen):
        f=frames[index];raw=cv2.imread(str(OUT/f['image']));shown=overlay(gamma(raw),f,video)
        im=Image.fromarray(cv2.cvtColor(shown,cv2.COLOR_BGR2RGB));im.thumbnail((480,270))
        x=slot%2*480;y=slot//2*294
        contact.paste(im,(x,y+24));draw.text((x+6,y+3),f'{Path(video["source"]).name}  f{f["frame"]}  L={f["luma"]:.1f}',font=font,fill=(25,44,48))
    contacts=OUT/'contacts';contacts.mkdir(exist_ok=True)
    contact.save(contacts/f'{name}.jpg',quality=90)
    return dict(id=video['id'],source=video['source'],sha256=video['sha256'],aliases=video['aliases'],
        dark_frames=len(frames),source_frames=video['frames'],
        no_box_frames=sum(not f['people'] for f in frames),
        risk_frames=sum(bool(f['risks']) for f in frames),
        boxes=sum(len(f['people']) for f in frames),
        luma_min=min(f['luma'] for f in frames),luma_max=max(f['luma'] for f in frames),
        split=video['frozen_split']['split'],video=dest.relative_to(OUT).as_posix(),
        contact=(contacts/f'{name}.jpg').relative_to(OUT).as_posix())


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial',action='store_true')
    parser.add_argument('--no-video',action='store_true')
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();cv2.setNumThreads(1)
    paths=sorted((OUT/'data').glob('*/annotations.json'))
    scan=json.loads((OUT/'scan-summary.json').read_text())
    if not args.partial:
        assert len(paths)==scan['dark_videos'],(len(paths),scan['dark_videos'])
    records=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index,result in enumerate(pool.map(lambda p:build_one(p,not args.no_video),paths),1):
            records.append(result)
            print(f'[review] {index}/{len(paths)} {result["source"]} dark={result["dark_frames"]} missing={result["no_box_frames"]}',flush=True)
    # Start with the two visually anchored examples, then uncertain and other videos.
    records.sort(key=lambda v:(v['id'] not in {'1d4d072b22d10fe1','4078760fdadd05f6'},v['source']))
    summary=dict(scan,annotated_videos=len(records),annotated_frames=sum(v['dark_frames'] for v in records),
        no_box_frames=sum(v['no_box_frames'] for v in records),proposed_person_boxes=sum(v['boxes'] for v in records),
        risk_frames=sum(v['risk_frames'] for v in records),user_confirmed_boxes=2,user_confirmed_poses=0,
        training_frames=0,complete=not args.partial)
    if not args.partial:
        assert summary['annotated_frames']==summary['dark_frames']
    catalog=dict(dataset='dark-real-20261023-v1',summary=summary,videos=records)
    save(OUT/'catalog.json',catalog)
    (OUT/'catalog.js').write_text('const CATALOG='+json.dumps(catalog,ensure_ascii=False,separators=(',',':'))+';\n')
    save(OUT/'summary.json',summary)
    with (OUT/'review-index.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['id','source','dark_frames','source_frames','no_box_frames','risk_frames','boxes','luma_min','luma_max','split','video','contact'])
        writer.writeheader();writer.writerows({k:v for k,v in row.items() if k in writer.fieldnames} for row in records)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
