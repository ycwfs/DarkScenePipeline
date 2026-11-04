"""Render the two test negative sources with changed decisions from existing event evidence."""
import json
import subprocess
from types import SimpleNamespace
import cv2
import numpy as np
import torch
from evaluate import ROOT, OUT, OP, save, sha, mosaic, associate_poses, DownscaleStage, RetinexformerStage


def main() -> None:
    cv2.setNumThreads(1);torch.set_num_threads(2)
    enhancer=RetinexformerStage(str(OP/'ckpts'),chunk=1);enhancer.load('cuda:0')
    protocol=json.loads((OUT/'protocol.json').read_text());records=[]
    selected={'7e9bc4c0e9a12906','02cfb195aba51c8e'}
    for v in protocol['videos']:
        if v['id']not in selected:continue
        assert not v['processed']
        d=json.loads((OUT/'data'/v['id']/'result.json').read_text())
        index={g:{}for g in (75,15)}
        for g in index:
            for e in d['events'][str(g)]:index[g].setdefault(e['frame_index'],[]).append(e)
        cap=cv2.VideoCapture(str(ROOT/v['source']));down=DownscaleStage(720);last={75:{},15:{}}
        vp=OUT/'videos'/f'{v["id"]}.mp4'
        encoder=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1440x488',
            '-r',str(v['fps']),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-threads','2',
            '-movflags','+faststart',str(vp)],stdin=subprocess.PIPE)
        n=0
        try:
            for f in d['frames']:
                ok,raw=cap.read();assert ok
                image=enhancer(down([raw]))[0];tracks={}
                for g in (75,15):
                    for e in index[g].get(n,[]):last[g][e['track_id']]=e
                    tracks[g]=[]
                    for t in f['tracks'][str(g)]:
                        e=last[g].get(t['track_id']);box=np.asarray(t['bbox'])
                        j=next(j for j,b in enumerate(f['boxes'])if np.max(np.abs(box-b[:4]))<1e-6)
                        po=f['poses'][j]if f['luma']>=g else None
                        tracks[g].append(SimpleNamespace(tid=t['track_id'],box=box,pose=None if po is None else np.array(po),
                            prediction=e,last_inference=e['timestamp']if e else -1e9))
                shown=mosaic(image,tracks,f['timestamp'],f['luma'],v['source']);encoder.stdin.write(shown.tobytes())
                if n==round(v['fps']*1.6):cv2.imwrite(str(OUT/'posters'/f'{v["id"]}.jpg'),shown)
                n+=1
            assert not cap.read()[0]
        finally:
            cap.release();encoder.stdin.close()
            assert encoder.wait(timeout=60)==0
        assert n==v['frames'];records.append(v)
    enhancer.close();save(OUT/'extra-demos.json',dict(videos=records,method='Exact cached detections and event vectors; original frames enhanced again with same fixed checkpoint; post-result error examples, not a new accuracy test.'))
    print('Rendered',len(records),'test negative comparisons',flush=True)


if __name__=='__main__':main()
