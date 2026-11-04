"""Run the independent three-GPU candidate on RTSP input with real network outputs."""
import argparse
import json
from pathlib import Path
import signal
import sys
import threading
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'reports/person-optimization-20261023/scripts'))
from runtime import PersonRuntime, OutputWorker
from network_capture import NetworkCapture
from person_clips import PersonClips
from darkpipe.server import LatestSlot
from darkpipe.streams import rtmp_push


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--push',action='append',default=[])
    p.add_argument('--seconds',type=float,default=240)
    p.add_argument('--cards',type=int,default=3)
    p.add_argument('--processed',action='store_true')
    p.add_argument('--clip-denoise',choices=['off','fast','quality'],default='off')
    p.add_argument('--audit-markers',action='store_true',help='Encode a source-frame counter in the output top edge for receiver latency checks')
    p.add_argument('--warmup-video',default='real-actiondata-all/挥手3.mp4')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);cv2.setNumThreads(2)
    stop=threading.Event()
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda *_:stop.set())
    runtime=PersonRuntime([f'cuda:{i}'for i in range(args.cards)],coalesce=True,
                          process_workers=args.cards>=3)
    capture=clips=None
    outputs=[]; telemetry=[]
    monitor_stop=threading.Event()
    eventlog=None
    error=None
    try:
        for i in range(2):
            runtime.run(dict(name='warmup',source=args.warmup_video,processed=False),args.output/f'warmup-{i}')
        capture=NetworkCapture(args.input,seconds=args.seconds,processed=args.processed,stop=stop)
        clips=PersonClips(args.output/'clips',capture.origin,denoise=args.clip_denoise)
        slot=LatestSlot()
        display_shape=[808,1440,3]
        for url in args.push:
            outputs.append(rtmp_push(slot,25,url,bitrate='4M'))
        eventlog=(args.output/'events.jsonl').open('w',buffering=1)
        def on_event(row):
            row=dict(row,published=time.monotonic())
            eventlog.write(json.dumps(row,ensure_ascii=False)+'\n')
            clips.event(row)
        def on_frame(image,record):
            display_shape[:]=image.shape
            if args.audit_markers:
                image=image.copy()
                for bit in range(24):
                    image[:16,bit*8:(bit+1)*8]=255 if (record['frame'] >> bit)&1 else 0
            clips.frame(image,record)
            ok,jpg=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,85])
            if not ok:raise RuntimeError('JPEG encode failed')
            slot.put(jpg.tobytes())
        def monitor():
            while not monitor_stop.wait(1):
                now=time.monotonic()-capture.origin
                stale=capture.last_frame_at is None or now-capture.last_frame_at>2.
                if stale:
                    # Never keep presenting a stale action as a live result during an outage.
                    warning=np.zeros(tuple(display_shape),np.uint8)
                    warning[:16,:192]=255  # reserved audit counter: no current source frame
                    cv2.putText(warning,'INPUT LOST - RECONNECTING',(30,80),cv2.FONT_HERSHEY_SIMPLEX,
                                1.0,(0,180,255),2,cv2.LINE_AA)
                    ok,jpg=cv2.imencode('.jpg',warning)
                    if ok:slot.put(jpg.tobytes())
                row=dict(time=now,input_frames=capture.total,input_stale=stale,
                    input_last=capture.last_frame_at,processed_frames=capture.count,
                    outputs=[dict(alive=o.alive(),restarts=o.restarts,snap=o.snap())for o in outputs],
                    clips=clips.stats())
                row['clips'].pop('clips',None)
                telemetry.append(row)
                (args.output/'health.json').write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
        monitor_thread=threading.Thread(target=monitor,daemon=True);monitor_thread.start()
        (args.output/'ready.json').write_text(json.dumps(dict(origin=capture.origin,input=args.input,
            output_urls=args.push,seconds=args.seconds,cards=args.cards))+'\n')
        runtime.run(dict(name='network',source=args.input,processed=args.processed),args.output/'network',paced=True,
            source_factory=lambda *_:capture,
            sink_factory=lambda path,fps,origin,paced:OutputWorker(path,fps,origin,paced,on_frame=on_frame),
            event_callback=on_event)
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        stop.set()
        monitor_stop.set()
        if capture:capture.close()
        for output in outputs:output.close()
        if clips:
            try:clips.close()
            except Exception as exc:
                error=error or f'{type(exc).__name__}: {exc}'
        if eventlog:eventlog.close()
        runtime.close()
        result=dict(error=error,capture=capture.stats()if capture else None,
                    outputs=[dict(restarts=o.restarts,snap=o.snap())for o in outputs],
                    clips=clips.stats()if clips else None,telemetry=telemetry)
        (args.output/'session.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        if error:raise RuntimeError(error)


if __name__=='__main__':main()
