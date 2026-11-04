"""Real-time recording capacity with controlled simultaneous events, not model output."""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'platform/dark-demo-v2'))
from person_clips import PersonClips


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--people',type=int,nargs='+',default=[4,8])
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(2)
    all_results=[]
    for people in args.people:
        cap=cv2.VideoCapture(str(args.source));assert cap.isOpened()
        origin=time.monotonic();rec=PersonClips(args.output/f'{people}-actors',origin,denoise='off')
        try:
            for i in range(300):
                stamp=i/25
                time.sleep(max(0,origin+stamp-time.monotonic()))
                ok,image=cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,0);ok,image=cap.read()
                assert ok
                if i%12==0 and 3<=stamp<=8:
                    for tid in range(1,people+1):
                        rec.event(dict(session=1,track_id=tid,history_generation=0,timestamp=stamp,
                            frame_index=i,label='wave'if tid%2 else 'drink',confidence=.95))
                rec.frame(image,dict(frame=i,timestamp=stamp,capture_epoch=0))
            result=rec.close()
        finally:cap.release()
        result.update(people=people,elapsed=time.monotonic()-origin,
                      stimulus='controlled simultaneous events; no claim of real model detections')
        all_results.append(result)
        (args.output/'results.json').write_text(json.dumps(all_results,ensure_ascii=False,indent=2)+'\n')
        assert len(result['clips'])==people
        assert all(result['totals'][k]==0 for k in ['frames_dropped','clips_abandoned','clips_failed','write_errors'])
        assert result.get('dispatcher_frames_dropped',0)==result.get('late_action_events',0)==0
        assert all(c['context_complete']for c in result['clips'])
        print(people,'simultaneous full-frame clips: PASS',flush=True)


if __name__=='__main__':main()
