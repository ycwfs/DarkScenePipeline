"""Verify that gate changes really leave the cached new-scene frontend unchanged."""
from common import *
import cv2
from darkpipe.stages.downscale import DownscaleStage
from darkpipe.stages.recognize import frame_luma


def main():
    cv2.setNumThreads(1)
    protocol=json.loads((OUT/'protocol.json').read_text())
    stats=[]; arrays={}; down=DownscaleStage(720)
    for s in protocol['sources']:
        if s['scope']!='new_multipeople': continue
        cap=cv2.VideoCapture(str(ROOT/s['source'])); assert cap.isOpened()
        values=[]
        while True:
            ok,frame=cap.read()
            if not ok:break
            values.append(frame_luma(down([frame])[0]))
        cap.release(); assert len(values)==s['frames']
        a=np.array(values); arrays[s['id']]=a
        events=json.loads((ROOT/s['cache']).read_text())
        errs=[abs(np.mean(a[e['source_frames']])-e['raw_luma']) for e in events]
        assert max(errs)<1e-7, (s['id'],max(errs))
        row=dict(id=s['id'],frames=len(a),minimum=float(a.min()),maximum=float(a.max()),
                 frames_below_gate={str(g):int((a<g).sum()) for g in GATES},max_window_luma_error=max(errs))
        stats.append(row); print(row,flush=True)
    np.savez_compressed(OUT/'new-frame-luma.npz',**arrays)
    save(OUT/'new-luma-audit.json',dict(sources=stats,all_above_max_gate=all(r['minimum']>=max(GATES) for r in stats)))


if __name__=='__main__':main()
