"""A loopback peer accepts TCP but sends no RTSP response; compare open deadlines."""
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]/'evidence/capture-timeout.json'


def main():
    server=socket.socket();server.bind(('127.0.0.1',0));server.listen(8);server.settimeout(.2)
    halt=threading.Event();connections=[]
    def accept():
        while not halt.is_set():
            try:connections.append(server.accept()[0])
            except socket.timeout:pass
    thread=threading.Thread(target=accept,daemon=True);thread.start()
    url=f'rtsp://127.0.0.1:{server.getsockname()[1]}/silent'
    statements={
        'current_production':f"import sys;sys.path.insert(0,{str(ROOT/'platform/dark-demo-v2')!r});from darkpipe.media import open_capture;open_capture({url!r})",
        'candidate_bounded':f"import cv2;c=cv2.VideoCapture({url!r},cv2.CAP_FFMPEG,[cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,3000,cv2.CAP_PROP_READ_TIMEOUT_MSEC,2000]);print(c.isOpened());c.release()"}
    rows=[]
    try:
        for name,code in statements.items():
            start=time.monotonic();proc=subprocess.Popen([sys.executable,'-c',code],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            timed_out=False
            try:stdout,stderr=proc.communicate(timeout=7)
            except subprocess.TimeoutExpired:
                timed_out=True;proc.kill();stdout,stderr=proc.communicate()
            rows.append(dict(name=name,elapsed=time.monotonic()-start,test_deadline_exceeded=timed_out,
                             returncode=proc.returncode,stdout=stdout.decode(),stderr=stderr.decode()))
    finally:
        halt.set();thread.join(timeout=1);server.close()
        for conn in connections:conn.close()
    OUT.write_text(json.dumps(rows,indent=2)+'\n')
    assert rows[0]['test_deadline_exceeded'] and not rows[1]['test_deadline_exceeded']
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
