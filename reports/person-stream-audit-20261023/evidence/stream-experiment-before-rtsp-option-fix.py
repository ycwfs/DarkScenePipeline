"""Real loopback RTSP/RTMP experiment with owned relays and injected outages."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
BASE=Path(__file__).resolve().parents[1]


def save(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def port():
    with socket.socket()as s:
        s.bind(('127.0.0.1',0))
        return s.getsockname()[1]


def terminate(proc):
    if proc is not None and proc.poll()is None:
        proc.terminate()
        try:proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill();proc.wait(timeout=5)


class Receiver:
    def __init__(self,url):
        self.url=url;self.stop=threading.Event();self.rows=[];self.errors=[]
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()

    def run(self):
        while not self.stop.is_set():
            cap=None
            try:
                cap=cv2.VideoCapture(self.url,cv2.CAP_FFMPEG,[cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,3000,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC,2000])
                if not cap.isOpened():raise RuntimeError('viewer open failed')
                while not self.stop.is_set():
                    ok,frame=cap.read()
                    if not ok:raise RuntimeError('viewer read ended')
                    index=sum(int(frame[4:12,bit*8+2:bit*8+6].mean()>127)<<bit for bit in range(24))
                    self.rows.append(dict(time=time.monotonic(),frame=index))
            except Exception as exc:
                self.errors.append(dict(time=time.monotonic(),error=str(exc)))
                self.stop.wait(.5)
            finally:
                if cap is not None:cap.release()

    def close(self):
        self.stop.set();self.thread.join(timeout=8)
        assert not self.thread.is_alive()


class NativeReceiver:
    """System FFmpeg client, with a small marker-only raw output and bounded I/O."""
    def __init__(self,url,logpath):
        self.url=url;self.stop=threading.Event();self.rows=[];self.errors=[];self.proc=None
        self.logpath=logpath
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()

    def run(self):
        with self.logpath.open('a')as log:
            while not self.stop.is_set():
                transport=['-rtsp_transport','tcp']if self.url.startswith('rtsp:')else []
                cmd=['ffmpeg','-hide_banner','-loglevel','warning','-nostdin',*transport,
                     '-rw_timeout','3000000','-fflags','nobuffer','-flags','low_delay',
                     '-analyzeduration','100000','-probesize','32768','-i',self.url,
                     '-an','-vf','crop=192:16:0:0,format=gray','-threads','1',
                     '-pix_fmt','gray','-f','rawvideo','pipe:1']
                self.proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=log)
                try:
                    while not self.stop.is_set():
                        data=self.proc.stdout.read(192*16)
                        if len(data)!=192*16:break
                        frame=np.frombuffer(data,np.uint8).reshape(16,192)
                        index=sum(int(frame[4:12,bit*8+2:bit*8+6].mean()>127)<<bit for bit in range(24))
                        self.rows.append(dict(time=time.monotonic(),frame=index))
                finally:
                    terminate(self.proc)
                    self.proc.stdout.close()
                self.errors.append(dict(time=time.monotonic(),error='receiver disconnected',returncode=self.proc.returncode))
                self.stop.wait(.5)

    def close(self):
        self.stop.set()
        terminate(self.proc)
        self.thread.join(timeout=8)
        assert not self.thread.is_alive()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=240)
    p.add_argument('--source',default='real-actiondata/挥手.mp4')
    p.add_argument('--denoise',default='off',choices=['off','fast','quality'])
    p.add_argument('--quick-faults',action='store_true')
    p.add_argument('--stop-at',type=float,default=0.,help='Send SIGTERM to the owned service at this source time')
    args=p.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=(ROOT/args.source).resolve()
    input_port,output_port,rtmp_port=port(),port(),port()
    input_url=f'rtsp://127.0.0.1:{input_port}/input'
    output_rtsp=f'rtsp://127.0.0.1:{output_port}/enhanced'
    output_rtmp=f'rtmp://127.0.0.1:{rtmp_port}/enhanced-rtmp'
    logs=[];owned=[];receivers=[];faults=[]
    def launch(cmd,logname):
        log=(out/logname).open('a');logs.append(log)
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        owned.append(proc);return proc
    def relay(name,rtsp,rtmp=None):
        config=out/f'{name}.yml'
        config.write_text(f'logLevel: warn\nrtspAddress: 127.0.0.1:{rtsp}\nrtspTransports: [tcp]\n'
            f'rtmp: {"yes" if rtmp else "no"}\n'+(f'rtmpAddress: 127.0.0.1:{rtmp}\n'if rtmp else '')+
            'hls: no\nwebrtc: no\nsrt: no\nmoq: no\npaths:\n  all_others:\n')
        return launch([str(BASE/'.tools/mediamtx'),str(config)],f'{name}.log')
    def publisher():
        return launch(['ffmpeg','-hide_banner','-loglevel','warning','-nostdin','-re','-stream_loop','-1',
            '-i',str(source),'-an','-c:v','libx264','-preset','ultrafast','-tune','zerolatency',
            '-threads','2','-pix_fmt','yuv420p','-g','25','-bf','0','-f','rtsp','-rtsp_transport','tcp',input_url],
            'source.log')
    service=None;source_proc=None;output_relay=None
    try:
        relay('input-relay',input_port)
        output_relay=relay('output-relay',output_port,rtmp_port)
        time.sleep(.7)
        if any(proc.poll() is not None for proc in owned):
            raise RuntimeError('A test relay exited; inspect relay logs')
        source_proc=publisher()
        service=launch([sys.executable,'-u',str(BASE/'scripts/run_stream.py'),'--input',input_url,
            '--output',str(out/'operator'),'--push',output_rtsp,'--push',output_rtmp,
            '--seconds',str(args.seconds),'--clip-denoise',args.denoise,'--audit-markers'], 'operator.log')
        wait_start=time.monotonic()
        while not (out/'operator/ready.json').exists():
            if service.poll()is not None:raise RuntimeError('Service exited during initialization')
            if time.monotonic()-wait_start>150:raise TimeoutError('Service initialization timeout')
            time.sleep(.5)
        ready=json.loads((out/'operator/ready.json').read_text())
        origin=ready['origin']
        receivers=[NativeReceiver(output_rtsp,out/'viewer-rtsp.log'),NativeReceiver(output_rtmp,out/'viewer-rtmp.log')]
        save(out/'protocol.json',dict(source=str(source),input_url=input_url,outputs=[output_rtsp,output_rtmp],
            origin=origin,seconds=args.seconds,denoise=args.denoise,gpus=os.environ.get('CUDA_VISIBLE_DEVICES'),
            model='same ROI VideoMamba and ST-GCN',test='loopback network; independent capture and output relays'))
        completed=set()
        schedule=[(60,'output_down'),(72,'output_up'),(110,'source_pause'),(118,'source_resume'),
                  (155,'source_down'),(163,'source_up'),(200,'viewer_reconnect')]
        if args.quick_faults:
            schedule=[(12,'source_pause'),(20,'source_resume'),(30,'output_down'),(38,'output_up'),
                      (50,'source_down'),(56,'source_up'),(65,'viewer_reconnect')]
        while service.poll()is None:
            elapsed=time.monotonic()-origin
            if args.stop_at and elapsed>=args.stop_at and 'sigterm'not in completed:
                service.send_signal(signal.SIGTERM);completed.add('sigterm')
                faults.append(dict(time=elapsed,action='sigterm'));save(out/'faults.json',faults)
            for second,action in schedule:
                if second>=args.seconds-10 or action in completed or elapsed<second:continue
                if action=='output_down':terminate(output_relay)
                elif action=='output_up':output_relay=relay('output-relay',output_port,rtmp_port)
                elif action=='source_pause':source_proc.send_signal(signal.SIGSTOP)
                elif action=='source_resume':source_proc.send_signal(signal.SIGCONT)
                elif action=='source_down':terminate(source_proc)
                elif action=='source_up':source_proc=publisher()
                elif action=='viewer_reconnect':
                    receivers[0].close();receivers.append(NativeReceiver(output_rtsp,out/'viewer-rtsp-reconnect.log'))
                completed.add(action);faults.append(dict(time=time.monotonic()-origin,action=action))
                save(out/'faults.json',faults)
            if elapsed>args.seconds+120:raise TimeoutError('Service did not stop')
            time.sleep(.25)
        if service.returncode:raise RuntimeError(f'Service exited {service.returncode}')
    finally:
        if source_proc is not None and source_proc.poll()is None:
            source_proc.send_signal(signal.SIGCONT)
        for r in receivers:r.close()
        for proc in reversed(owned):terminate(proc)
        for log in logs:log.close()
        save(out/'receivers.json',[dict(url=r.url,frames=r.rows,errors=r.errors)for r in receivers])
        save(out/'faults.json',faults)
        save(out/'exit.json',dict(service_returncode=service.returncode if service else None,
                                 child_processes_left=sum(p.poll()is None for p in owned)))
    assert receivers and all(r.rows for r in receivers), 'Every receiver must decode actual output frames'
    save(out/'complete.json',dict(completed=True,receiver_frames=[len(r.rows)for r in receivers]))


if __name__=='__main__':main()
