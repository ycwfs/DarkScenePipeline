"""Real RTSP input and RTMP output; external full recording, no platform recorder."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from urllib.request import urlopen
from urllib.error import HTTPError

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reports/person-stream-audit-20261023/scripts'))
from stream_experiment import port, terminate, NativeReceiver


def save(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


class Publisher:
    """One uninterrupted RTSP session: blank preroll, source once, blank tail."""
    def __init__(self, source, url, out):
        self.source,self.out=source,out
        self.begin,self.done,self.stop=threading.Event(),threading.Event(),threading.Event()
        self.error=None;self.timeline={};self.count=0
        self.log=(out/'publisher.log').open('w')
        self.command=['ffmpeg','-nostdin','-hide_banner','-loglevel','warning','-f','rawvideo',
            '-pixel_format','bgr24','-video_size','1920x1080','-framerate','25','-i','pipe:0',
            '-an','-c:v','libx264','-preset','ultrafast','-tune','zerolatency','-crf','18',
            '-threads','2','-pix_fmt','yuv420p','-g','25','-bf','0','-f','rtsp','-rtsp_transport','tcp',url]
        self.proc=subprocess.Popen(self.command,stdin=subprocess.PIPE,stdout=self.log,stderr=self.log)
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()

    def run(self):
        cap=None
        try:
            blank=np.zeros((1080,1920,3),np.uint8).tobytes()
            next_time=time.monotonic()
            with (self.out/'input-send.jsonl').open('w',buffering=1) as rows:
                while not self.stop.is_set():
                    if self.begin.is_set() and cap is None:
                        cap=cv2.VideoCapture(str(self.source));assert cap.isOpened()
                    content=None
                    if cap is not None and not self.done.is_set():
                        ok,frame=cap.read()
                        if ok:
                            content=frame.tobytes()
                        else:
                            self.timeline['source_end_monotonic']=time.monotonic()
                            self.timeline['source_frames']=self.count
                            save(self.out/'source-timeline.json',self.timeline)
                            self.done.set()
                    before=time.monotonic()
                    if content is not None and self.count==0:
                        self.timeline['source_start_monotonic']=before
                        save(self.out/'source-timeline.json',self.timeline)
                    self.proc.stdin.write(content if content is not None else blank)
                    if content is not None:
                        rows.write(json.dumps(dict(frame=self.count,send_begin=before,send_end=time.monotonic()))+'\n')
                        self.count+=1
                    next_time+=.04
                    self.stop.wait(max(0.,next_time-time.monotonic()))
        except BaseException as exc:
            if not self.stop.is_set():self.error=f'{type(exc).__name__}: {exc}'
        finally:
            if cap is not None:cap.release()

    def close(self):
        self.stop.set();terminate(self.proc);self.thread.join(timeout=8)
        if self.proc.stdin:self.proc.stdin.close()
        self.log.close()
        assert not self.thread.is_alive()


class Events:
    def __init__(self,base,out):
        self.base,self.out=base,out;self.stop=threading.Event();self.rows=[];self.errors=[]
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
    def run(self):
        with (self.out/'sse.jsonl').open('w',buffering=1) as log:
            while not self.stop.is_set():
                try:
                    with urlopen(self.base+'/events',timeout=20) as response:
                        for line in response:
                            if self.stop.is_set():break
                            if line.startswith(b'data:'):
                                value=json.loads(line[5:]);row=dict(received_monotonic=time.monotonic(),event=value)
                                self.rows.append(row);log.write(json.dumps(row,ensure_ascii=False)+'\n')
                except Exception as exc:
                    if not self.stop.is_set():self.errors.append(dict(time=time.monotonic(),error=str(exc)))
                self.stop.wait(.2)
    def close(self):
        self.stop.set();self.thread.join(timeout=22)
        save(self.out/'sse-status.json',dict(events=len(self.rows),errors=self.errors,thread_stopped=not self.thread.is_alive()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cards',default='0,1,4');args=parser.parse_args()
    cv2.setNumThreads(2)
    out=OUT/'live';out.mkdir(exist_ok=False)
    manifest=json.loads((OUT/'input-manifest.json').read_text())
    archive=ROOT/'platform/dark-demo-v2_person_tau075_20261024.tar.gz'
    package_manifest=json.loads(Path(str(archive)+'.manifest.json').read_text())
    operator=OUT/'operator/dark-demo-v2'
    for name,info in package_manifest['files'].items():
        assert hashlib.sha256((operator/name).read_bytes()).hexdigest()==info['sha256']
    ip,op,rp,hp=port(),port(),port(),port()
    input_url=f'rtsp://127.0.0.1:{ip}/input'
    push_url=f'rtmp://127.0.0.1:{rp}/enhanced'
    receive_url=f'rtsp://127.0.0.1:{op}/enhanced'
    base=f'http://127.0.0.1:{hp}'
    container='darkpipe-five-minute-'+str(os.getpid())
    owned=[];logs=[];publisher=service=receiver=sse=recorder=None
    health=[];http=[];successful=False
    def launch(command,name,stdin=None):
        logfile=(out/name).open('w');logs.append(logfile)
        process=subprocess.Popen(command,stdout=logfile,stderr=subprocess.STDOUT,cwd=ROOT,stdin=stdin)
        owned.append(process);return process
    def relay(label,rtsp,rtmp=None):
        config=out/(label+'.yml')
        config.write_text(f'logLevel: warn\nrtspAddress: 127.0.0.1:{rtsp}\nrtspTransports: [tcp]\n'
            f'rtmp: {"yes" if rtmp else "no"}\n'+(f'rtmpAddress: 127.0.0.1:{rtmp}\n'if rtmp else '')+
            'hls: no\nwebrtc: no\nsrt: no\nmoq: no\npaths:\n  all_others:\n')
        launch([str(ROOT/'reports/person-stream-audit-20261023/.tools/mediamtx'),str(config)],label+'.log')
    try:
        relay('input-relay',ip);relay('output-relay',op,rp);time.sleep(.5)
        publisher=Publisher(OUT/manifest['input'],input_url,out)
        command=['docker','run','--rm','--name',container,'--gpus','"device='+args.cards+'"',
            '--network','host','--ipc=host','--entrypoint','/opt/conda/envs/darkpipe/bin/python',
            '-e','DARKPIPE_AUDIT_DIR=/evidence/live','-v',str(operator)+':/app:ro',
            '-v',str(OUT)+':/evidence:rw','-w','/app','darkpipe-operator:0.3.0',
            '-u','/evidence/scripts/instrumented_operator.py','--video_path',input_url,
            '--gpu_ids',','.join(map(str,range(len(args.cards.split(','))))),
            '--clip_dir','/evidence/live/clips','--session_json','/evidence/live/session.json',
            '--serve_port',str(hp),'--extra_hosts','','--stream_formats','mjpeg,flv,hls',
            '--rtmp_push_url',push_url]
        service=launch(command,'operator.log');startup=time.monotonic()
        save(out/'protocol.json',dict(command=command,input=input_url,push=push_url,external_receive=receive_url,
            base=base,cards=args.cards,source=manifest,package_sha256=package_manifest['sha256'],
            scope='Original image, real RTSP decode and RTMP encode/network receive; local loopback; no offline model processing',
            observation='External wrapper logs output metadata and adds a 192x16 capture-frame barcode only after recognition. Models, inputs, clips and packaged files are unchanged.',
            full_recording='External FFmpeg receives actual RTMP push through relay RTSP; packaged full recorder disabled',
            no_faults_during_main_video=True))
        while True:
            if service.poll() is not None:raise RuntimeError('Service exited during initialization')
            if publisher.error:raise RuntimeError(publisher.error)
            try:
                with urlopen(base+'/health',timeout=1) as response:
                    status=json.load(response)
                    if status['status']=='ok':break
            except OSError:pass
            if time.monotonic()-startup>180:raise TimeoutError('Initialization timeout')
            time.sleep(.5)
        with urlopen(base+'/config',timeout=3) as response:save(out/'effective-config.json',json.load(response))
        receiver=NativeReceiver(receive_url,out/'receiver-latency.log')
        wait=time.monotonic()
        while len(receiver.rows)<25:
            if service.poll() is not None:raise RuntimeError('Service exited before receiver connected')
            if time.monotonic()-wait>40:raise TimeoutError('Push stream did not become readable')
            time.sleep(.2)
        # The main full recording consumes the actual active-push output, not the processing callback.
        recorder=launch(['ffmpeg','-nostdin','-hide_banner','-loglevel','warning','-rtsp_transport','tcp',
            '-stimeout','5000000','-i',receive_url,'-map','0:v:0','-an','-c:v','copy',
            '-progress',str(out/'record-progress.txt'),'-y',str(out/'full-output-untrimmed.mkv')], 'full-recorder.log')
        wait=time.monotonic()
        while True:
            if recorder.poll() is not None:raise RuntimeError('External recorder exited before playback')
            progress=out/'record-progress.txt'
            if progress.exists() and any(line.startswith('frame=') and int(line[6:])>=25 for line in progress.read_text().splitlines()):break
            if time.monotonic()-wait>30:raise TimeoutError('External recorder did not receive frames')
            time.sleep(.2)
        sse=Events(base,out)
        time.sleep(2)
        publisher.begin.set()
        print('PLAYING: five-minute source, recorder already receiving.',flush=True)
        playback=time.monotonic();last_report=-30;clients_started=set()
        while True:
            if service.poll() is not None:raise RuntimeError('Service exited during five-minute playback')
            if publisher.error:raise RuntimeError(publisher.error)
            if recorder.poll() is not None:raise RuntimeError('External recorder interrupted during playback')
            elapsed=time.monotonic()-playback
            if publisher.done.is_set() and time.monotonic()-publisher.timeline['source_end_monotonic']>=8:break
            if elapsed>manifest['seconds']+60:raise TimeoutError('Publisher did not finish on schedule')
            try:
                try:response=urlopen(base+'/health',timeout=2)
                except HTTPError as exc:response=exc
                with response:row=dict(monotonic=time.monotonic(),elapsed=elapsed,http_status=response.status,health=json.load(response))
            except OSError as exc:row=dict(monotonic=time.monotonic(),elapsed=elapsed,error=str(exc))
            health.append(row);save(out/'health-history.json',health)
            # Connect/reconnect independent HTTP viewers while the full push recording stays uninterrupted.
            for checkpoint in (10,140,275):
                if elapsed>=checkpoint and checkpoint not in clients_started:
                    clients_started.add(checkpoint)
                    for label,path in [('mjpeg','/stream'),('flv','/live.flv'),('hls','/hls/index.m3u8')]:
                        target=out/f'viewer-{checkpoint}-{label}.mkv'
                        client=launch(['ffmpeg','-nostdin','-v','error','-rw_timeout','10000000','-i',base+path,
                            '-t','2','-an','-c:v','copy','-y',str(target)],f'viewer-{checkpoint}-{label}.log')
                        http.append((target.name,client))
            if elapsed-last_report>=30:
                h=row.get('health',{})
                print(f'LIVE {elapsed:.0f}s input={publisher.count}/{manifest["frames"]} health={h.get("status")} fps={h.get("fps_proc")} events={h.get("events_total")}',flush=True)
                last_report=elapsed
            time.sleep(1)
        assert publisher.count==manifest['frames'], (publisher.count,manifest['frames'])
        save(out/'receivers-before-stop.json',dict(frames=len(receiver.rows),errors=receiver.errors))
        subprocess.run(['docker','kill','--signal=TERM',container],capture_output=True,check=True)
        service.wait(timeout=120)
        if service.returncode:raise RuntimeError(f'Service exit={service.returncode}')
        session=json.loads((out/'session.json').read_text());assert session['status']=='stopped',session
        if recorder.poll() is None:
            recorder.send_signal(signal.SIGINT);recorder.wait(timeout=15)
        save(out/'complete.json',dict(normal_stop=True,service_returncode=service.returncode,
            session_status=session['status'],source_frames=publisher.count,
            source_duration_seconds=manifest['seconds'],wall_playback_seconds=publisher.timeline['source_end_monotonic']-publisher.timeline['source_start_monotonic'],
            recorder_returncode=recorder.returncode,http_clients={name:proc.poll() for name,proc in http},
            completion_monotonic=time.monotonic()))
        successful=True
    except BaseException as exc:
        save(out/'failure.json',dict(error=f'{type(exc).__name__}: {exc}'));raise
    finally:
        if service is not None and service.poll() is None:
            subprocess.run(['docker','stop','-t','40',container],capture_output=True,timeout=50)
        if receiver is not None:
            receiver.close();save(out/'receiver.json',dict(url=receiver.url,frames=receiver.rows,errors=receiver.errors))
        if sse is not None:sse.close()
        if publisher is not None:publisher.close()
        for process in reversed(owned):terminate(process)
        for logfile in logs:logfile.close()
        save(out/'cleanup.json',dict(success=successful,owned_processes_alive=sum(p.poll() is None for p in owned)))
    subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(out/'full-output-untrimmed.mkv'),
        '-c','copy','-movflags','+faststart','-y',str(OUT/'full-output-stream.mp4')],check=True)
    print('COMPLETE: real stream saved, clips drained, service stopped normally.',flush=True)

if __name__=='__main__':main()
