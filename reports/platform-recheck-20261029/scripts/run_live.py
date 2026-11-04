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
    """Pace already encoded H.264 with FFmpeg -re; avoid decode/re-encode sender bottlenecks."""
    def __init__(self, source, url, out, duration=0, manifest=None):
        self.source,self.out=source,out
        self.begin,self.done,self.stop=threading.Event(),threading.Event(),threading.Event()
        self.started=threading.Event();self.error=None;self.timeline={};self.count=0
        self.duration=manifest['seconds']
        self.total_frames=manifest['frames']
        if duration:
            self.duration=min(duration,self.duration);self.total_frames=round(self.duration*25)
        self.preroll=75.
        self.log=(out/'publisher.log').open('w')
        playlist=out/'publisher-concat.txt'
        playlist.write_text("file '../transport/preroll.mp4'\nfile '../"+manifest["input"]+"'\nfile '../transport/tail.mp4'\n")
        self.command=['ffmpeg','-nostdin','-hide_banner','-loglevel','warning','-re',
            '-f','concat','-safe','0','-i',str(playlist),'-map','0:v:0','-an','-c:v','copy',
            '-stats_period','0.2','-progress',str(out/'publisher-progress.txt'),
            '-f','rtsp','-rtsp_transport','tcp',url]
        self.proc=subprocess.Popen(self.command,stdout=self.log,stderr=self.log)
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()

    def run(self):
        try:
            path=self.out/'publisher-progress.txt'
            previous=-1
            with (self.out/'source-progress.jsonl').open('w',buffering=1) as log:
                while not self.stop.is_set():
                    if path.exists():
                        values={}
                        for line in path.read_text().splitlines():
                            if '=' in line:
                                key,value=line.split('=',1);values[key]=value
                        media=float(values.get('out_time_us','0'))/1e6
                        sent=int(values.get('frame','0'))
                        now=time.monotonic()
                        if sent!=previous:
                            log.write(json.dumps(dict(monotonic=now,media_time=media,frames=sent))+'\n')
                            previous=sent
                        self.count=max(0,min(self.total_frames,sent-round(self.preroll*25)))
                        if media>=self.preroll and not self.started.is_set():
                            self.timeline.update(source_start_monotonic=now-(media-self.preroll),
                                preroll_seconds=self.preroll,clock='FFmpeg media progress, sampled every 0.2s; approx wall-clock alignment')
                            save(self.out/'source-timeline.json',self.timeline)
                            self.started.set()
                        if media>=self.preroll+self.duration and not self.done.is_set():
                            self.timeline.update(source_end_monotonic=now-(media-self.preroll-self.duration),source_frames=self.count)
                            save(self.out/'source-timeline.json',self.timeline);self.done.set()
                    if self.proc.poll() is not None:
                        if not self.done.is_set():self.error=f'Publisher exited early: {self.proc.returncode}'
                        break
                    self.stop.wait(.1)
        except BaseException as exc:
            if not self.stop.is_set():self.error=f'{type(exc).__name__}: {exc}'

    def close(self):
        self.stop.set();terminate(self.proc);self.thread.join(timeout=8);self.log.close()
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
    parser=argparse.ArgumentParser();parser.add_argument('--cards',default='0,1,2');parser.add_argument('--name',default='live');parser.add_argument('--duration',type=float,default=0);parser.add_argument('--manifest',default='input-manifest.json');args=parser.parse_args()
    cv2.setNumThreads(2)
    out=OUT/args.name;out.mkdir(exist_ok=False)
    manifest=json.loads((OUT/args.manifest).read_text())
    archive=ROOT/'platform/dark-demo-v2_full_priors_20261027.tar.gz'
    package_manifest=json.loads(Path(str(archive)+'.manifest.json').read_text())
    operator=OUT/'release-extracted/dark-demo-v2'
    for name,info in package_manifest['files'].items():
        assert hashlib.sha256((operator/name).read_bytes()).hexdigest()==info['sha256']
    ip,op,rp,hp=port(),port(),port(),port()
    input_url=f'rtsp://127.0.0.1:{ip}/input'
    push_url=f'rtmp://127.0.0.1:{rp}/enhanced'
    receive_url=f'rtsp://127.0.0.1:{op}/enhanced'
    base=f'http://127.0.0.1:{hp}'
    container='darkpipe-object-live-'+str(os.getpid())
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
        publisher=Publisher(OUT/manifest['input'],input_url,out,args.duration,manifest)
        manifest=dict(manifest,seconds=publisher.duration,frames=publisher.total_frames)
        command=['docker','run','--rm','--name',container,'--gpus','"device='+args.cards+'"',
            '--network','host','--ipc=host','--entrypoint','/opt/conda/envs/darkpipe/bin/python',
            '-e','DARKPIPE_AUDIT_DIR=/evidence/'+args.name,'-v',str(operator)+':/app:ro',
            '-v',str(OUT)+':/evidence:rw','-w','/app','darkpipe-operator:0.3.0',
            '-u','/evidence/scripts/instrumented_operator.py','--video_path',input_url,
            '--gpu_ids',','.join(map(str,range(len(args.cards.split(','))))),
            '--clip_dir','/evidence/'+args.name+'/clips','--session_json','/evidence/'+args.name+'/session.json',
            '--serve_port',str(hp),'--extra_hosts','','--stream_formats','mjpeg,flv,hls',
            '--rtmp_push_url',push_url]
        service=launch(command,'operator.log');startup=time.monotonic()
        save(out/'protocol.json',dict(command=command,input=input_url,push=push_url,external_receive=receive_url,
            base=base,cards=args.cards,source=manifest,package_sha256=package_manifest['sha256'],
            scope='Original image, real RTSP decode and RTMP encode/network receive; local loopback; no offline model processing',
            observation='External wrapper logs output metadata only; no barcode or pixel modifications. Actual push is recorded externally; no independent receiver latency measurement.',
            full_recording='External FFmpeg receives actual RTMP push through relay RTSP; packaged full recorder disabled',
            no_faults_during_main_video=True,publisher_command=publisher.command,publisher_preroll_seconds=publisher.preroll))
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
        # /health can be ready before FFmpeg has published the first RTMP keyframe.
        wait=time.monotonic()
        while True:
            probe=subprocess.run(['ffprobe','-v','error','-rtsp_transport','tcp',
                '-stimeout','3000000','-show_entries','stream=codec_name','-of','json',receive_url],
                capture_output=True,timeout=8)
            if probe.returncode==0 and json.loads(probe.stdout).get('streams'):break
            if service.poll() is not None:raise RuntimeError('Service exited before push became readable')
            if time.monotonic()-wait>35:raise TimeoutError('Push stream did not become readable')
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
        if publisher.started.is_set():raise RuntimeError('Playback began before recorder ready; increase test preroll')
        print('READY: receiver connected; waiting for encoded-source preroll to end.',flush=True)
        while not publisher.started.wait(.1):
            if publisher.error:raise RuntimeError(publisher.error)
            if service.poll() is not None or recorder.poll() is not None:raise RuntimeError('Service/recorder exited during preroll')
        print(f'PLAYING: {manifest["seconds"]:g}-second encoded source at original speed.',flush=True)
        playback=time.monotonic();last_report=-30;clients_started=set()
        while True:
            if service.poll() is not None:raise RuntimeError('Service exited during 300-second playback')
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
            for checkpoint in (10,120,250,410,610,850):
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
        '-c','copy','-movflags','+faststart','-y',str(out/'full-output-stream.mp4')],check=True)
    print('COMPLETE: real stream saved, clips drained, service stopped normally.',flush=True)

if __name__=='__main__':main()
