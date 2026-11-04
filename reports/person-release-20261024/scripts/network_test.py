"""Owned local RTSP input, real image execution, output clients, faults, graceful stop."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reports/person-stream-audit-20261023/scripts'))
from stream_experiment import port, terminate, NativeReceiver


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--name',required=True);p.add_argument('--seconds',type=float,default=300)
    p.add_argument('--source',default='real-actiondata/挥手.mp4')
    p.add_argument('--operator',type=Path,default=ROOT/'platform/dark-demo-v2')
    p.add_argument('--cards',default='0,1,3');p.add_argument('--push-protocol',default='rtmp',choices=['rtmp','rtsp'])
    p.add_argument('--processed',action='store_true');p.add_argument('--no-faults',action='store_true')
    args=p.parse_args()
    out=OUT/args.name;out.mkdir(parents=True,exist_ok=False)
    ip,op,rp,hp=port(),port(),port(),port()
    input_url=f'rtsp://127.0.0.1:{ip}/input'
    rtmp=f'rtmp://127.0.0.1:{rp}/enhanced';rtsp=f'rtsp://127.0.0.1:{op}/enhanced'
    name='darkpipe-person-audit-'+str(os.getpid())
    logs=[];owned=[];receivers=[];telemetry=[];faults=[]
    def launch(cmd,logname):
        log=(out/logname).open('a');logs.append(log)
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        owned.append(proc);return proc
    def relay(label,rtsp_port,rtmp_port=None):
        config=out/(label+'.yml')
        config.write_text(f'logLevel: warn\nrtspAddress: 127.0.0.1:{rtsp_port}\nrtspTransports: [tcp]\n'
            f'rtmp: {"yes" if rtmp_port else "no"}\n'+(f'rtmpAddress: 127.0.0.1:{rtmp_port}\n' if rtmp_port else '')+
            'hls: no\nwebrtc: no\nsrt: no\nmoq: no\npaths:\n  all_others:\n')
        return launch([str(ROOT/'reports/person-stream-audit-20261023/.tools/mediamtx'),str(config)],label+'.log')
    def publisher():
        return launch(['ffmpeg','-hide_banner','-loglevel','warning','-nostdin','-re','-stream_loop','-1',
            '-i',str(ROOT/args.source),'-an','-c:v','libx264','-preset','ultrafast','-tune','zerolatency',
            '-threads','2','-pix_fmt','yuv420p','-g','25','-bf','0','-f','rtsp','-rtsp_transport','tcp',input_url],'publisher.log')
    service=None;publisher_process=None;output_relay=None
    error=None
    try:
        relay('input-relay',ip);output_relay=relay('output-relay',op,rp)
        time.sleep(.7)
        publisher_process=publisher()
        command=['docker','run','--rm','--name',name,'--gpus','"device='+args.cards+'"','--network','host','--ipc=host',
            '--entrypoint','/opt/conda/envs/darkpipe/bin/python','-e','DARKPIPE_AUDIT_DIR=/evidence/'+args.name,
            '-v',str(args.operator.resolve())+':/app:ro','-v',str(OUT)+':/evidence:rw','-w','/app','darkpipe-operator:0.3.0',
            '-u','/evidence/scripts/instrumented_operator.py','--video_path',input_url,
            '--gpu_ids',','.join(str(i) for i in range(len(args.cards.split(',')))),
            '--clip_dir','/evidence/'+args.name+'/clips','--session_json','/evidence/'+args.name+'/session.json',
            '--serve_port',str(hp),'--extra_hosts','','--stream_formats','mjpeg,flv,hls',
            '--rtmp_push_url',rtmp if args.push_protocol=='rtmp' else rtsp,
            '--input_processed',str(args.processed).lower()]
        service=launch(command,'operator.log')
        start=time.monotonic()
        base=f'http://127.0.0.1:{hp}'
        while True:
            if service.poll() is not None:
                raise RuntimeError('Operator exited during startup')
            try:
                with urlopen(base+'/health',timeout=1) as response:
                    if json.load(response)['status']=='ok':break
            except (OSError,HTTPError,URLError):pass
            if time.monotonic()-start>180:raise TimeoutError('Operator initialization exceeded180s')
            time.sleep(.5)
        ready=time.monotonic()
        receivers=[NativeReceiver(rtsp,out/'receiver-rtsp.log'),NativeReceiver(rtmp,out/'receiver-rtmp.log')]
        save(out/'protocol.json',dict(command=command,source=args.source,processed=args.processed,ready=ready,
            ready_seconds=ready-start,cards=args.cards,seconds=args.seconds,scope='local loopback, original0.3.0 image',
            instrumentation='JPEG frame counter only; clips and model pixels unchanged',base=base))
        # Real HTTP clients consume finite MJPEG, FLV and HLS intervals; no synthetic success.
        http_clients=[]
        time.sleep(7)  # allow the first HLS playlist and media segment to finish
        for label,path in [('mjpeg','/stream'),('flv','/live.flv'),('hls','/hls/index.m3u8')]:
            http_clients.append((label,launch(['ffmpeg','-v','error','-rw_timeout','10000000','-i',base+path,
                '-t','3','-an','-c:v','copy','-y',str(out/f'http-{label}.mkv')],f'http-{label}.log')))
        schedule=[] if args.no_faults else [(45,'source_pause'),(53,'source_resume'),(90,'output_down'),
            (100,'output_up'),(140,'source_down'),(150,'source_up'),(200,'viewer_reconnect')]
        done=set()
        while time.monotonic()-ready<args.seconds:
            if service.poll() is not None:raise RuntimeError('Operator exited before requested stop')
            elapsed=time.monotonic()-ready
            for second,action in schedule:
                if second>=args.seconds-10 or action in done or elapsed<second:continue
                if action=='source_pause':publisher_process.send_signal(signal.SIGSTOP)
                elif action=='source_resume':publisher_process.send_signal(signal.SIGCONT)
                elif action=='output_down':terminate(output_relay)
                elif action=='output_up':output_relay=relay('output-relay',op,rp)
                elif action=='source_down':terminate(publisher_process)
                elif action=='source_up':publisher_process=publisher()
                elif action=='viewer_reconnect':
                    receivers[0].close();receivers.append(NativeReceiver(rtsp,out/'receiver-reconnect.log'))
                done.add(action);faults.append(dict(time=elapsed,action=action,monotonic=time.monotonic()))
                save(out/'faults.json',faults)
            try:
                try:response=urlopen(base+'/health',timeout=2)
                except HTTPError as exc:response=exc
                with response:health=json.load(response)
                telemetry.append(dict(time=time.monotonic()-ready,health=health))
            except OSError as exc:telemetry.append(dict(time=time.monotonic()-ready,error=str(exc)))
            save(out/'health-history.json',telemetry)
            time.sleep(1)
        subprocess.run(['docker','kill','--signal=TERM',name],check=True,capture_output=True)
        service.wait(timeout=120)
        if service.returncode:raise RuntimeError(f'Operator returned{service.returncode}')
        session=json.loads((out/'session.json').read_text())
        assert session['status']=='stopped',session
        save(out/'complete.json',dict(returncode=service.returncode,normal_stop=True,
            http_clients={label:proc.poll() for label,proc in http_clients},session_status=session['status']))
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}'
        save(out/'failure.json',dict(error=error))
        raise
    finally:
        for receiver in receivers:receiver.close()
        save(out/'receivers.json',[dict(url=r.url,frames=r.rows,errors=r.errors) for r in receivers])
        save(out/'faults.json',faults)
        if service is not None and service.poll() is None:
            subprocess.run(['docker','stop','-t','40',name],capture_output=True,timeout=50)
        if publisher_process is not None and publisher_process.poll() is None:
            publisher_process.send_signal(signal.SIGCONT)
        for proc in reversed(owned):terminate(proc)
        for log in logs:log.close()


if __name__=='__main__':
    main()
