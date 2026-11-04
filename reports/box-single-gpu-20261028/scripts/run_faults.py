"""Fault injection against final archive, owned loopback relays and real decoders."""
from pathlib import Path
import argparse,json,os,signal,subprocess,sys,threading,time
from urllib.request import urlopen
from urllib.error import HTTPError
from run_live import port,terminate,save
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parents[1]
class Receiver:
 def __init__(self,url,out):
  self.url=url;self.out=out;self.stop=threading.Event();self.rows=[];self.errors=[];self.proc=None
  self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
 def run(self):
  with (self.out/'receiver.log').open('a')as log:
   while not self.stop.is_set():
    self.proc=subprocess.Popen(['ffmpeg','-nostdin','-v','warning','-rtsp_transport','tcp','-stimeout','3000000','-fflags','nobuffer','-flags','low_delay','-i',self.url,'-an','-vf','scale=32:18,format=gray','-threads','1','-f','rawvideo','pipe:1'],stdout=subprocess.PIPE,stderr=log)
    try:
     while not self.stop.is_set():
      data=self.proc.stdout.read(32*18)
      if len(data)!=32*18:break
      self.rows.append(dict(time=time.monotonic(),bytes=len(data)))
    finally:
     terminate(self.proc);self.proc.stdout.close()
    if not self.stop.is_set():self.errors.append(dict(time=time.monotonic(),returncode=self.proc.returncode))
    self.stop.wait(.3)
 def close(self):
  self.stop.set();terminate(self.proc);self.thread.join(timeout=10);assert not self.thread.is_alive()
def main():
 p=argparse.ArgumentParser();p.add_argument('--cards',default='0');a=p.parse_args()
 out=OUT/'faults';out.mkdir(exist_ok=False);ip,op,rp,hp=port(),port(),port(),port();base=f'http://127.0.0.1:{hp}'
 input_url=f'rtsp://127.0.0.1:{ip}/input';push=f'rtmp://127.0.0.1:{rp}/enhanced';recv=f'rtsp://127.0.0.1:{op}/enhanced'
 logs=[];owned=[];receivers=[];health=[];faults=[];container='darkpipe-full-fault-'+str(os.getpid());service=source=None
 def launch(cmd,name):
  log=(out/name).open('a');logs.append(log);proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT);owned.append(proc);return proc
 def relay(name,rtsp,rtmp=None):
  cfg=out/(name+'.yml');cfg.write_text(f'logLevel: warn\nrtspAddress: 127.0.0.1:{rtsp}\nrtspTransports: [tcp]\nrtmp: '+('yes\n'+f'rtmpAddress: 127.0.0.1:{rtmp}\n'if rtmp else'no\n')+'hls: no\nwebrtc: no\nsrt: no\nmoq: no\npaths:\n  all_others:\n')
  return launch([str(ROOT/'reports/person-stream-audit-20261023/.tools/mediamtx'),str(cfg)],name+'.log')
 def publisher():return launch(['ffmpeg','-nostdin','-v','warning','-re','-stream_loop','-1','-i',str(OUT/'input-live-300s.mp4'),'-an','-c:v','copy','-f','rtsp','-rtsp_transport','tcp',input_url],'publisher.log')
 def get_health():
  try:
   try:resp=urlopen(base+'/health',timeout=2)
   except HTTPError as exc:resp=exc
   with resp:return dict(time=time.monotonic(),http_status=resp.status,health=json.load(resp))
  except OSError as exc:return dict(time=time.monotonic(),error=str(exc))
 try:
  relay('input-relay',ip);sink=relay('output-relay',op,rp);time.sleep(.7);source=publisher()
  operator=Path(os.environ['BOX_RUNTIME'])
  os.environ['DARKPIPE_AUDIT_DIR']=str(out)
  cmd=[sys.executable,'-u',str(OUT/'scripts/instrumented_box.py'),'--video_path',input_url,
       '--gpu_ids','0','--ckpt_dir',os.environ['BOX_WEIGHTS'],
       '--clip_dir',str(out/'clips'),'--session_json',str(out/'session.json'),
       '--serve_port',str(hp),'--extra_hosts','','--stream_formats','mjpeg,flv,hls',
       '--rtmp_push_url',push,'--sr','off']
  service=launch(cmd,'operator.log');startup=time.monotonic()
  while True:
   assert service.poll()is None,'service startup failed'
   if get_health().get('health',{}).get('status')=='ok':break
   if time.monotonic()-startup>180:raise TimeoutError('startup')
   time.sleep(.5)
  origin=time.monotonic();receivers.append(Receiver(recv,out));done=set()
  schedule=[(15,'source_pause'),(23,'source_resume'),(40,'output_down'),(48,'output_up'),(65,'source_down'),(73,'source_up'),(90,'viewer_reconnect')]
  save(out/'protocol.json',dict(command=cmd,origin=origin,schedule=schedule,seconds=110,markers=False,scope='Deliberate network faults; do not include in clean throughput numbers'))
  while time.monotonic()-origin<110:
   assert service.poll()is None,'service unexpectedly exited'
   elapsed=time.monotonic()-origin
   for second,action in schedule:
    if elapsed<second or action in done:continue
    if action=='source_pause':source.send_signal(signal.SIGSTOP)
    elif action=='source_resume':source.send_signal(signal.SIGCONT)
    elif action=='output_down':terminate(sink)
    elif action=='output_up':sink=relay('output-relay',op,rp)
    elif action=='source_down':terminate(source)
    elif action=='source_up':source=publisher()
    elif action=='viewer_reconnect':receivers[-1].close();receivers.append(Receiver(recv,out))
    done.add(action);faults.append(dict(time=time.monotonic(),elapsed=time.monotonic()-origin,action=action));save(out/'faults.json',faults);print('FAULT',action,flush=True)
   health.append(get_health());save(out/'health-history.json',health);time.sleep(.5)
  service.send_signal(signal.SIGTERM);service.wait(timeout=90);assert service.returncode==0
  session=json.loads((out/'session.json').read_text());assert session['status']=='stopped'
  save(out/'complete.json',dict(normal_stop=True,service_returncode=0,session_status=session['status']))
 finally:
  if source and source.poll()is None:source.send_signal(signal.SIGCONT)
  if service and service.poll()is None:terminate(service)
  for r in receivers:r.close()
  for proc in reversed(owned):terminate(proc)
  for log in logs:log.close()
  save(out/'receivers.json',[dict(url=r.url,frames=r.rows,errors=r.errors)for r in receivers]);save(out/'cleanup.json',dict(owned_processes_alive=sum(p.poll()is None for p in owned)))
 print('FAULT RUN COMPLETE',flush=True)
if __name__=='__main__':main()
