"""Own one native unlimited Android recording; signal only its exact process."""
import json
from pathlib import Path
import subprocess
import time
import uuid


class AndroidRecording:
    def __init__(self, adb, directory):
        self.adb=list(adb)
        self.directory=Path(directory)
        self.remote='/sdcard/qw3-record-'+uuid.uuid4().hex+'.mp4'
        self.command=['screenrecord','--bugreport','--bit-rate','2000000','--time-limit','0',self.remote]
        self.process=None
        self.native_pid=None
        self.started=None
        self.collected=False

    def adb_run(self, arguments, **kwargs):
        return subprocess.run([*self.adb,*arguments],check=True,timeout=30,**kwargs)

    def matches(self,pid):
        try:
            raw=self.adb_run(['exec-out','cat',f'/proc/{pid}/cmdline'],capture_output=True).stdout
            return [s.decode() for s in raw.split(b'\0') if s]==self.command
        except subprocess.CalledProcessError:
            return False

    def start(self):
        help_result=self.adb_run(['shell','screenrecord','--help'],capture_output=True,text=True)
        help_text=help_result.stdout+help_result.stderr
        if 'Set to 0' not in help_text or 'remove the time limit' not in help_text:
            raise ValueError('Native recorder does not document unlimited recording')
        (self.directory/'native-recorder-help.txt').write_text(help_text)
        with (self.directory/'recording.log').open('xb') as log:
            self.process=subprocess.Popen([*self.adb,'shell',*self.command],stdout=log,stderr=subprocess.STDOUT)
        self.started=time.monotonic()
        self.save({'status':'starting','started_at':time.time(),'host_adb_pid':self.process.pid,
                   'command':self.command,'continuous':True,'native_time_limit_seconds':0})
        deadline=self.started+10
        while time.monotonic()<deadline:
            if self.process.poll() is not None:
                raise RuntimeError('Owned recording exited during startup')
            result=subprocess.run([*self.adb,'shell','pidof','screenrecord'],capture_output=True,text=True,timeout=15)
            owned=[int(pid) for pid in result.stdout.split() if pid.isdigit() and self.matches(int(pid))]
            if len(owned)==1:
                self.native_pid=owned[0]
                break
            time.sleep(.2)
        if self.native_pid is None:
            raise TimeoutError('Cannot establish native recording process identity')
        self.save({'started_at':time.time(),'native_pid':self.native_pid,'host_adb_pid':self.process.pid,
                   'command':self.command,'continuous':True,'native_time_limit_seconds':0})

    def save(self,value):
        (self.directory/'recording.json').write_text(json.dumps(value,indent=2)+'\n')

    def finish(self,name='continuous.mp4'):
        if self.collected:
            return json.loads((self.directory/'recording.json').read_text())
        if self.process is None or self.native_pid is None:
            raise ValueError('Recording ownership was never established')
        if self.process.poll() is None:
            if not self.matches(self.native_pid):
                raise ValueError('Recorder process identity changed; refusing signal')
            self.adb_run(['shell','kill','-2',str(self.native_pid)],capture_output=True)
            self.process.wait(timeout=30)
        with (self.directory/name).open('xb') as stream:
            self.adb_run(['exec-out','cat',self.remote],stdout=stream)
        receipt=json.loads((self.directory/'recording.json').read_text())
        receipt.update(finished_at=time.time(),seconds=time.monotonic()-self.started,
                       adb_exit_code=self.process.returncode,video_bytes=(self.directory/name).stat().st_size,
                       file=name,passed=self.process.returncode==0)
        self.save(receipt)
        self.collected=True
        return receipt
