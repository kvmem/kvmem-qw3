"""Reuse the timestamped read-only VNC recorder for an owned Linux VM pilot."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


class LinuxRecording:
    def __init__(self, directory, container_id, name='qw3-computer-linux'):
        self.directory=Path(directory)
        self.container_id=container_id
        self.name=name
        self.worker=None
        self.collected=False
        self.receipt=None

    def inspect(self):
        info=json.loads(subprocess.check_output(['docker','inspect',self.name],text=True))[0]
        if info['Id']!=self.container_id or info['Config'].get('Labels',{}).get('qw3.goal')!='cross-device-20261003':
            raise ValueError('Linux recording container identity changed')
        if not info['State']['Running']:
            raise ValueError('Linux recording container is not running')
        return info

    def start(self):
        import imageio_ffmpeg
        info=self.inspect()
        address=info['NetworkSettings']['Networks']['bridge']['IPAddress']
        if not address:raise ValueError('Owned VM has no bridge console address')
        command=[sys.executable,'-m','vm.record_console','--output',str(self.directory/'video'),
                 '--ffmpeg',imageio_ffmpeg.get_ffmpeg_exe(),'--host',address,'--port','5900','--fps','2']
        log=(self.directory/'recorder.log').open('xb')
        try:
            self.worker=subprocess.Popen(command,env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT)
        finally:
            log.close()
        (self.directory/'recorder-launch.json').write_text(json.dumps({'kind':'owned-read-only-vnc-recorder',
            'container_id':self.container_id,'pid':self.worker.pid,'command':command,'started_at':time.time(),
            'input_events':0,'executor_observe_calls':0},indent=2)+'\n')
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            p=self.directory/'video/recording.json'
            if p.exists():
                current=json.loads(p.read_text())
                if current['status']=='recording':
                    self.inspect()
                    return
            if self.worker.poll() is not None:
                raise RuntimeError('Linux VNC recorder exited during startup; see recorder.log')
            time.sleep(.1)
        raise TimeoutError('Linux VNC recorder readiness deadline')

    def finish(self,name='continuous.mp4'):
        if self.collected:return self.receipt
        if name not in {'continuous.mp4','interrupted.mp4'}:raise ValueError('Unsupported recording output')
        if self.worker is None:raise ValueError('Linux recording was not started')
        if self.worker.poll() is None:
            # Only the Popen child owned by this instance. A timeout does not
            # start another recorder or send a stop command to the VM service.
            self.worker.terminate()
        self.worker.wait(timeout=110)
        self.receipt=json.loads((self.directory/'video/recording.json').read_text())
        self.collected=True
        if self.worker.returncode!=0 or self.receipt.get('complete') is not True:
            raise RuntimeError('Linux VNC recording not complete; original artifacts retained')
        source=self.directory/'video/screen.mp4'
        import hashlib
        if hashlib.sha256(source.read_bytes()).hexdigest()!=self.receipt['artifacts']['screen.mp4']['sha256']:
            raise ValueError('Linux recorded video differs from its terminal receipt')
        target=self.directory/name
        if target.exists():raise ValueError('Refuse to overwrite another pilot recording')
        shutil.copyfile(source,target)
        return self.receipt
