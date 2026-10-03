"""Live device captures with measured timestamps, never historical reconstruction."""
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import subprocess
import threading
import time

from PIL import Image


class LiveAndroidRecording:
    def __init__(self,adb,directory):
        import av
        self.av=av
        self.adb=list(adb)
        self.directory=Path(directory)
        self.stop=threading.Event()
        self.ready=threading.Event()
        self.worker=None
        self.collected=False
        self.error=None
        self.receipt={'kind':'live-device-sampled-video','source':'ADB screencap PNG','sample_hz':1,
            'historical_reconstruction':False,'timestamps':'Measured host capture midpoint; see frame capture bounds',
            'continuous_capture_loop':True,'video_file':'continuous.mp4','frames':0}

    def save(self):
        (self.directory/'recording.json').write_text(json.dumps(self.receipt,indent=2)+'\n')

    def capture(self):
        began=time.monotonic();wall=time.time()
        result=subprocess.run([*self.adb,'exec-out','screencap','-p'],check=True,capture_output=True,timeout=15)
        ended=time.monotonic()
        with Image.open(io.BytesIO(result.stdout)) as image:
            if image.format!='PNG':raise ValueError('Capture was not a PNG')
            pixels=image.convert('RGB')
        return pixels,(began+ended)/2,{'request_started_at':wall,'capture_seconds':ended-began,
            'sha256':hashlib.sha256(result.stdout).hexdigest(),'bytes':len(result.stdout)}

    def loop(self):
        container=None
        try:
            container=self.av.open(str(self.directory/'continuous.mp4'),'w')
            stream=container.add_stream('libx264',rate=1)
            stream.pix_fmt='yuv420p';stream.time_base=Fraction(1,1000)
            stream.codec_context.time_base=Fraction(1,1000)
            stream.options={'preset':'ultrafast','crf':'20','tune':'zerolatency'}
            with (self.directory/'frame-captures.jsonl').open('x') as log:
                while True:
                    last=self.stop.is_set()
                    image,stamp,metadata=self.capture()
                    if self.receipt['frames']==0:
                        stream.width,stream.height=image.size
                        self.started=stamp
                        self.receipt.update(started_at=metadata['request_started_at']+metadata['capture_seconds']/2,
                            width=image.width,height=image.height)
                    frame=self.av.VideoFrame.from_image(image)
                    frame.pts=round((stamp-self.started)*1000);frame.time_base=Fraction(1,1000)
                    for packet in stream.encode(frame):container.mux(packet)
                    metadata.update(video_timestamp_seconds=frame.pts/1000,last_capture=last)
                    log.write(json.dumps(metadata)+'\n');log.flush()
                    self.receipt['frames']+=1;self.receipt['last_capture_seconds']=frame.pts/1000
                    self.ready.set()
                    if last:break
                    self.stop.wait(max(0,1-(time.monotonic()-stamp)))
            for packet in stream.encode():container.mux(packet)
        except BaseException as exc:
            self.error=type(exc).__name__+': '+str(exc)
            self.receipt['error']=self.error
        finally:
            if container:
                try:container.close()
                except Exception as exc:
                    self.error=type(exc).__name__+': '+str(exc)
                    self.receipt['error']=self.error
            self.receipt.update(finished_at=time.time(),passed=self.error is None)
            self.save();self.ready.set()

    def start(self):
        self.worker=threading.Thread(target=self.loop,name='qw3-live-device-recording',daemon=True)
        self.worker.start()
        if not self.ready.wait(20) or self.error:
            self.stop.set()
            raise RuntimeError(self.error or 'Live recording readiness deadline')

    def finish(self,name='continuous.mp4'):
        if self.collected:return self.receipt
        if self.worker is None:raise ValueError('Live capture was not started')
        self.stop.set();self.worker.join(timeout=30)
        if self.worker.is_alive():raise TimeoutError('Live capture did not finish; no second recorder started')
        self.collected=True
        path=self.directory/'continuous.mp4'
        if name!='continuous.mp4' and path.exists():
            path.rename(self.directory/name)
            self.receipt['video_file']=name
        if (self.directory/name).exists():self.receipt['video_bytes']=(self.directory/name).stat().st_size
        self.save()
        return self.receipt
