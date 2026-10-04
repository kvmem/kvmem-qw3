"""Qualify a recorder during native GUI changes in a dedicated original VM.

The Tk app is a calibration fixture, not a business task or a QW3 benchmark.
No existing VM, input owner, model engine or frozen trial source is changed.
"""
import argparse
import base64
import contextlib
import hashlib
import inspect
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from unittest.mock import patch
import uuid

import requests
import recoverable_linux_recording as recording
from osworld_linux import environment
from osworld_windows.source import ROOT
from qw3_computer.device_lease import DeviceLease


PORTS=dict(server=18940,chromium=18941,vnc=18942,vlc=18943)


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def valid_rectangle(rectangle,width,height):
    return (isinstance(rectangle,list) and len(rectangle)==4 and all(type(v) is int for v in rectangle)
            and rectangle[0]>=0 and rectangle[1]>=0 and rectangle[2]>0 and rectangle[3]>0
            and rectangle[0]+rectangle[2]<=width and rectangle[1]+rectangle[3]<=height)


def marker_matches(pixels,rectangle,color):
    import numpy as np
    if not valid_rectangle(rectangle,pixels.shape[1],pixels.shape[0]):return False
    x,y,width,height=rectangle
    tile=pixels[y:y+height,x:x+width,:3].astype('int16')
    return bool((np.abs(tile-np.asarray(color,dtype='int16')).max(axis=2)<=15).mean()>=.95)


def locate_marker(pixels,bounds,color):
    """Locate the fixture's unique solid marker inside its owned X11 window."""
    import numpy as np
    if not valid_rectangle(bounds,pixels.shape[1],pixels.shape[0]):raise ValueError('Invalid owned window bounds')
    x,y,width,height=bounds;region=pixels[y:y+height,x:x+width,:3]
    mask=(np.abs(region.astype('int16')-np.asarray(color,dtype='int16')).max(axis=2)==0)
    rows,columns=np.nonzero(mask)
    if not len(rows):raise ValueError('Owned native marker is not visible')
    left,right,top,bottom=int(columns.min()),int(columns.max()),int(rows.min()),int(rows.max())
    if not (50<=right-left+1<=70 and 50<=bottom-top+1<=70):raise ValueError('Native marker is small, occluded or ambiguous')
    rectangle=[x+left+10,y+top+10,right-left-19,bottom-top-19]
    if not marker_matches(pixels,rectangle,color):raise ValueError('Native marker is not positively established')
    return rectangle


def guest(code):
    with requests.Session() as session:
        session.trust_env=False
        response=session.post(f'http://127.0.0.1:{PORTS["server"]}/execute',
                              json=dict(command=['python','-c',code],shell=False),timeout=30)
    response.raise_for_status();result=response.json()
    if result.get('returncode',0)!=0 or result.get('status')=='error':
        raise RuntimeError('Dedicated calibration guest operation failed: '+str(result.get('error'))+' '+str(result.get('output',''))[-2000:])
    return json.loads(result['output'])


def owned_vm(name,cid):
    information=json.loads(subprocess.check_output(['docker','inspect',name],text=True,timeout=3))
    if (len(information)!=1 or information[0]['Id']!=cid or information[0]['Name']!='/'+name
            or information[0]['Config'].get('Labels',{}).get('qw3.goal')!='cross-device-20261003'
            or information[0]['Image']!=environment.IMAGE or not information[0]['State']['Running']):
        raise RuntimeError('Dedicated native VM identity changed')
    return information[0]


def execute_input(directory,identity,nonce,index):
    kind=('key','click','scroll')[index%3]
    # Require the same process generation and a positively visible fixture
    # marker before input. No unresolved operation is retried on a new identity.
    helpers='\n'.join(inspect.getsource(f) for f in (valid_rectangle,marker_matches,locate_marker))
    code=helpers+f'''\nimport json,os,subprocess,time
from pathlib import Path
import pyautogui
import numpy as np
from PIL import ImageGrab
p=Path({str(directory)!r});s=json.loads((p/'state.json').read_text());expected={identity!r}
process=Path('/proc')/str(expected['pid']);stat=(process/'stat').read_text().split(') ',1)[1].split()
assert process.stat().st_uid==expected['uid'] and int(stat[19])==expected['start_ticks']
assert (process/'cmdline').read_bytes().split(b'\\0')[:len(expected['command'])]==[v.encode() for v in expected['command']]
assert os.readlink(process/'cwd')==expected['cwd']
assert s['nonce']=={nonce!r} and s['phase']=={index}
ids=subprocess.check_output(['xdotool','search','--name','^'+s['title']+'$'],text=True).split()
assert len(ids)==1
subprocess.run(['xdotool','windowraise',ids[0]],check=True,timeout=5)
subprocess.run(['xdotool','windowfocus','--sync',ids[0]],check=True,timeout=5)
assert subprocess.check_output(['xdotool','getwindowfocus'],text=True).strip()==ids[0]
geometry=dict(line.split('=',1) for line in subprocess.check_output(['xdotool','getwindowgeometry','--shell',ids[0]],text=True).splitlines())
bounds=[int(geometry[k]) for k in ['X','Y','WIDTH','HEIGHT']]
image=ImageGrab.grab(xdisplay=os.environ.get('DISPLAY',':0'))
rectangle=locate_marker(np.asarray(image),bounds,s['color']);x,y,w,h=rectangle
if {kind!r}=='key': pyautogui.press('space')
elif {kind!r}=='click': pyautogui.click(x+w//2,y+h//2)
else: pyautogui.moveTo(x+w//2,y+h//2);pyautogui.scroll(1)
deadline=time.monotonic()+5
matched=False
while time.monotonic()<deadline:
 s=json.loads((p/'state.json').read_text())
 if s['phase']=={index+1}:
  geometry=dict(line.split('=',1) for line in subprocess.check_output(['xdotool','getwindowgeometry','--shell',ids[0]],text=True).splitlines())
  bounds=[int(geometry[k]) for k in ['X','Y','WIDTH','HEIGHT']]
  try: rectangle=locate_marker(np.asarray(ImageGrab.grab(xdisplay=os.environ.get('DISPLAY',':0'))),bounds,s['color']);matched=True;break
  except ValueError:pass
 time.sleep(.05)
assert matched and s['phase']=={index+1} and s['native_input_events'][-1]['kind']=={kind!r}
s['tk_reported_rectangle']=s['rectangle'];s['rectangle']=rectangle;s['window_bounds']=bounds
print(json.dumps(s))'''
    return guest(code)


def run(args):
    directory=args.output.resolve();directory.mkdir(parents=True,exist_ok=False)
    nonce=uuid.uuid4().hex;name='qw3-recording-active-'+nonce[:12]
    record=dict(kind='native_active_calibration_recording_qualification',mode=args.mode,status='preparing',
                started_at=time.time(),producer=recording.owned.process_identity(os.getpid()),
                recipe_sha256=digest(__file__),model_calls=0,qw3_task_success_claimed=False,
                scope='Dedicated original Linux VM, controlled native Tk calibration GUI; not a model business task.',
                native_1800_second_active_calibration_sampling_proven=False,inputs=[])
    def checkpoint(**values):
        record.update(values);temporary=directory/'state.tmp';temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(directory/'state.json')
    checkpoint()
    env=wrapper=lease=None;cid=None
    try:
        qualification=json.loads(args.qualification.read_text())
        if (qualification.get('passed') is not True or qualification.get('tests_run',0)<9
                or qualification.get('model_calls')!=0 or qualification.get('native_device_inputs')!=0):
            raise ValueError('Require the completed real subprocess/RFB/codec wrapper qualification')
        for relative,sha in qualification['source_sha256'].items():
            if digest(Path(recording.__file__).parent/relative)!=sha:raise ValueError('Qualified recorder source changed')
        record['source_sha256']=qualification['source_sha256']
        # Docker does not replace colliding containers. Check the dedicated
        # host ports before booting; the lease never targets an existing VM.
        with contextlib.ExitStack() as stack:
            for port in PORTS.values():
                sock=stack.enter_context(socket.socket());sock.bind(('127.0.0.1',port))
        lease=DeviceLease(f'http://127.0.0.1:{PORTS["server"]}').acquire()
        checkpoint(status='booting-dedicated-vm',container_name=name,ports=PORTS,
                   environment_source_sha256=digest(Path(environment.__file__)))
        with patch.object(environment,'PORTS',PORTS):env=environment.create(directory,name)
        vm=json.loads((directory/'vm.json').read_text());cid=vm['container_id'];info=owned_vm(name,cid)
        checkpoint(container_id=cid,container_started_at=info['State']['StartedAt'],status='starting-recorder')
        duration=60 if args.mode=='smoke' else 1800
        wrapper=recording.RecoverableLinuxRecording(directory,cid,name=name,
                    profile=recording.RecordingProfile(duration_seconds=duration+30))
        launch=wrapper.start();checkpoint(recorder=launch,status='launching-native-gui')
        fixture=Path(__file__).with_name('active_recording_fixture.py');data=fixture.read_bytes()
        guest_root='/home/user/.qw3-recording-calibration-'+nonce;state_root=guest_root+'/evidence'
        code=f'''from pathlib import Path
import base64,json,os,subprocess,time
p=Path({guest_root!r});p.mkdir(exist_ok=False);source=p/'fixture.py';source.write_bytes(base64.b64decode({base64.b64encode(data).decode()!r}))
log=(p/'fixture.log').open('xb');child=subprocess.Popen(['python',str(source),'--output',{state_root!r},'--nonce',{nonce!r}],cwd=p,stdout=log,stderr=log)
deadline=time.monotonic()+10
while not (Path({state_root!r})/'state.json').exists():
 assert child.poll() is None
 if time.monotonic()>=deadline:raise TimeoutError('Native fixture readiness deadline; original process preserved')
 time.sleep(.1)
print((Path({state_root!r})/'state.json').read_text())'''
        first=guest(code);identity=first['identity'];assert first['nonce']==nonce and first['phase']==0
        # Python's argv differs from /proc (which includes the interpreter).
        identity=guest(f"from pathlib import Path;import json,os;p=Path('/proc/{identity['pid']}');s=json.loads((Path({state_root!r})/'state.json').read_text());s['identity']['command']=[v.decode() for v in (p/'cmdline').read_bytes().split(b'\\0') if v];print(json.dumps(s['identity']))")
        checkpoint(status='recording-active-native-gui',fixture_identity=identity,fixture_nonce=nonce,
                   fixture_source_sha256=hashlib.sha256(data).hexdigest(),native_initial_state=first)
        started=time.monotonic();index=0;next_input=started+3
        while time.monotonic()-started<duration:
            owned_vm(name,cid)
            if not wrapper.worker_live():raise RuntimeError('Recorder ended while native GUI task was still active')
            now=time.monotonic()
            if now>=next_input:
                checkpoint(input_in_flight=dict(index=index,host_started_monotonic_ns=time.monotonic_ns()))
                result=execute_input(Path(state_root),identity,nonce,index)
                event=dict(index=index,phase=result['phase'],kind=result['native_input_events'][-1]['kind'],
                           color=result['color'],rectangle=result['rectangle'],tk_reported_rectangle=result['tk_reported_rectangle'],
                           window_bounds=result['window_bounds'],host_acknowledged_monotonic_ns=time.monotonic_ns())
                record['inputs'].append(event);index+=1;next_input=now+15
                checkpoint(input_in_flight=None,native_input_events=index)
            checkpoint(observed_active_seconds=time.monotonic()-started)
            time.sleep(.5)
        receipt=wrapper.finish();checkpoint(status='verifying-native-video',recording_complete=receipt['encoding_complete'])
        import av
        witnessed=set();frames=0
        with av.open(str(directory/'continuous.mp4')) as video:
            video.streams.video[0].codec_context.thread_count=2
            for frame in video.decode(video=0):
                frames+=1;elapsed=float(frame.time or 0)
                relevant=[e for e in record['inputs'] if e['index'] not in witnessed
                          and 0<elapsed-(e['host_acknowledged_monotonic_ns']-receipt['started_monotonic_ns'])/1e9<5]
                if relevant:
                    pixels=frame.to_ndarray(format='rgb24')
                    for event in relevant:
                        if marker_matches(pixels,event['rectangle'],event['color']):witnessed.add(event['index'])
        geometry=[json.loads(line) for line in (directory/'video/framebuffer-geometry.jsonl').read_text().splitlines()]
        checks=dict(real_inputs_observed_by_native_app=index>=4,
                    key_click_and_scroll_all_exercised={e['kind'] for e in record['inputs']}=={'key','click','scroll'},
                    every_input_result_seen_in_decoded_video=len(witnessed)==index,
                    every_video_frame_decodes=frames==receipt['video_frames'],
                    full_framebuffers_without_scaling=all(e['full_framebuffer_received'] and e['pixel_scaling'] is False for e in geometry),
                    declared_active_duration_covered=receipt['video_duration_seconds']>=duration,
                    owned_recorder_retired=wrapper.worker.poll()==0 and recording.owned.process_identity(wrapper.worker.pid) is None,
                    encoding_and_recovery_evidence_verified=receipt['encoding_complete'] is True,
                    source_pins_unchanged=all(digest(Path(recording.__file__).parent/k)==v for k,v in record['source_sha256'].items()),
                    fixture_result_not_promoted_to_qw3_business_success=record['qw3_task_success_claimed'] is False)
        checkpoint(passed=all(checks.values()),status='passed' if all(checks.values()) else 'failed',checks=checks,
                   decoded_frames=frames,video_seconds=receipt['video_duration_seconds'],witnessed_input_results=len(witnessed),
                   video_sha256=digest(directory/'continuous.mp4'),fresh_sampling_uninterrupted=receipt['fresh_sampling_uninterrupted'],
                   reconnects=receipt['reconnects'],transport_gaps=receipt['transport_gaps'],finished_at=time.time(),
                   native_1800_second_active_calibration_sampling_proven=args.mode=='stability' and all(checks.values()))
        if not record['passed']:raise RuntimeError('Active native calibration qualification failed; original evidence retained')
    except BaseException as error:
        checkpoint(status='failed',passed=False,error_type=type(error).__name__,error=str(error),finished_at=time.time())
        raise
    finally:
        if wrapper is not None and wrapper.identity is not None and not wrapper.collected:
            try:wrapper.stop_owned_worker();checkpoint(owned_worker_retired_after_failure=wrapper.worker.poll() is not None)
            except Exception as error:checkpoint(worker_cleanup_error=str(error))
        if env is not None:
            try:
                owned_vm(name,cid);env.close()
                remaining=subprocess.run(['docker','inspect',cid],capture_output=True,text=True,timeout=5)
                checkpoint(dedicated_vm_retired=remaining.returncode!=0)
            except Exception as error:checkpoint(dedicated_vm_retired=False,vm_cleanup_error=str(error))
        if lease is not None:lease.close()
    if record.get('passed') is True and record.get('dedicated_vm_retired') is not True:
        checkpoint(status='failed',passed=False,error_type='RuntimeError',error='Dedicated VM retirement unconfirmed')
        raise RuntimeError('Native qualification cannot pass with unconfirmed dedicated VM retirement')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['smoke','stability'],required=True)
    parser.add_argument('--qualification',type=Path,required=True)
    result=run(parser.parse_args())
    print(json.dumps({k:result.get(k) for k in ['passed','mode','native_input_events','decoded_frames','video_seconds','dedicated_vm_retired']}))


if __name__=='__main__':main()
