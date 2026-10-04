"""Read native Chrome tab provenance in a fresh, separately owned Linux VM.

This is an administrative observation probe, not a QW3/business task. It does
not change the input guard, a running model package or an existing device.
"""
import argparse
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
from osworld_linux import environment
from osworld_windows.source import ROOT
from qw3_computer.device_lease import DeviceLease
import kvmem_osworld_queue as owned


PORTS=dict(server=18950,chromium=18951,vnc=18952,vlc=18953)


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def chrome_command_matches_launch(actual,declared,executable):
    """Only exact own launch args, including Chrome's observed argv0 packing.

    This is startup attribution. After registration the complete actual raw
    /proc command is compared literally, without any normalization.
    """
    from pathlib import Path
    if (not isinstance(actual,list) or not isinstance(declared,list) or len(declared)<2
            or any(not isinstance(v,str) or not v or any(c.isspace() for c in v) for v in declared)
            or not isinstance(executable,str) or not executable.startswith('/')
            or Path(executable).name!='chrome' or Path(declared[0]).name!='google-chrome'):
        return False
    aliases=[declared,[executable,*declared[1:]]]
    return any(actual==parts or actual==[' '.join(parts)] for parts in aliases)


def guest(code,seconds=8):
    with requests.Session() as session:
        session.trust_env=False
        response=session.post(f'http://127.0.0.1:{PORTS["server"]}/execute',
            json=dict(command=['python','-c',code],shell=False,timeout=seconds),timeout=seconds+3)
    response.raise_for_status();result=response.json()
    if result.get('returncode',0)!=0 or result.get('status')=='error':
        raise RuntimeError('Own browser source probe failed: '+str(result.get('error'))+' '+str(result.get('output',''))[-2000:])
    return json.loads(result['output'])


def inspect_vm(name,cid):
    info=json.loads(subprocess.check_output(['docker','inspect',name],text=True,timeout=3))[0]
    if (info['Id']!=cid or info['Name']!='/'+name or info['Image']!=environment.IMAGE
            or info['Config'].get('Labels',{}).get('qw3.goal')!='cross-device-20261003'
            or not info['State']['Running']):raise RuntimeError('Browser probe VM identity changed')
    return info


def native_snapshot(expected,title,*,startup=False):
    # Names and text are retained verbatim. The original server reader is not
    # edited. Parent roles come from native objects rather than HTML strings.
    return guest(inspect.getsource(chrome_command_matches_launch)+f'''\nimport json,os,subprocess,sys,time
from pathlib import Path
expected={expected!r};process=Path('/proc')/str(expected['pid'])
assert process.stat().st_uid==expected['uid']
assert int((process/'stat').read_text().split(') ',1)[1].split()[19])==expected['start_ticks']
actual_command=[v.decode() for v in (process/'cmdline').read_bytes().split(b'\\0') if v]
assert os.readlink(process/'exe')==expected['executable']
assert chrome_command_matches_launch(actual_command,expected['declared_command'],expected['executable']),json.dumps({{'expected':expected,'actual_command':actual_command}})
actual_cwd=os.readlink(process/'cwd')
if not {startup!r}:
 assert actual_command==expected['command'],json.dumps({{'expected':expected,'actual_command':actual_command}})
 assert actual_cwd==expected['cwd']
sys.path.insert(0,'/home/user/server')
from src import accessibility as a
frames=[]
for app in a.pyatspi.Registry.getDesktop(0):
 for frame in app:
  if frame.getRoleName()=='frame' and frame.name=={title!r}:
   assert frame.get_process_id()==expected['pid'],json.dumps({{'expected_pid':expected['pid'],'native_window_pid':frame.get_process_id()}})
   frames.append(frame)
tabs=[];visits=0
def walk(node,parents):
 global visits
 visits+=1
 if visits>512 or len(parents)>16:raise RuntimeError('Native browser source traversal exceeded its declared bounds')
 role=node.getRoleName();state=node.getState()
 if role=='page tab':
  try:text=node.queryText().getText(0,-1)
  except Exception:text=None
  tabs.append(dict(role=role,name=node.name,description=node.description,text=text,
                   native_bus_name=node.app.bus_name,object_path=node.path,
                   owner_pid=node.get_process_id(),selected=state.contains(a.pyatspi.STATE_SELECTED),
                   visible=state.contains(a.pyatspi.STATE_VISIBLE),showing=state.contains(a.pyatspi.STATE_SHOWING),
                   enabled=state.contains(a.pyatspi.STATE_ENABLED),parent_roles=parents,
                   attributes=node.get_attributes(),children=[dict(role=c.getRoleName(),name=c.name) for c in node]))
 if role.startswith('document') or role in ('table','table cell'):return
 for child in node:walk(child,parents+[role])
if len(frames)==1:walk(frames[0],[])
version=subprocess.run(['google-chrome','--version'],capture_output=True,text=True,timeout=2)
print(json.dumps(dict(captured_at=time.time(),chrome_version=version.stdout.strip(),frames=len(frames),
                     browser_identity=expected|{{'command':actual_command,'cwd':actual_cwd}},startup_read={startup!r},
                     frame_title=frames[0].name if len(frames)==1 else None,
                     frame_path=frames[0].path if len(frames)==1 else None,
                     frame_bus_name=frames[0].app.bus_name if len(frames)==1 else None,
                     visits=visits,tabs=tabs)))''')


def run(args):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    nonce=uuid.uuid4().hex;name='qw3-native-tab-source-'+nonce[:12]
    state=dict(kind='actual_native_chrome_tab_source_probe',status='preparing',
        producer=owned.process_identity(os.getpid()),started_at=time.time(),recipe_sha256=digest(__file__),
        model_calls=0,calibrated_key_mouse_actions=0,existing_device_inputs=0,
        administrative_browser_launch=True,business_success_claimed=False,snapshots=[],startup_snapshots=[])
    def save(**values):
        state.update(values);p=output/'state.tmp';p.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n');p.replace(output/'state.json')
    save();env=lease=None;cid=None
    try:
        for port in PORTS.values():
            with socket.socket() as sock:sock.bind(('127.0.0.1',port))
        lease=DeviceLease(f'http://127.0.0.1:{PORTS["server"]}').acquire()
        save(status='booting-dedicated-vm',container_name=name,ports=PORTS)
        with patch.object(environment,'PORTS',PORTS):env=environment.create(output,name)
        cid=json.loads((output/'vm.json').read_text())['container_id'];inspect_vm(name,cid);save(container_id=cid)
        title='QW3 Native Tab Source '+nonce
        html='<!doctype html><meta charset="utf-8"><title>'+title+'</title><body tabindex="-1"><h1>Native tab source qualification</h1><p>The visible task text stays fixed.</p><script>document.body.focus();setTimeout(()=>{window.qw3Memory=Array.from({length:48},()=>new Uint8Array(1024*1024).fill(7));},15000);</script></body>'
        directory='/home/user/.qw3-native-tab-source-'+nonce
        browser=guest(inspect.getsource(chrome_command_matches_launch)+f'''\nimport json,os,shutil,subprocess,time
from pathlib import Path
p=Path({directory!r});p.mkdir(exist_ok=False);page=p/'fixture.html';page.write_text({html!r})
log=(p/'chrome.log').open('xb');profile=str(p/'profile')
declared=[shutil.which('google-chrome'),'--no-first-run','--no-default-browser-check','--disable-background-networking','--disable-gpu','--force-renderer-accessibility','--user-data-dir='+profile,'--new-window',page.as_uri()]
child=subprocess.Popen(declared,cwd=p,stdout=log,stderr=log)
deadline=time.monotonic()+5
while True:
 matches=[];candidate=Path('/proc')/str(child.pid)
 try:
  command=[v.decode() for v in (candidate/'cmdline').read_bytes().split(b'\\0') if v]
  executable=os.readlink(candidate/'exe')
  if chrome_command_matches_launch(command,declared,executable):
   stat=(candidate/'stat').read_text().split(') ',1)[1].split()
   matches.append(dict(pid=child.pid,uid=candidate.stat().st_uid,start_ticks=int(stat[19]),command=command,cwd=os.readlink(candidate/'cwd'),
                       executable=executable,declared_command=declared,profile_argument='--user-data-dir='+profile))
 except (OSError,UnicodeDecodeError):pass
 if len(matches)==1:break
 if child.poll() is not None:raise RuntimeError('Own spawned Chrome process exited before registration')
 if time.monotonic()>=deadline:raise RuntimeError('Own spawned Chrome process identity not established')
 time.sleep(.1)
print(json.dumps(matches[0]))''')
        save(status='waiting-for-native-browser-ready',browser_startup_seed=browser,fixture_title=title,fixture_html_sha256=hashlib.sha256(html.encode()).hexdigest())
        # Chrome can rewrite argv during initialization. Register the complete
        # command only after the same own generation has a native window and
        # the complete command is stable across two consecutive snapshots.
        ready_deadline=time.monotonic()+30;previous_identity=None
        while time.monotonic()<ready_deadline:
            inspect_vm(name,cid);snapshot=native_snapshot(browser,title+' - Google Chrome',startup=True)
            state['startup_snapshots'].append(snapshot);save()
            selected=[tab for tab in snapshot['tabs'] if tab['selected']]
            if snapshot['frames']==1 and len(selected)==1 and snapshot['browser_identity']==previous_identity:
                browser=snapshot['browser_identity'];break
            previous_identity=snapshot['browser_identity'];time.sleep(.5)
        else:raise TimeoutError('Own native Chrome frame and stable full command not established during startup')
        save(status='reading-native-tab-source',browser_identity=browser,native_startup_registration_complete=True)
        began=time.monotonic()
        while time.monotonic()-began<65:
            inspect_vm(name,cid);snapshot=native_snapshot(browser,title+' - Google Chrome')
            state['snapshots'].append(snapshot);save(elapsed_seconds=time.monotonic()-began)
            if len(state['snapshots'])>=3 and time.monotonic()-began>=45:break
            time.sleep(2)
        tabs=[tab for snapshot in state['snapshots'] for tab in snapshot['tabs']]
        checks=dict(native_browser_owner_verified=bool(tabs) and all(t['owner_pid']==browser['pid'] for t in tabs),
                    selected_tab_observed=any(t['selected'] for t in tabs),
                    selected_native_tab_parentage_retained=any(t['selected'] and 'page tab list' in t['parent_roles'] for t in tabs),
                    actual_resource_name_observed=any(' - Memory usage - ' in t['name'] for t in tabs),
                    observations_only_not_input_guard_change=state['calibrated_key_mouse_actions']==0 and state['business_success_claimed'] is False)
        save(status='passed' if all(checks.values()) else 'failed',passed=all(checks.values()),checks=checks,finished_at=time.time())
    except BaseException as error:
        save(status='failed',passed=False,error_type=type(error).__name__,error=str(error),finished_at=time.time());raise
    finally:
        if env is not None:
            try:
                inspect_vm(name,cid);env.close()
                retired=subprocess.run(['docker','inspect',cid],capture_output=True,timeout=5).returncode!=0
                save(dedicated_vm_retired=retired)
            except Exception as error:save(dedicated_vm_retired=False,cleanup_error=str(error))
        if lease is not None:lease.close()
    if state.get('dedicated_vm_retired') is not True:raise RuntimeError('Own browser probe VM retirement not confirmed')
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    result=run(parser.parse_args());print(json.dumps({k:result.get(k) for k in ['status','checks','dedicated_vm_retired']}))
    if not result.get('passed'):raise SystemExit(1)


if __name__=='__main__':main()
