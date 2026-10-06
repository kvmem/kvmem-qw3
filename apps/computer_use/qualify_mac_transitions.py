"""Live Mac transition qualification; no model calls or installer advancement.

Temporarily extends the existing test scope with fixed Terminal/Installer
aliases, restores the original config on exit, and has a remote expiry watchdog.
The current native binary is unchanged, preserving its granted permissions.
"""
import argparse
import asyncio
import contextlib
import json
from pathlib import Path
import secrets
import shlex
import time

from aiohttp import web

from qualify_macos_native import ssh
from macos_agent_driver import MacDriver
from macos_gateway_entry import create_mac_app
from qw3_computer.desktop import Desktop, DesktopError, DesktopRejected
from qw3_computer.device_gateway import Gateway
from qw3_computer.device_lease import DeviceLease
from qw3_computer.target_execution import TargetExecution


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def scope_script(connection, scope_id, restore=False, *, allowed_apps=None, window_prefixes=None):
    values = json.dumps({'runtime': connection['remote_runtime'], 'task': connection['remote_task_dir'],
                         'scope_id': scope_id, 'restore': restore,
                         'allowed_apps': allowed_apps or ['com.apple.TextEdit','com.apple.Terminal','com.apple.installer'],
                         'window_prefixes': window_prefixes or ['QW3-Mac-20261006-','sunny — QW3-Python-Install-20261006-01','Install Python','安装Python','安装 Python']})
    return """import json,os,signal,subprocess,time
from pathlib import Path
values=json.loads(""" + repr(values) + """)
base=Path(values['runtime']);task=Path(values['task'])/values['scope_id']
config_path=base/'config.json';status_path=base/'status.json'
config=json.loads(config_path.read_text());status=json.loads(status_path.read_text())
def restart():
    status=json.loads(status_path.read_text());pid=status['pid']
    assert status['paused'] and not status['owner_present'],'Native owner must be released before scope change'
    cmd=subprocess.check_output(['/bin/ps','-p',str(pid),'-o','command='],text=True)
    assert 'QW3 Mac Agent.app/Contents/MacOS/qw3-mac-agent' in cmd,'Native process identity changed'
    os.kill(pid,signal.SIGTERM)
    for _ in range(80):
        try:os.kill(pid,0)
        except ProcessLookupError:break
        time.sleep(.1)
    else:raise RuntimeError('Old native process did not exit')
    subprocess.run(['/usr/bin/open',str(Path.home()/'Applications/QW3 Mac Agent.app')],check=True)
    for _ in range(80):
        time.sleep(.1)
        fresh=json.loads(status_path.read_text())
        if fresh['pid']!=pid and fresh['listener_ready']:return fresh
    raise RuntimeError('Native restart not confirmed')
if values['restore']:
    if config.get('task_scope_id')==values['scope_id']:
        assert status['paused'] and not status['owner_present']
        config_path.write_bytes((task/'original-config.json').read_bytes())
        fresh=restart();(task/'scope-restored.json').write_text(json.dumps({'at':time.time()}))
        print(json.dumps({'restored':True,'paused':fresh['paused'],'permissions':fresh['ax_trusted'] and fresh['screen_capture_allowed']}))
    else:print(json.dumps({'restored':False,'reason':'Scope marker no longer belongs to this qualification'}))
else:
    assert 'task_scope_id' not in config,'Another temporary scope is active'
    assert status['paused'] and not status['owner_present'],'Existing native owner must be retired'
    assert status['ax_trusted'] and status['screen_capture_allowed'],'Native permissions are unavailable'
    task.mkdir(mode=0o700)
    (task/'original-config.json').write_bytes(config_path.read_bytes())
    expires=time.time()+1800
    record={'scope_id':values['scope_id'],'expires_at':expires}
    (task/'scope.json').write_text(json.dumps(record))
    watcher='''import json,os,signal,subprocess,time
from pathlib import Path
task=Path(__file__).parent;record=json.loads((task/'scope.json').read_text())
base=Path.home()/'Library/Application Support/QW3ComputerUseTest/native'
while time.time()<record['expires_at']:
    config=json.loads((base/'config.json').read_text())
    if config.get('task_scope_id')!=record['scope_id']:raise SystemExit(0)
    time.sleep(2)
config=json.loads((base/'config.json').read_text())
if config.get('task_scope_id')==record['scope_id']:
    status=json.loads((base/'status.json').read_text());pid=status['pid']
    cmd=subprocess.check_output(['/bin/ps','-p',str(pid),'-o','command='],text=True)
    assert 'QW3 Mac Agent.app/Contents/MacOS/qw3-mac-agent' in cmd
    os.kill(pid,signal.SIGTERM)
    (base/'config.json').write_bytes((task/'original-config.json').read_bytes())
    time.sleep(1);subprocess.run(['/usr/bin/open',str(Path.home()/'Applications/QW3 Mac Agent.app')],check=True)
    (task/'scope-expired.json').write_text(json.dumps({'at':time.time()}))
'''
    (task/'watchdog.py').write_text(watcher)
    log=(task/'watchdog.log').open('ab')
    config.update(allowed_apps=values['allowed_apps'],window_prefixes=values['window_prefixes'],
        task_scope_id=values['scope_id'],scope_expires_at=expires)
    config_path.write_text(json.dumps(config));os.chmod(config_path,0o600)
    child=subprocess.Popen(['/usr/bin/python3',str(task/'watchdog.py')],stdout=log,stderr=log,start_new_session=True)
    (task/'watchdog-pid.txt').write_text(str(child.pid))
    fresh=restart()
    print(json.dumps({'prepared':True,'expires_at':expires,'paused':fresh['paused'],
                     'permissions':fresh['ax_trusted'] and fresh['screen_capture_allowed']}))
"""


async def qualify_window_focus(connection,scope_id,desktop,driver,report,output):
    """Prepared test files only; exact window identity and Unicode disk checks."""
    directory=connection['remote_task_dir']+'/'+scope_id
    paths=[directory+'/QW3-Mac-20261006-foundation-'+suffix+'.txt' for suffix in ('A','B')]
    preparation="import json,subprocess\nfrom pathlib import Path\npaths=json.loads("+repr(json.dumps(paths))+ ")\nfor name in paths:\n p=Path(name);assert not p.exists();p.write_text('Foundation fixture\\n',encoding='utf-8')\n subprocess.run(['/usr/bin/open','-a','TextEdit',str(p)],check=True)\n"
    await ssh(connection,'/usr/bin/python3 -',preparation.encode())
    await asyncio.sleep(1)
    observation=await desktop.observe()
    def check(case,name,passed,evidence=None):
        report['checks'].append({'case':case,'name':name,'passed':bool(passed),'evidence':evidence})
        save(output/'result.json',report)
        if not passed:raise RuntimeError('Native foundation check failed: '+name)
    async def act(action,name):
        latest=desktop.latest
        result=await desktop.act({'desktop_id':latest['desktop_id'],'observation_id':latest['observation_id'],
                                 'action_id':name,'action':action})
        save(output/(name+'.json'),{k:v for k,v in result.items() if k!='observation'}|{
            'observation':{k:v for k,v in result['observation'].items() if k!='image'}})
        return result['observation']
    for suffix in ('A','B','A'):
        expected='QW3-Mac-20261006-foundation-'+suffix+'.txt'
        matches=[w for w in observation.get('windows',[]) if w.get('title')==expected]
        check('T02-01','unique-test-window-'+suffix,len(matches)==1,[{k:v for k,v in w.items()} for w in matches])
        window=matches[0]
        observation=await act({'kind':'activate_window','window_ref':window['window_ref']},'window-'+suffix+'-'+str(len(report['checks'])))
        check('T02-01','selected-native-window-'+suffix,
              observation.get('foreground',{}).get('window_identity')==window['identity'],observation.get('foreground'))
    editors=[c for c in observation.get('accessibility',{}).get('controls',[]) if c.get('kind')=='AXTextArea' and c.get('native_ref')]
    check('T04-01','one-editable-native-control',len(editors)==1,{'count':len(editors)})
    field=editors[0]
    observation=await act({'kind':'focus_control','control_ref':field['native_ref']},'focus-test-editor')
    check('T04-01','native-control-focus-confirmed',observation.get('focused_control',{}).get('identity')==field['native_identity'],
          observation.get('focused_control'))
    text='QW3 foundation.\n中文与换行验证。\nUnicode: café, λ, 😀.\n'
    await act({'kind':'key','key':'CTRL+A'},'select-test-editor')
    observation=await act({'kind':'type','text':text},'type-test-unicode')
    fields=[c for c in observation.get('accessibility',{}).get('controls',[]) if c.get('kind')=='AXTextArea']
    check('T04-03','exact-field-content',len(fields)==1 and fields[0].get('value')==text)
    await act({'kind':'key','key':'CTRL+S'},'save-test-editor')
    verify="import json\nfrom pathlib import Path\npaths=json.loads("+repr(json.dumps(paths))+ ")\nprint(json.dumps([Path(p).read_text(encoding='utf-8') for p in paths],ensure_ascii=False))\n"
    content=json.loads(await ssh(connection,'/usr/bin/python3 -',verify.encode()))
    check('T12-02','disk-content-and-other-document-unchanged',content==[text,'Foundation fixture\n'])
    # Independent foreground change between observation and input must refuse.
    before=(await driver.request('GET','/status'))['synthetic_actions']
    await ssh(connection,'/usr/bin/open -a /System/Applications/Utilities/Terminal.app')
    refused=False
    try:await act({'kind':'type','text':'MUST-NOT-BE-TYPED'},'stolen-focus-rejected')
    except DesktopRejected:refused=True
    after=(await driver.request('GET','/status'))['synthetic_actions']
    check('T04-02','changed-foreground-refuses-input',refused and before==after,{'synthetic_before':before,'synthetic_after':after})


async def run(args):
    if args.output.exists(): raise ValueError('Use a fresh immutable qualification directory')
    args.output.mkdir(parents=True, mode=0o700)
    connection=json.loads(args.connection.read_text())
    if 'QW3ComputerUseTest/tasks/' not in connection['remote_task_dir']:
        raise ValueError('Only the explicit Mac test directory may be used')
    scope_id=args.output.name
    report={'scope':'Live Mac application transitions; no model calls or installer advancement',
            'model_calls':0,'installer_buttons_clicked':0,'checks':[],'started_at':time.time(),'status':'preparing'}
    driver=MacDriver(connection['endpoint'],connection['device_id'],connection['native_token_path'],
                     args.output/'native-dispatches',activation_connection=args.connection)
    lease=DeviceLease('macos:'+connection['device_id']).acquire()
    runner=desktop=gateway=None
    prepared=False
    try:
        prepared=True  # Cleanup checks the marker even if preparation loses its reply.
        raw=await ssh(connection,'/usr/bin/python3 -',scope_script(connection,scope_id).encode())
        preparation=json.loads(raw);save(args.output/'scope-preparation.json',preparation)
        prepared=preparation['prepared']
        await driver.start();initial=await driver.request('GET','/status')
        save(args.output/'initial-native.json',initial)
        gateway=Gateway(driver,args.output/'gateway');token=secrets.token_hex(32)
        runner=web.AppRunner(create_mac_app(gateway,token));await runner.setup()
        site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
        port=site._server.sockets[0].getsockname()[1]
        desktop=Desktop('http://127.0.0.1:'+str(port),token,args.output/'desktop')
        await desktop.start();desktop.task_id=scope_id
        target=TargetExecution(desktop)
        report['checks'].append({'case':'T06-01','name':'permissions-effective-and-paused',
                                 'passed':initial['ax_trusted'] and initial['screen_capture_allowed'] and initial['paused']})
        await desktop.control('resume')
        for alias in ('terminal','installer'):
            result=await target.invoke({'operation':'ensure_app','app':alias,'transition_id':'live-'+alias,'timeout_seconds':12})
            observation=await desktop.observe() if result['observation'].get('state_only') else result['observation']
            save(args.output/(alias+'-transition.json'),{k:v for k,v in result.items() if k!='observation'} | {
                'observation':{k:v for k,v in observation.items() if k!='image'}})
            report['checks'].append({'case':'T01-02','name':alias+'-actual-foreground',
                                     'passed':result['verified'],'status':result['status']})
            count=gateway.db.execute('SELECT COUNT(*) FROM actions').fetchone()[0]
            duplicate=await target.invoke({'operation':'ensure_app','app':alias,'transition_id':'live-'+alias,'timeout_seconds':12})
            report['checks'].append({'case':'T08-01','name':alias+'-duplicate-no-relaunch',
                'passed':duplicate['duplicate'] and gateway.db.execute('SELECT COUNT(*) FROM actions').fetchone()[0]==count})
            if alias=='installer':
                fg=observation['foreground'];parent=fg.get('scope_parent',{})
                report['checks'].append({'case':'T03-02','name':'actual-titleless-sheet-owned-parent',
                    'passed':fg.get('title')=='' and parent.get('pid')==fg.get('pid') and bool(parent.get('title'))
                        and parent.get('relation') in ('AXParent','AXWindow','AXSheets')
                        and observation.get('accessibility',{}).get('status')=='ok',
                    'foreground':fg,'accessibility_status':observation.get('accessibility',{}).get('status')})
        report['host_launch_receipts']=gateway.db.execute('SELECT COUNT(*) FROM actions WHERE status="completed"').fetchone()[0]
        if 'activate_window' in driver.capabilities['actions']:
            await qualify_window_focus(connection,scope_id,desktop,driver,report,args.output)
        report['native_synthetic_actions_delta']=(await driver.request('GET','/status'))['synthetic_actions']-initial['synthetic_actions']
        report['checks'].append({'case':'T08-02','name':'no-uncertain-actions',
                                 'passed':not desktop.unresolved() and gateway.db.execute('SELECT COUNT(*) FROM actions WHERE status="uncertain"').fetchone()[0]==0})
        report['status']='passed' if all(c['passed'] for c in report['checks']) else 'failed'
    except BaseException as exc:
        report.update(status='failed',error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        if desktop:
            with contextlib.suppress(Exception):await desktop.control('pause')
            await desktop.close()
        if runner:await runner.cleanup()
        if gateway:await gateway.close()
        else:await driver.close()
        lease.close()
        if prepared:
            try:
                restored=json.loads(await ssh(connection,'/usr/bin/python3 -',scope_script(connection,scope_id,True).encode()))
                report['scope_restoration']=restored
            except Exception as exc:
                report.update(status='failed',restoration_error=type(exc).__name__+': '+str(exc))
        report['finished_at']=time.time();save(args.output/'result.json',report)
    print(json.dumps({'status':report['status'],'checks':report['checks'],'result':str((args.output/'result.json').resolve())},ensure_ascii=False))
    if report['status']!='passed':raise RuntimeError('Live Mac transition qualification did not pass')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--connection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    asyncio.run(run(parser.parse_args()))
