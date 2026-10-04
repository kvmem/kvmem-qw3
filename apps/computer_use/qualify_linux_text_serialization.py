"""Owned Linux GUI qualification of literal paste versus saved editor bytes."""
import argparse
import asyncio
import base64
import hashlib
import json
from pathlib import Path
import time

from osworld_linux.runner import request,save
from platform_pilots import owned_container
from qw3_computer.linux_driver import LinuxDriver


TEXT='Local computer use\nQW3 Windows Linux Android\n'


async def action(driver,output,action):
    before=await driver.observe()
    await driver.preflight(action,before)
    receipt=await driver.dispatch(action)
    await asyncio.sleep(.4)
    after=await driver.observe()
    with (output/'actions.jsonl').open('a') as stream:
        stream.write(json.dumps({'action':action,'before_foreground':before['foreground'],
            'after_foreground':after['foreground'],'driver_receipt':receipt,'at':time.time()})+'\n')
    (output/'latest.png').write_bytes(after['png'])
    return after


async def run(args):
    initial=request(args.app,'/api/state')
    if initial['status']!='idle' or initial.get('uncertain_actions'):
        raise ValueError('Use the idle development app, never a running agent')
    container=owned_container('qw3-computer-linux','qw3.goal','cross-device-20261003')
    if not container['State']['Running']:raise ValueError('Development VM is not running')
    args.output.mkdir(parents=True,exist_ok=False)
    report={'kind':'native_linux_editor_serialization_qualification','model_used':False,'formal':False,
            'container_id':container['Id'],'driver_source_sha256':hashlib.sha256(Path(__file__).with_name('qw3_computer').joinpath('linux_driver.py').read_bytes()).hexdigest(),
            'started_at':time.time(),'cases':[],'changes_editor_preferences':False}
    driver=LinuxDriver('http://127.0.0.1:18126','serialization-qualification')
    await driver.start()
    try:
        for label,payload in [('with_final_lf',TEXT),('without_final_lf',TEXT.rstrip('\n'))]:
            filename='/home/user/Desktop/'+args.output.name+'-'+label+'.txt'
            # Refuse to overwrite any existing artifact. A new named editor
            # window is created without closing other documents or apps.
            code='from pathlib import Path;import subprocess;assert not Path('+repr(filename)+').exists();subprocess.Popen(["gedit","--new-window"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)'
            await driver.execute(code)
            await asyncio.sleep(2)
            before=await driver.observe()
            if before['foreground']['process']!='Gedit':raise ValueError('Editor did not become foreground')
            await action(driver,args.output,{'kind':'type','text':payload})
            await action(driver,args.output,{'kind':'key','key':'CTRL+SHIFT+S'})
            await action(driver,args.output,{'kind':'key','key':'CTRL+L'})
            await action(driver,args.output,{'kind':'type','text':filename})
            observed=await action(driver,args.output,{'kind':'key','key':'ENTER'})
            # Some GTK versions require the final Save button after the
            # location entry. Resolve the unique real control when present.
            candidates=[c for c in observed['accessibility'].get('controls',[])
                        if c.get('name')=='Save' and 'button' in c.get('kind','').lower()]
            if len(candidates)==1:
                target=candidates[0]
                await action(driver,args.output,{'kind':'click','x':target['center_x'],'y':target['center_y'],'button':'left','clicks':1})
            await asyncio.sleep(1)
            code='from pathlib import Path;import base64,json;p=Path('+repr(filename)+');print(json.dumps({"exists":p.exists(),"data":base64.b64encode(p.read_bytes()).decode() if p.exists() else None}))'
            data=json.loads((await driver.execute(code))['output'])
            actual=base64.b64decode(data['data']) if data['exists'] else None
            case={'label':label,'typed_text':payload,'filename':filename,'saved':data['exists'],
                  'saved_bytes':len(actual) if actual is not None else None,
                  'saved_text':actual.decode() if actual is not None else None,
                  'exact_expected':actual==TEXT.encode(),'same_as_typed':actual==payload.encode()}
            report['cases'].append(case);save(args.output/'result.json',report)
            foreground=await driver.foreground()
            if Path(filename).name not in foreground['title'] or foreground['process']!='Gedit':
                raise ValueError('Expected saved qualification document is not foreground; no close input sent')
            await action(driver,args.output,{'kind':'key','key':'ALT+F4'})
        report['passed']=len(report['cases'])==2 and all(row['saved'] for row in report['cases'])
    except Exception as exc:report.update(passed=False,error=type(exc).__name__+': '+str(exc))
    finally:
        await driver.close();report['finished_at']=time.time();save(args.output/'result.json',report)
    print({key:report.get(key) for key in ['passed','error','model_used']},flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--app',default='http://127.0.0.1:18205')
    args=parser.parse_args()
    raise SystemExit(0 if asyncio.run(run(args))['passed'] else 1)
