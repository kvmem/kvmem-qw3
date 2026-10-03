"""Real owned Linux VM: switch foreground before input, require zero delivery."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request
import uuid

from device_receipt_qualification import owned_gateway
from osworld_linux.runner import request,save
from platform_pilots import linux_execute
from qw3_computer.device_registry import load_registry
from qw3_computer.linux_driver import FOREGROUND_SCRIPT


def run(args):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    result=dict(kind='real_linux_foreground_preflight_qualification',model_used=False,
                desktop_task_success_claimed=False,passed=False,started_at=time.time(),dispatch_attempted=False)
    device=None
    try:
        item=load_registry(args.registry)['linux-vm']
        app=request(args.app,'/api/state')
        if app['status']!='idle' or app['uncertain_actions'] or app['devices']['selected']=='linux-vm':
            raise ValueError('Candidate must be idle on a different device')
        container=json.loads(subprocess.check_output(['docker','inspect','qw3-computer-linux']))[0]
        if container['Config']['Labels'].get('qw3.goal')!='cross-device-20261003':
            raise ValueError('Development VM ownership changed')
        gateway=owned_gateway(item)
        token=item['token_file'].read_text().strip()
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        def device(route,body=None):
            req=urllib.request.Request(item['endpoint']+route,
                 data=json.dumps(body).encode() if body is not None else None,
                 headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
            try:
                with opener.open(req,timeout=15) as response:return response.status,json.load(response)
            except urllib.error.HTTPError as error:return error.code,json.loads(error.read())
        status,before=device('/observe')
        if status!=200 or not before.get('paused') or before['device_id']!=gateway['device_id']:
            raise ValueError('Development gateway is not paused with its owned identity')
        result.update(container_id=container['Id'],gateway=gateway,device_id=before['device_id'],
                      before_foreground=before['foreground'])
        guest='/home/user/.qw3-preflight-qualification/'+output.name
        linux_execute('from pathlib import Path;import subprocess;'
            f'p=Path({guest!r});p.mkdir(parents=True,exist_ok=False);'
            'f=p/"qualification-note.txt";f.write_text("Local recovery qualification.\\n");'
            'subprocess.Popen(["gedit","--new-window",str(f)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)')
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            current=json.loads(linux_execute(FOREGROUND_SCRIPT)['output'])
            if current['handle']!=before['foreground']['handle']:break
            time.sleep(.25)
        else:raise TimeoutError('Foreground did not change; no test input authorized')
        result['changed_foreground']=current
        status,resumed=device('/control',{'command':'resume'})
        if status!=200 or resumed.get('paused'):raise ValueError('Owned gateway resume failed')
        action_id='preflight_'+uuid.uuid4().hex
        arguments=dict(action_id=action_id,desktop_id=before['desktop_id'],
                       observation_id=before['observation_id'],action={'kind':'key','key':'ESC'})
        result['dispatch_attempted']=True
        status,response=device('/act',arguments)
        _,receipt=device('/actions/'+action_id)
        result.update(http_status=status,response=response,remote_receipt_status=receipt['status'],
                      canonical_non_delivery=status==409 and response.get('delivery')=='rejected')
        checks=dict(expected_http_status=status==args.expected_status,
                    foreground_rejection_observed='Foreground changed' in response.get('error',''),
                    no_dispatch_reservation=receipt['status']=='unknown')
        if args.expected_status==409:checks['confirmed_non_delivery']=result['canonical_non_delivery']
        result.update(checks=checks,passed=all(checks.values()),
                      input_sent=False if receipt['status']=='unknown' else None)
    except Exception as exc:result['error']=type(exc).__name__+': '+str(exc)
    finally:
        if device:
            try:
                status,paused=device('/control',{'command':'pause'})
                result['gateway_paused']=status==200 and paused.get('paused') is True
            except Exception as exc:result['cleanup_error']=type(exc).__name__+': '+str(exc)
        result['finished_at']=time.time();save(output/'result.json',result)
    print({k:result.get(k) for k in ['passed','http_status','canonical_non_delivery','input_sent','error']},flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--app',default='http://127.0.0.1:18205')
    parser.add_argument('--expected-status',type=int,choices=[409,503],default=409)
    raise SystemExit(0 if run(parser.parse_args())['passed'] else 1)
