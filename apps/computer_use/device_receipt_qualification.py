"""Qualify lost-reply recovery against an idle, owned real device, without a model."""
import argparse
import asyncio
import contextlib
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.parse import urlparse
import uuid

import aiohttp

from osworld_linux.runner import request,save
from qw3_computer.desktop import Desktop
from qw3_computer.device_registry import load_registry
from receipt_fault_proxy import ReceiptFaultProxy


def owned_gateway(item):
    matches=[]
    for source in Path('/proc').iterdir():
        if not source.name.isdigit():continue
        try:
            if source.stat().st_uid!=os.getuid():continue
            argv=[v.decode() for v in (source/'cmdline').read_bytes().split(b'\0') if v]
            if not any(entry in argv for entry in ('qw3_computer.device_gateway','device_gateway_entry',
                                                   'semantic_device_gateway_entry')):continue
            def option(name):return argv[argv.index(name)+1]
            if 'semantic_device_gateway_entry' in argv and option('--platform')!='linux':continue
            if (option('--platform')!=item['platform']
                    or option('--port')!=str(urlparse(item['endpoint']).port)
                    or Path(option('--runtime')).resolve()!=item['token_file'].parent):continue
            matches.append(dict(pid=int(source.name),device_id=option('--device-id'),command=argv))
        except (OSError,UnicodeError,ValueError,IndexError):continue
    if len(matches)!=1:raise ValueError('Expected exactly one owned gateway for the catalogue endpoint')
    return matches[0]


async def exercise(desktop, proxy, key='ANDROID_HOME'):
    reopened=None
    await desktop.start()
    task_id='receipt-qualification-'+uuid.uuid4().hex
    desktop.task_id=task_id
    await desktop.control('resume')
    await desktop.observe()
    arguments=dict(action_id='lost-reply-one',desktop_id=desktop.latest['desktop_id'],
                   observation_id=desktop.latest['observation_id'],
                   action={'kind':'key','key':key})
    pending=asyncio.create_task(desktop.act(arguments))
    delivery=asyncio.create_task(proxy.delivered.wait())
    try:
        done,_=await asyncio.wait({pending,delivery},timeout=30,return_when=asyncio.FIRST_COMPLETED)
        if delivery not in done:
            if pending in done:await pending
            raise TimeoutError('No completed device receipt was available for fault injection')
        pending.cancel()
        with contextlib.suppress(asyncio.CancelledError):await pending
        uncertain=desktop.unresolved()
        if len(uncertain)!=1 or not desktop.paused:
            raise AssertionError('Lost reply did not reserve its action and pause input')
        proxy.release.set()
        endpoint,token,runtime=desktop.endpoint,desktop.token,desktop.runtime
        await desktop.close()
        desktop=reopened=Desktop(endpoint,token,runtime)
        # New client process/session opens the saved action record.
        desktop.task_id=task_id
        await desktop.start()
        await desktop.control('resume')
        duplicate=await desktop.act(arguments)
        checks=dict(uncertain_action_preserved_after_lost_reply=bool(uncertain),
                    saved_receipt_reconciled_after_client_restart=not desktop.unresolved(),
                    duplicate_returns_saved_receipt=duplicate.get('duplicate') is True,
                    single_actual_dispatch=proxy.dispatches.get(proxy.selected)==1,
                    receipt_read_without_action_replay=proxy.selected in proxy.receipt_queries)
        return desktop,dict(passed=all(checks.values()),task_id=task_id,checks=checks,fault=proxy.evidence())
    except BaseException:
        pending.cancel()
        with contextlib.suppress(asyncio.CancelledError,Exception):await pending
        if reopened is not None:
            with contextlib.suppress(Exception):await reopened.control('pause')
            await reopened.close()
        raise
    finally:
        delivery.cancel()
        with contextlib.suppress(asyncio.CancelledError):await delivery


async def perform(args, output):
    item=load_registry(args.registry)[args.device]
    keys={'android':'ANDROID_HOME','linux':'ESC'}
    if item['platform'] not in keys:raise ValueError('Use a qualified Linux or Android gateway')
    app=request(args.app,'/api/state')
    if (app['status']!='idle' or app.get('uncertain_actions')
            or app.get('devices',{}).get('selected')==args.device):
        raise ValueError('Qualification requires the candidate idle on a different device')
    container=json.loads(subprocess.check_output(['docker','inspect','qw3-computer-'+item['platform']]))[0]
    if container['Config']['Labels'].get('qw3.goal')!='cross-device-20261003' or not container['State']['Running']:
        raise ValueError('Device container ownership or state changed')
    token=item['token_file'].read_text().strip()
    gateway=owned_gateway(item)
    async with aiohttp.ClientSession(trust_env=False,headers={'Authorization':'Bearer '+token}) as session:
        async with session.get(item['endpoint']+'/observe') as response:
            response.raise_for_status();before=await response.json()
    if before.get('paused') is not True or before.get('device_id')!=gateway['device_id']:
        raise ValueError('Device must already be paused with its expected identity')
    proxy=ReceiptFaultProxy(item['endpoint'],token)
    desktop=None
    report=dict(kind='real_device_receipt_transport_qualification',platform=item['platform'],model_used=False,
                desktop_task_success_claimed=False,container_id=container['Id'],started_at=time.time())
    report.update(catalogue_device_id=args.device,gateway=gateway,device_id=before['device_id'],
                  session_id=before['session_id'])
    save(output/'admission.json',report)
    try:
        await proxy.start()
        desktop=Desktop(proxy.origin,token,output/'desktop')
        desktop,result=await exercise(desktop,proxy,key=keys[item['platform']])
        report.update(result)
    except Exception as exc:
        report.update(passed=False,error=type(exc).__name__+': '+str(exc),fault=proxy.evidence())
    finally:
        proxy.release.set()
        if desktop:
            with contextlib.suppress(Exception):await desktop.control('pause')
            await desktop.close()
        await proxy.close()
        report['finished_at']=time.time()
        save(output/'result.json',report)
    print({k:report.get(k) for k in ['passed','error','model_used']},flush=True)
    return report


async def run(args):
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    started=time.time()
    try:return await perform(args,output)
    except Exception as exc:
        admitted=(output/'admission.json').exists()
        report=dict(kind='real_device_receipt_transport_qualification',model_used=False,
                    desktop_task_success_claimed=False,passed=False,
                    status='execution-error' if admitted else 'preflight-rejected',
                    input_sent=None if admitted else False,needs_reconciliation=admitted,
                    started_at=started,finished_at=time.time(),
                    error=type(exc).__name__+': '+str(exc))
        save(output/'result.json',report)
        print({k:report.get(k) for k in ['passed','error','model_used']},flush=True)
        return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry',type=Path,required=True)
    parser.add_argument('--device',default='android-emulator')
    parser.add_argument('--app',default='http://127.0.0.1:18205')
    parser.add_argument('--output',type=Path,required=True)
    raise SystemExit(0 if asyncio.run(run(parser.parse_args()))['passed'] else 1)
