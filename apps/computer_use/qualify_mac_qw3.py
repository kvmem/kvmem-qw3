"""Opt-in autonomous local QW3 Mac task with independent file verification.

Preparation creates dedicated fixtures; every task decision and input is QW3's.
No Installer advancement or cloud advisor. Earlier paused tasks are untouched.
"""
import argparse
import asyncio
import contextlib
import fcntl
import json
from pathlib import Path
import secrets
import time

from aiohttp import web

from macos_agent_driver import MacDriver
from macos_gateway_entry import create_mac_app
from qualify_macos_native import ssh
from qualify_mac_transitions import scope_script
from qw3_computer.app import Controller, web_app
from qw3_computer.desktop import Desktop
from qw3_computer.device_gateway import Gateway
from qw3_computer.device_lease import DeviceLease


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


async def run(args):
    if args.output.exists():raise ValueError('Use a fresh immutable output directory')
    args.output.mkdir(parents=True,mode=0o700)
    connection=json.loads(args.connection.read_text())
    if 'QW3ComputerUseTest/tasks/' not in connection['remote_task_dir']:raise ValueError('Dedicated Mac fixture directory required')
    report={'kind':'autonomous-local-QW3-foundation-pilot','formal':False,'advisor':False,
            'installer_actions':0,'root_task_inputs':0,'started_at':time.time(),'status':'preparing'}
    model_lock=open('/tmp/qw3-cross-device-model-queue.lock','a')
    lease=driver=gateway=desktop=controller=runner=app_runner=None
    prepared=False
    try:
        fcntl.flock(model_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lease=DeviceLease('macos:'+connection['device_id']).acquire()
        prepared=True
        preparation=json.loads(await ssh(connection,'/usr/bin/python3 -',scope_script(connection,args.output.name).encode()))
        save(args.output/'scope-preparation.json',preparation)
        remote=connection['remote_task_dir']+'/'+args.output.name
        paths=[remote+'/QW3-Mac-20261006-agent-'+suffix+'.txt' for suffix in ('A','B')]
        original=['Draft A.\n','KEEP B UNCHANGED.\n']
        expected='本地 QW3 桌面验收\nPython 环境：待配置\nLaTeX 环境：待配置\n'
        script="import json,subprocess\nfrom pathlib import Path\nvalues=json.loads("+repr(json.dumps(list(zip(paths,original))))+")\nfor name,text in values:\n p=Path(name);assert not p.exists();p.write_text(text,encoding='utf-8')\n subprocess.run(['/usr/bin/open','-a','TextEdit',str(p)],check=True)\n"
        await ssh(connection,'/usr/bin/python3 -',script.encode())
        driver=MacDriver(connection['endpoint'],connection['device_id'],connection['native_token_path'],args.output/'native-dispatches')
        await driver.start();initial=await driver.request('GET','/status');save(args.output/'initial-native.json',initial)
        gateway=Gateway(driver,args.output/'gateway');token=secrets.token_hex(32)
        runner=web.AppRunner(create_mac_app(gateway,token));await runner.setup()
        site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
        desktop=Desktop('http://127.0.0.1:'+str(site._server.sockets[0].getsockname()[1]),token,args.output/'desktop')
        controller=Controller(args.output/'app',desktop,args.backend,0,device_platform='macos',workflow=True,
            thinking=True,tool_call_limit=40,output_policy='task',model_output_tokens=8192,
            required_features=['observe','ensure_app','select_window','focus_control'],qualification_mode=True)
        app_runner=web.AppRunner(web_app(controller));await app_runner.setup()
        app_site=web.TCPSite(app_runner,'127.0.0.1',0);await app_site.start()
        controller.port=app_site._server.sockets[0].getsockname()[1]
        await controller.start()
        prompt=('这是 Mac 桌面基础验收，仅操作 TextEdit 中已经打开的两个测试文档。'
            '先辨认并选择 QW3-Mac-20261006-agent-A.txt，将它的全文替换为下面三行，最后一行后也要有换行，并保存到原文件：\n'
            +expected+'\n确认保存，再切换到 QW3-Mac-20261006-agent-B.txt，确认内容仍为 KEEP B UNCHANGED.（末尾有换行），不要修改 B。'
            '使用当前 windows 列表中的 window_ref 精确选择文档，有 editable AX 控件时使用 focus_control 并核验焦点。'
            '在 workflow 的步骤结果中保留文本核验。最终确认 A 的内容与 B 不变后结束。'
            '不要操作 Terminal、Installer 或其他测试文档；不要安装任何软件。')
        save(args.output/'task-spec.json',{'prompt':prompt,'fixture_paths':paths,'expected_contents':[expected,original[1]]})
        await controller.start_task(prompt)
        report.update(status='running',task_id=controller.state['id'],thread_id=controller.state['thread_id'],
                      app_port=controller.port,fixture_paths=paths,model='local QW3',model_context=controller.model_context_window)
        save(args.output/'result.json',report)
        began=time.monotonic()
        while controller.state['status'] in ('starting','running','verifying'):
            if time.monotonic()-began>=args.timeout:
                await controller.control('pause');report['timeout']=True;break
            await asyncio.sleep(2)
        report.update(agent_status=controller.state['status'],tools_used=controller.state['tool_calls_used'],
                      model_calls=len(list((args.output/'app/inference').glob('*/request.json'))))
        await desktop.control('pause')
        verify="import json\nfrom pathlib import Path\npaths=json.loads("+repr(json.dumps(paths))+")\nprint(json.dumps([Path(p).read_text(encoding='utf-8') for p in paths],ensure_ascii=False))\n"
        contents=json.loads(await ssh(connection,'/usr/bin/python3 -',verify.encode()))
        independent={'exact_A':contents[0]==expected,'unchanged_B':contents[1]==original[1],
                     'read_by_new_remote_process':True,'contents':contents}
        save(args.output/'independent-verification.json',independent)
        report['independent_verification']=independent
        report['native_model_actions']=(await driver.request('GET','/status'))['synthetic_actions']-initial['synthetic_actions']
        report['status']='passed' if controller.state['status']=='completed' and independent['exact_A'] and independent['unchanged_B'] else 'failed'
        report['uncertain_actions']=desktop.unresolved()
    except BaseException as exc:
        report.update(status='failed',error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        if controller:
            if controller.state['status'] in ('starting','running','verifying'):
                with contextlib.suppress(Exception):await controller.control('pause')
            await controller.close()
        elif desktop:await desktop.close()
        if app_runner:await app_runner.cleanup()
        if runner:await runner.cleanup()
        if gateway:await gateway.close()
        elif driver:await driver.close()
        if lease:lease.close()
        if prepared:
            try:report['scope_restoration']=json.loads(await ssh(connection,'/usr/bin/python3 -',scope_script(connection,args.output.name,True).encode()))
            except Exception as exc:report.update(status='failed',restoration_error=type(exc).__name__+': '+str(exc))
        model_lock.close()
        report['finished_at']=time.time();save(args.output/'result.json',report)
    print(json.dumps({k:report.get(k) for k in ('status','agent_status','tools_used','native_model_actions','error')},ensure_ascii=False))
    if report['status']!='passed':raise RuntimeError('Autonomous Mac foundation pilot did not pass')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--connection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--backend',required=True)
    parser.add_argument('--timeout',type=int,default=600)
    asyncio.run(run(parser.parse_args()))
