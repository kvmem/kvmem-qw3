"""Real local-model lost-receipt pilot; administrative recovery stays separate.

The ordinary browser task is unchanged. One completed type receipt is hidden,
then the stopped client is reopened and its original receipt queried while
input and model inference remain paused. This is fault acceptance, not
unassisted business completion.
"""
import argparse
import asyncio
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import time

from osworld_linux.runner import request, save
from osworld_windows.bundle import verify_bundle
from osworld_windows.source import ROOT
from qw3_computer.device_lease import DeviceLease
from qw3_computer.device_registry import load_registry
from receipt_fault_proxy import ReceiptFaultProxy
from platform_pilots import android_command, linux_execute, owned_container


async def api(origin, route, body=None):
    return await asyncio.to_thread(request, origin, route, body, timeout=30)


def recovery_checks(before, after, fault, *, backend_calls_before, backend_calls_after,
                    ledger_receipt_id, ledger_status, main_idle, dispatch_counts_before):
    selected = fault.get('selected_action_id')
    pending = before.get('uncertain_actions', [])
    return {
        'completed_reply_actually_withheld': any(e.get('event') == 'completed_receipt_withheld'
                                                for e in fault.get('events', [])),
        'original_receipt_reserved': bool(selected) and len(pending) == 1
                                    and pending[0].get('receipt_id') == selected,
        'pause_and_model_interrupt_confirmed': before.get('status') == 'paused'
                    and before.get('desktop_paused') is True and before.get('turn_status') == 'interrupted',
        'same_task_after_restart': bool(before.get('id')) and after.get('id') == before.get('id'),
        'same_budget_after_read_only_recovery': type(before.get('tool_calls_used')) is int
                    and after.get('tool_calls_used') == before['tool_calls_used']
                    and after.get('tool_call_limit') == before.get('tool_call_limit'),
        'input_and_model_stay_paused': after.get('status') == 'paused'
                    and after.get('desktop_paused') is True and after.get('turn_status') == 'interrupted',
        'original_receipt_reconciled': selected in fault.get('receipt_queries', [])
                    and ledger_receipt_id == selected and ledger_status == 'completed'
                    and not after.get('uncertain_actions'),
        'exactly_one_dispatch_per_saved_intent': bool(fault.get('dispatch_counts'))
                    and all(count == 1 for count in fault['dispatch_counts'].values())
                    and fault['dispatch_counts'].get(selected) == 1,
        'no_dispatch_during_restart_or_receipt_query': fault.get('dispatch_counts') == dispatch_counts_before,
        'no_new_inference_during_recovery': backend_calls_before > 0
                    and backend_calls_after == backend_calls_before,
        'main_application_remains_idle': main_idle is True,
    }


async def close_owned_app(process, command):
    if process is None or process.poll() is not None:
        return
    proc = Path('/proc')/str(process.pid)
    if proc.stat().st_uid != os.getuid():
        raise RuntimeError('Owned receipt client UID changed; no signal sent')
    live = [v.decode() for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
    if live != command or Path(os.readlink(proc/'cwd')).resolve() != ROOT:
        raise RuntimeError('Owned receipt client identity changed; no signal sent')
    process.terminate()
    await asyncio.to_thread(process.wait, timeout=60)


async def run(args):
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    result = dict(kind='integration-pilot-not-formal', formal=False, platform=None,
        model_used=False, accepted=False, fault_accepted=False, fault_condition='receipt_loss',
        administrative_recovery=True, unassisted_recovery_claimed=False, started_at=time.time())
    proxy = process = lease = fixture = recorder = None
    command = None; initial_device = None; app_origin = 'http://127.0.0.1:18212'
    raw_state = after = None; initialized = False
    with open('/tmp/qw3-cross-device-model-queue.lock', 'a') as model_lock, \
         open('/tmp/qw3-osworld-vm.lock', 'a') as vm_lock:
        try:
            fcntl.flock(model_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            result.update(status='preflight-rejected',error='Another model task owns the queue',
                          input_sent=False,owned_task_retired=False,finished_at=time.time())
            save(output/'integration-result.json',result)
            return result
        try:
            registry = load_registry(args.registry); item = registry[args.device]
            platform = item['platform']; result['platform'] = platform
            if platform == 'windows': fcntl.flock(vm_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            native = verify_bundle(args.bundle)
            candidate = json.loads((args.runtime/'candidate-manifest.json').read_text())
            framework = args.runtime/'framework'
            inventory = json.loads((args.bundle/'manifest.json').read_text())['source_files']
            for name, digest in candidate['files'].items():
                if (hashlib.sha256((framework/'qw3_computer'/name).read_bytes()).hexdigest() != digest
                        or inventory['qw3_computer/'+name] != digest):
                    raise ValueError('Candidate differs from its qualified native core')
            main = await api(args.app, '/api/state')
            if main['status'] != 'idle' or not main['desktop_paused'] or main['uncertain_actions']:
                raise ValueError('Main model application must be safely idle')
            if (main.get('thinking') is not True or main.get('model_output_tokens') != 16384
                    or main.get('tool_call_limit') != 100):
                raise ValueError('Use the declared local thinking/16K/100-call profile')
            initial_device = main['devices']['selected']
            other = 'linux-vm' if args.device == 'windows-vm' else 'windows-vm'
            if initial_device == args.device: await api(args.app, '/api/device', {'device_id':other})
            # Claim the real transport as well as the proxy namespace. The
            # proxy must not create a second route around device ownership.
            lease = DeviceLease(item['endpoint']).acquire()
            token = item['token_file'].read_text().strip()
            import aiohttp
            async with aiohttp.ClientSession(trust_env=False, headers={'Authorization':'Bearer '+token}) as client:
                async with client.get(item['endpoint']+'/observe', timeout=20) as response:
                    response.raise_for_status(); observed = await response.json()
            if observed.get('paused') is not True:
                raise ValueError('Target input is not already paused')
            label, owner = (('com.qw3.computer-use.owner','codex-goal-20260930') if platform == 'windows'
                            else ('qw3.goal','cross-device-20261003'))
            vm = await asyncio.to_thread(owned_container, 'qw3-computer-'+platform, label, owner)
            if not vm['State']['Running']: raise ValueError('Owned VM is not running')
            save(output/'admission.json', {'platform':platform,'device':args.device,'container_id':vm['Id'],
                'qualified_bundle':native,'main_application':args.app,'main_runtime':str(args.runtime),
                'physical_device_lease_held':True,'model_concurrency':1,'model_backend':'http://127.0.0.1:18080/v1',
                'thinking':True,'model_output_tokens':16384,'tool_call_limit':100,'advisor':False,
                'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
            if platform == 'windows':
                import windows_fault_fixture as provider
                from fault_browser_pilot import FIELDS, prompt
                from windows_fault_pilot import WindowsRecording
                fixture = await asyncio.to_thread(provider.initialize, output.name, 'normal', output, args.bundle)
                await asyncio.to_thread(provider.arm, fixture, output)
                instruction = prompt('http://127.0.0.1:8086/', FIELDS)
                recorder = WindowsRecording(output, vm['Id'], 'qw3-computer-windows')
            else:
                import browser_pilot as provider
                fixture, instruction = await asyncio.to_thread(provider.initialize, platform, output.name, output, linux_execute)
                if platform == 'linux':
                    from linux_recording import LinuxRecording
                    await asyncio.to_thread(linux_execute, 'import subprocess;subprocess.Popen('+repr(
                        ['google-chrome','--new-window',fixture['url']])+',stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)')
                    recorder = LinuxRecording(output, vm['Id'])
                else:
                    from live_android_recording import LiveAndroidRecording
                    await asyncio.to_thread(android_command, ['shell','am','start','-a','android.intent.action.VIEW',
                        '-d',fixture['url'],'-p','com.android.chrome'], capture_output=True)
                    recorder = LiveAndroidRecording(['docker','exec','qw3-computer-android', str(
                        ROOT/'apps/computer_use/.runtime/android/sdk/platform-tools/adb'),'-s','emulator-5556'], output)
            initialized = True
            await asyncio.to_thread(recorder.start)
            proxy = ReceiptFaultProxy(item['endpoint'], token, kind='type')
            await proxy.start()
            raw_registry = json.loads(args.registry.read_text())
            for device in raw_registry['devices']:
                device['token_file'] = str(registry[device['id']]['token_file'])
                if device['id'] == args.device: device['endpoint'] = proxy.origin
            registry_path = output/'proxy-registry.json'; save(registry_path, raw_registry); registry_path.chmod(0o600)
            client_runtime = output/'app'
            command = [str(ROOT/'.venv/bin/python'),'-m','qw3_computer.app','--port','18212','--runtime',str(client_runtime),
                '--device-registry',str(registry_path),'--selected-device',args.device,'--execution-mode','verified',
                '--thinking','--model-output-tokens','16384','--tool-call-limit','100','--output-policy','task']
            env = dict(os.environ, PYTHONPATH=str(framework)+':'+str(ROOT/'apps/computer_use'))
            async def launch():
                nonlocal process
                with socket.socket() as check:
                    check.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                    check.bind(('127.0.0.1',18212))
                with (output/'client-launch.log').open('ab') as log:
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                process = child
                save(output/'owned-client.json', {'pid':child.pid,'command':command})
                for _ in range(100):
                    try:
                        state = await api(app_origin, '/api/state')
                        if state['ready']: return child, state
                    except Exception:
                        if child.poll() is not None: raise RuntimeError('Receipt client exited before readiness')
                    await asyncio.sleep(.2)
                raise TimeoutError('Receipt client readiness is unconfirmed')
            process, _ = await launch()
            save(output/'pilot-manifest.json', {'kind':'integration-pilot-not-formal','case':platform+'_browser',
                'fault_condition':'receipt_loss','prompt':instruction,'deadline_seconds':args.seconds,
                'no_model_access_to_initializer_or_grader':True,'no_fault_details_in_model_prompt':True,
                'app_runtime':str(client_runtime),'qualified_framework':str(framework), 'started_at':time.time()})
            state = await api(app_origin, '/api/task', {'prompt':instruction}); result['model_used'] = True
            task_id = state['id']; started = time.monotonic()
            with (output/'progress.jsonl').open('w') as log:
                while True:
                    state = await api(app_origin, '/api/state')
                    if state.get('id') != task_id: raise RuntimeError('Receipt pilot task was replaced')
                    log.write(json.dumps({'at':time.time(),'status':state['status'],'tools':state['tool_calls_used']})+'\n');log.flush()
                    if state['status'] in {'paused','completed','failed','stopped'}: break
                    if time.monotonic()-started >= args.seconds:
                        raise TimeoutError('Lost receipt fault did not reach a terminal task within the declared budget')
                    if getattr(args,'record_gpu',True):
                        try:
                            gpu = await asyncio.to_thread(subprocess.run, ['nvidia-smi',
                                '--query-gpu=timestamp,memory.used,memory.free,utilization.gpu',
                                '--format=csv,noheader'],capture_output=True,text=True,timeout=5)
                            with (output/'gpu.csv').open('a') as sample: sample.write(gpu.stdout)
                        except Exception as exc:
                            with (output/'gpu-sampling-errors.jsonl').open('a') as sample:
                                sample.write(json.dumps({'at':time.time(),'error':type(exc).__name__})+'\n')
                    await asyncio.sleep(1)
            for _ in range(60):
                if state.get('turn_status') == 'interrupted': break
                await asyncio.sleep(1); state = await api(app_origin, '/api/state')
            raw_state = state; save(output/'task-before-reconciliation.json', raw_state)
            model_task_seconds = time.monotonic()-started
            if not proxy.delivered.is_set() or len(state.get('uncertain_actions',[])) != 1:
                raise RuntimeError('Completed input reply was not withheld and reserved; no recovery claim')
            if state.get('turn_status') != 'interrupted' or state.get('desktop_paused') is not True:
                raise RuntimeError('Input/model stop is not confirmed; no client restart')
            before_calls = len(list((client_runtime/'inference').glob('*/request.json')))
            before_dispatches = dict(proxy.dispatches)
            recovery_started = time.monotonic()
            await close_owned_app(process, command); process = None
            proxy.release.set()
            process, reopened = await launch()
            if reopened.get('id') != task_id or reopened.get('status') != 'paused':
                raise RuntimeError('Saved receipt task did not survive restart')
            after = await api(app_origin, '/api/reconcile', {})
            save(output/'task-after-reconciliation.json', after)
            after_calls = len(list((client_runtime/'inference').glob('*/request.json')))
            ledger = sqlite3.connect('file:'+str(client_runtime/'devices'/args.device/'desktop.sqlite')+'?mode=ro', uri=True)
            try: row = ledger.execute('SELECT id,status FROM actions WHERE id=?',(proxy.selected,)).fetchone()
            finally: ledger.close()
            main = await api(args.app, '/api/state')
            checks = recovery_checks(raw_state, after, proxy.evidence(), backend_calls_before=before_calls,
                backend_calls_after=after_calls, ledger_receipt_id=row[0] if row else None,
                ledger_status=row[1] if row else None, main_idle=main['status']=='idle' and not main['uncertain_actions'],
                dispatch_counts_before=before_dispatches)
            save(output/'independent-fault-response.json', {'kind':'receipt-loss-safe-halt-and-administrative-reconciliation',
                'passed':all(checks.values()),'checks':checks,'fault_exercised':proxy.delivered.is_set(),
                'unassisted_business_passed':False,'unassisted_recovery_claimed':False})
            if platform == 'windows':
                from windows_fault_pilot import independent_grade
                await asyncio.to_thread(provider.collect, fixture, output)
                independent = independent_grade(fixture, output); fixture = None
            else: independent = await asyncio.to_thread(provider.grade, fixture, linux_execute)
            save(output/'independent-browser.json', independent)
            media = await asyncio.to_thread(recorder.finish)
            result.update(app_status=raw_state['status'], seconds=model_task_seconds,
                total_seconds=time.monotonic()-started,recovery_seconds=time.monotonic()-recovery_started,
                tool_calls=raw_state['tool_calls_used'], uncertain_actions=len(after['uncertain_actions']),
                uncertain_actions_before_reconciliation=len(raw_state['uncertain_actions']),
                artifact_passed=independent['passed'], fault_exercised=True, fault_response_passed=all(checks.values()),
                unassisted_passed=False, video_collection_passed=bool(media.get('passed',media.get('complete'))),
                fault_accepted=all(checks.values()) and bool(media.get('passed',media.get('complete'))),
                video_bytes=(output/'continuous.mp4').stat().st_size)
        except Exception as exc:
            result.update(error=type(exc).__name__+': '+str(exc), status='preflight-rejected' if not initialized else 'infrastructure-failed')
        finally:
            if raw_state and raw_state.get('uncertain_actions') and (after is None or after.get('uncertain_actions')):
                result['requires_reconciliation'] = True
            if proxy: save(output/'fault-transport.json', proxy.evidence())
            if process:
                try:
                    final = await api(app_origin,'/api/state')
                    if final.get('uncertain_actions'): result['requires_reconciliation'] = True
                    await api(app_origin,'/api/control',{'command':'stop'})
                except Exception as exc: result.update(owned_task_stop_unconfirmed=True,retirement_error=str(exc))
                try:
                    await close_owned_app(process,command); process = None
                except Exception as exc: result.update(owned_task_stop_unconfirmed=True,client_stop_error=str(exc))
            if fixture:
                try:
                    if result['platform']=='windows': await asyncio.to_thread(provider.collect,fixture,output)
                    else: await asyncio.to_thread(provider.stop,fixture,linux_execute)
                except Exception as exc: result['fixture_cleanup_error']=str(exc)
            if recorder and not recorder.collected:
                try: await asyncio.to_thread(recorder.finish,'interrupted.mp4')
                except Exception as exc: result['recording_cleanup_error']=str(exc)
            if proxy: await proxy.close()
            if lease: lease.close()
            if initial_device is not None and process is None:
                try:
                    final_main = await api(args.app,'/api/device',{'device_id':initial_device})
                    result['main_idle_restored']=final_main['status']=='idle' and not final_main['uncertain_actions']
                except Exception as exc: result['main_restore_error']=str(exc)
            result['owned_task_retired'] = (process is None and not result.get('requires_reconciliation')
                and not result.get('owned_task_stop_unconfirmed') and result.get('main_idle_restored') is True)
            if result.get('fixture_cleanup_error') or result.get('recording_cleanup_error'):
                result['owned_task_retired'] = False
            if not result['owned_task_retired']:
                result.update(accepted=False,fault_accepted=False,owned_task_stop_unconfirmed=True)
            result['finished_at']=time.time();save(output/'integration-result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app',required=True);parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--registry',type=Path,required=True);parser.add_argument('--device',required=True)
    parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=900)
    args=parser.parse_args()
    if not 60<=args.seconds<=900: parser.error('Use60..900seconds')
    outcome=asyncio.run(run(args))
    print({key:outcome.get(key) for key in ['fault_accepted','accepted','error','owned_task_retired']},flush=True)
    raise SystemExit(0 if outcome['fault_accepted'] else 1)


if __name__=='__main__':main()
