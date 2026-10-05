"""Exercise the fixed native dispatch gate in the owned VM without input.

The gated body only prints a read-only probe result. Negative probes supply
explicitly altered expected identities; they never modify the real window.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.request import ProxyHandler, Request, build_opener

from native_keyboard_execution import confirmed_native_rejection, guarded_input_script
from native_window_evidence import guest_script
from platform_pilots import owned_container
import shared_quota_staged_successors as staged


CASES=('stable-1','changed-focus','changed-process-generation','changed-geometry','stable-2','stable-3')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def qualify(output, declaration, build):
    q=staged.RUNTIME
    candidate=q.read(build)
    if candidate.get('kind')!='isolated_sdk51_native_keyboard_execution_candidate':
        raise ValueError('Use the independent native keyboard build')
    for name,expected in candidate['helper_source_sha256'].items():
        if digest(Path(__file__).with_name(name))!=expected:
            raise ValueError('Candidate helper source changed')
    for name,expected in candidate['candidate_files_sha256'].items():
        if digest(Path(build).parent/'framework'/name)!=expected:
            raise ValueError('Candidate framework changed')
    d=q.read(declaration);registration=q.read(d['sdk_registration'])
    q.base.require_identity(registration['identity'])
    state=q.read(registration['queue'])
    matches=[row for row in state['new_gateways'] if row['catalogue_id']=='linux-vm']
    if len(matches)!=1:raise RuntimeError('Require the exact registered Linux gateway')
    gateway=matches[0];q.base.require_identity(gateway['process_identity'])
    container=owned_container('qw3-computer-linux','qw3.goal','cross-device-20261003')
    if container['Id']!=state['containers']['linux'] or not container['State']['Running']:
        raise RuntimeError('Registered owned Linux VM changed or stopped')
    output.mkdir(parents=True,exist_ok=False)
    report=dict(kind='actual_readonly_native_input_gate_qualification',started_at=time.time(),passed=False,
        source_sha256={name:digest(Path(__file__).with_name(name)) for name in
                       ('native_keyboard_execution.py','qualify_native_keyboard_gate.py')},
        candidate_build=dict(path=str(build),sha256=digest(build)),controller_identity=registration['identity'],
        gateway_identity=gateway['process_identity'],gateway_session=gateway['session_id'],container_id=container['Id'],
        planned_cases=list(CASES),samples=[],model_requests=0,device_inputs=0,
        new_gateway_or_input_owner_started=False,production_driver_or_frozen_policy_changed=False,
        business_success_improvement_proven=False)
    opener=build_opener(ProxyHandler({}))

    def save():
        (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')

    def execute(script):
        body=json.dumps(dict(command=['python','-c',script],shell=False,timeout=5)).encode()
        request=Request('http://127.0.0.1:18126/execute',data=body,headers={'Content-Type':'application/json'})
        with opener.open(request,timeout=7) as response:result=json.load(response)
        if result.get('returncode',0)!=0 or result.get('status')=='error':
            raise RuntimeError('Read-only guest gate probe failed')
        return result

    save()
    try:
        for case in CASES:
            sample=dict(case=case,started_at=time.time(),passed=False)
            try:
                expected=json.loads(execute(guest_script())['output'])
                sample['original_native_read']=expected
                if expected.get('status')!='ok':raise RuntimeError('No stable native foreground read')
                # Deep-copy so the original native facts remain unchanged.
                altered=json.loads(json.dumps(expected))
                if case=='changed-focus':altered['focus_window_id']+=1
                elif case=='changed-process-generation':altered['reported_owner']['start_ticks']+=1
                elif case=='changed-geometry':altered['bounds'][1]+=1
                sample['expected_for_gate']=altered
                original=('import pyautogui;pyautogui.FAILSAFE=False;'
                          'print(json.dumps(dict(kind="readonly-native-gate-passed",input_sent=False)))')
                script,nonce=guarded_input_script(original,altered,'key')
                path=output/(case+'.py');path.write_text(script)
                sample['fixed_probe_script_sha256']=digest(path)
                result=execute(script);sample['guest_result']=result
                rejection=confirmed_native_rejection(result,nonce)
                sample['confirmed_rejection']=rejection
                if case.startswith('stable-'):
                    sample['passed']=(json.loads(result['output'])==dict(kind='readonly-native-gate-passed',input_sent=False))
                else:sample['passed']=rejection is not None and rejection['clipboard_prepared'] is False
            except Exception as error:
                sample.update(error_kind=type(error).__name__,error=str(error))
            sample['seconds']=time.time()-sample['started_at']
            report['samples'].append(sample);save()
        q.base.require_identity(registration['identity']);q.base.require_identity(gateway['process_identity'])
        report['passed_cases']=sum(sample['passed'] for sample in report['samples'])
        report['passed']=report['passed_cases']==len(CASES)
        report['limitations']=['Fixed native gate only; its body printed a probe and never called an input API.',
            'Negative expected identities were test data, not actual focus/process mutations.',
            'Real keyboard delivery, clipboard recovery and QW3 task acceptance require separate trials.',
            'Ongoing task state changes can make a stable probe fail; no resampling or input retries.']
    finally:
        report['finished_at']=time.time();save()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--declaration',type=Path,required=True)
    parser.add_argument('--build',type=Path,required=True)
    args=parser.parse_args();report=qualify(args.output.resolve(),args.declaration.resolve(),args.build.resolve())
    print(json.dumps({key:report[key] for key in ('passed','passed_cases','device_inputs','model_requests')}))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
