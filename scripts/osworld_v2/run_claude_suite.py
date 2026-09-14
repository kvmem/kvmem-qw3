#!/usr/bin/env python3
"""Resume native Claude Code pairs from a completed, validated reference pair.

This runner does not select methods using test scores or change task setup,
model budgets, image handling, client history, or the official evaluators.
External services are left under the operator's control.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from summarize_claude_suite import write_summary

ROOT=Path(__file__).resolve().parents[2]
PYTHON=ROOT/'third_party/OSWorld-V2/.venv/bin/python'
DRIVER=ROOT/'scripts/osworld_v2/run_claude_pair.py'

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def reference_config(reference):
    if json.loads((reference/'comparison_status.json').read_text())['status']!='complete':
        raise RuntimeError('Finish and assess the reference pair before expanding')
    if not json.loads((reference/'gpu_parity.json').read_text())['passed']:
        raise RuntimeError('Reference GPU equivalence validation did not pass')
    binary=reference/'qw3-claudecode'
    digest=hashlib.sha256(binary.read_bytes()).hexdigest()
    arms={}
    for arm in ('qw3','kvmem'):
        directory=reference/'full'/arm
        manifest=json.loads((directory/'server_manifest.json').read_text())
        if manifest['binary_sha256']!=digest:raise RuntimeError('Reference binaries differ')
        command=json.loads((directory/'server_command.json').read_text());command[0]=str(binary)
        arms[arm]={'command':command,'environment':manifest['environment']}
    return binary,digest,arms

def run_arm(task,arm,out,workspace,config,digest,args,state):
    for name,expected in args.driver_sources.items():
        if hashlib.sha256((DRIVER.parent/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Benchmark driver changed during the cohort: '+name)
    result_path=out/'result.json'
    if result_path.exists():
        result=json.loads(result_path.read_text())
        manifest=json.loads((out/'server_manifest.json').read_text())
        client=json.loads((out/'client_manifest.json').read_text())
        if (result.get('status')=='complete' and result.get('task')==task and
            manifest['binary_sha256']==digest and manifest['command']==config['command'] and
            manifest['environment']==config['environment'] and client.get('workspace')==str(workspace) and
            client.get('request_recording')=='zstd-groups' and client.get('driver_sources')==args.driver_sources):return result
        raise RuntimeError(f'Preserve or diagnose unfinished attempt before retry: {out}')
    if (out/'api.jsonl').exists():raise RuntimeError(f'Existing unfinished trajectory: {out}')
    out.mkdir(parents=True,exist_ok=True)
    env={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy') and not k.startswith('QW3_')}
    env.update(config['environment'])
    env['OSWORLD_REQUEST_RECORDING']='zstd-groups'
    env.update(OSWORLD_FILE_BASE_URL=str(ROOT/'third_party/OSWorld-V2/cache/osworld_v2_assets'),
        WEBSITE_HOST_SUFFIX='site.hku.icu',OSWORLD_BENCHMARK_RELEASE='osworld-v2-2026.08.08',
        AWS_REGION='us-east-1',AWS_SUBNET_ID='unused-for-docker',AWS_SECURITY_GROUP_ID='unused-for-docker',
        NO_PROXY='127.0.0.1,localhost',no_proxy='127.0.0.1,localhost')
    command=config['command']
    save(out/'server_command.json',command)
    save(out/'server_manifest.json',{'binary_sha256':digest,'command':command,
        'environment':config['environment'],'arm':arm,'task':task,'seed':0,'reference':str(args.reference)})
    driver=None
    with (out/'qw3_server.log').open('w') as server_log:
        server=subprocess.Popen(command,cwd=ROOT,env=env,stdout=server_log,stderr=subprocess.STDOUT,start_new_session=True)
        state.update(task=task,arm=arm,server_pid=server.pid,updated_at=utc());save(args.output/'suite_status.json',state)
        try:
            port=command[command.index('--port')+1]
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            for _ in range(180):
                if server.poll() is not None:raise RuntimeError('Model server exited during startup')
                try:
                    with opener.open(f'http://127.0.0.1:{port}/health',timeout=2):break
                except OSError:time.sleep(1)
            else:raise RuntimeError('Model server startup timeout')
            with (out/'runner.log').open('w') as log:
                driver=subprocess.Popen([str(PYTHON),str(DRIVER),'--task',task,'--arm',arm,'--output',str(out),
                    '--server-port',port,'--max-steps','500','--workspace',str(workspace)],cwd=ROOT/'third_party/OSWorld-V2',env=env,stdout=log,stderr=subprocess.STDOUT)
                state.update(driver_pid=driver.pid,updated_at=utc());save(args.output/'suite_status.json',state)
                while driver.poll() is None:
                    mem={line.split()[0].rstrip(':'):int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split())==3}
                    memory={line.split(':',1)[0]:int(line.split()[1]) for line in Path(f'/proc/{server.pid}/status').read_text().splitlines() if line.startswith(('VmRSS:','VmHWM:','VmSwap:'))}
                    record={'at':utc(),'stage':arm,'arm':arm,'mem_available_bytes':mem['MemAvailable'],'mem_total_bytes':mem['MemTotal'],'server_memory_kib':memory}
                    with (out.parents[1]/'resources.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                    try:driver.wait(timeout=10)
                    except subprocess.TimeoutExpired:pass
                code=driver.returncode
            result=json.loads(result_path.read_text())
            if code or result['status']!='complete':raise RuntimeError(f'Trial failed: {out}')
            return result
        finally:
            if driver and driver.poll() is None:
                driver.terminate()
                try:driver.wait(timeout=30)
                except subprocess.TimeoutExpired:driver.kill();driver.wait()
            if server.poll() is None:
                os.killpg(server.pid,signal.SIGTERM)
                try:server.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--tasks-file',type=Path,required=True)
    parser.add_argument('--efficiency-report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();args.reference=args.reference.resolve();args.output=args.output.resolve()
    tasks=json.loads(args.tasks_file.read_text())['tasks']
    if len(tasks)!=len(set(tasks)) or any(not (ROOT/f'third_party/OSWorld-V2/evaluation_examples/task_class/task_{t}.py').exists() for t in tasks):
        raise ValueError('Task set has duplicates or unavailable official tasks')
    binary,digest,configs=reference_config(args.reference)
    args.driver_sources={name:hashlib.sha256((DRIVER.parent/name).read_bytes()).hexdigest() for name in
        ('run_claude_pair.py','claude_desktop_mcp.py','request_archive.py')}
    efficiency=json.loads(args.efficiency_report.read_text())
    if not efficiency.get('all_outputs_equal') or efficiency.get('binary_sha256')!=digest:
        raise RuntimeError('Complete fixed-input output-equivalence evidence for this binary is required')
    variants=efficiency.get('variants',{})
    if len(variants)<2 or any(variant.get('selection_requests',0)<=0 for variant in variants.values()):
        raise RuntimeError('The efficiency replay must exercise real selections')
    plan={'tasks':tasks,'binary':str(binary),'binary_sha256':digest,'seed':0,'max_steps':500,
          'reference':str(args.reference),'arms':configs,'arm_order':'alternating by task index',
          'efficiency_report':str(args.efficiency_report.resolve()),'reuse_completed_reference_052':False,
          'workspace_policy':'one empty temporary directory outside the repository, shared by each pair',
          'request_recording':'zstd-groups','driver_sources':args.driver_sources}
    if args.dry_run:print(json.dumps(plan,indent=2));return
    if (args.output/'plan.json').exists():
        previous_plan=json.loads((args.output/'plan.json').read_text())
        if previous_plan!=plan:raise RuntimeError('Preserve the existing cohort before changing its configuration')
    prior=args.output/'suite_status.json'
    if prior.exists():
        previous=json.loads(prior.read_text())
        try:os.kill(previous['pid'],0)
        except ProcessLookupError:pass
        else:raise RuntimeError('A suite process recorded for this output is still alive')
    save(args.output/'plan.json',plan)
    state={'status':'running','pid':os.getpid(),'started_at':utc(),'completed':{}}
    save(args.output/'suite_status.json',state)
    try:
        for index,task in enumerate(tasks):
            root=args.output/'trials'/task
            if root.is_symlink():raise RuntimeError('Final cohort must contain fresh pairs: '+str(root))
            workspace_record=root/'workspace.json'
            if workspace_record.exists():
                workspace=Path(json.loads(workspace_record.read_text())['path'])
                if workspace.parent!=Path('/tmp') or not workspace.name.startswith(f'qw3-osworld-{task}-'):
                    raise RuntimeError('Unexpected persisted agent workspace')
                workspace.mkdir(exist_ok=True)
            else:
                workspace=Path(tempfile.mkdtemp(prefix=f'qw3-osworld-{task}-',dir='/tmp'))
                save(workspace_record,{'path':str(workspace)})
            for arm in (('kvmem','qw3') if index%2==0 else ('qw3','kvmem')):
                result=run_arm(task,arm,root/'full'/arm,workspace,configs[arm],digest,args,state)
                state['completed'][f'{task}/{arm}']=result;save(args.output/'suite_status.json',state)
            with (root/'analysis.log').open('w') as log:
                subprocess.run([str(PYTHON),str(ROOT/'scripts/osworld_v2/analyze_claude_pair.py'),str(root)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            write_summary(args.output)
            if not any(workspace.iterdir()):workspace.rmdir()
        state.update(status='complete',finished_at=utc());save(args.output/'suite_status.json',state)
    except Exception as exc:
        state.update(status='failed',error=str(exc),updated_at=utc());save(args.output/'suite_status.json',state);raise

if __name__=='__main__':main()
