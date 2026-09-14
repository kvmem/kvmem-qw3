"""Complete agent trials for the InfLLM windows selected in the frozen plan."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time

from run_pilot_arm import clean_environment, gpu_processes
from run_model_smoke import save

ROOT=Path(__file__).resolve().parent


def variant_order(plan, task_index):
    variants=set(plan['variants'])
    if not variants or not variants <= {'large','small'}:
        raise ValueError('select at least one supported InfLLM window')
    preferred=['large','small'] if task_index%2==0 else ['small','large']
    return [variant for variant in preferred if variant in variants]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--shared-gpu',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    plan=json.loads(args.plan.read_text())
    if not plan.get('frozen'): raise ValueError('campaign must be validated and frozen')
    for relative,digest in plan['source_sha256'].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=digest:
            raise ValueError('frozen source changed: '+relative)
    for evidence in plan['validation_evidence']:
        if json.loads((ROOT/evidence).read_text())['status']!='passed':
            raise ValueError('validation not passed: '+evidence)
    current=Path(plan['gguf']['path']).stat()
    if [current.st_size,current.st_mtime_ns]!=plan['gguf']['size_mtime']:
        raise ValueError('GGUF changed since historical content hash validation')
    tasks=json.loads((ROOT/plan['task_manifest']).read_text())['tasks']
    variants=variant_order(plan,0)
    args.out=args.out.resolve()
    args.out.mkdir(parents=True,exist_ok=args.resume)
    path=args.out/'plan.json'
    if path.exists() and path.read_bytes()!=args.plan.read_bytes(): raise ValueError('campaign changed')
    path.write_bytes(args.plan.read_bytes())
    state={'status':'starting','started':time.time(),'completed':[],
           'shared_gpu':args.shared_gpu,'planned_trials':len(tasks)*len(variants),
           'selected_variants':variants,
           'order':'task-major; alternate selected variants when both are enabled'}
    child=observer=None
    def interrupt(signum,frame): raise KeyboardInterrupt(f'signal {signum}')
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,interrupt)
    try:
        save(args.out/'state.json',state)
        with (args.out/'memory_observer.log').open('a') as log:
            observer=subprocess.Popen([str(ROOT/'.venv/bin/python'),str(ROOT/'observe_pilot_memory.py'),
                '--campaign',str(args.out)],stdout=log,stderr=log,env=clean_environment())
        for idx,task in enumerate(tasks):
            order=variant_order(plan,idx)
            for variant in order:
                parent=args.out/task['task_id']/variant
                previous=sorted(parent.glob('attempt_*'))
                if previous:
                    result=previous[-1]/'result.json'
                    if result.exists() and json.loads(result.read_text()).get('status')=='completed':
                        state['completed'].append(str(previous[-1]));continue
                    raise RuntimeError(f'inspect incomplete attempt before retry: {parent}')
                arm=parent/'attempt_01'
                arm_plan={**plan,'methods':{'infllm':plan['variants'][variant]},'variant':variant}
                variant_plan=args.out/f'plan_{variant}.json'
                save(variant_plan,arm_plan)
                state.update(status='waiting_for_spare_gpu_memory',task=task['task_id'],
                             variant=variant,method='infllm',arm=str(arm))
                while True:
                    free=float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
                        '--format=csv,noheader,nounits'],text=True).strip())/1024
                    state.update(free_gpu_gib=free,gpu_owners=gpu_processes())
                    save(args.out/'state.json',state)
                    if free>=plan['variants'][variant]['minimum_free_gpu_gib'] and (args.shared_gpu or not state['gpu_owners']):break
                    time.sleep(10)
                parent.mkdir(parents=True,exist_ok=True)
                command=[str(ROOT/'.venv/bin/python'),str(ROOT/'run_aligned_infllm_arm.py'),
                    '--plan',str(variant_plan),'--out',str(arm),'--method','infllm','--task-index',str(idx),'--port','18241']
                if args.shared_gpu:command.append('--shared-gpu')
                with (parent/'attempt_01.controller.log').open('w') as log:
                    child=subprocess.Popen(command,cwd=ROOT.parents[1],stdout=log,stderr=log,env=clean_environment())
                    state.update(status='running_arm',child_pid=child.pid)
                    save(args.out/'state.json',state)
                    code=child.wait()
                result=json.loads((arm/'result.json').read_text()) if (arm/'result.json').exists() else {}
                if code or result.get('status')!='completed':
                    raise RuntimeError(f'arm needs inspection: {arm}; exit={code}; status={result.get("status")}')
                state['completed'].append(str(arm));save(args.out/'state.json',state)
                subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'summarize_window_campaign.py'),
                    '--campaign',str(args.out)],check=True,env=clean_environment())
        state['status']='completed'
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=55)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        state['finished']=time.time();save(args.out/'state.json',state)
        if observer is not None and observer.poll() is None:observer.terminate()


if __name__=='__main__':main()
