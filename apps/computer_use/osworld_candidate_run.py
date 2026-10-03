"""Reuse the original task runner with declared candidate services and media.

The frozen campaign is never patched on disk. Hooks live only in this separate
process, and the underlying task initializer/evaluator remain unchanged.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import urllib.error
from unittest.mock import patch

from osworld_linux import environment, runner
from osworld_windows.source import ROOT


CANDIDATE_PORTS={'server':18500,'chromium':18501,'vnc':18502,'vlc':18503}
ORIGINAL_APP='http://127.0.0.1:18305'
CANDIDATE_APP='http://127.0.0.1:18505'


def candidate_command(command, output_tokens):
    command=list(command)
    if '-m' not in command:return command
    module=command[command.index('-m')+1]
    if module=='qw3_computer.device_gateway':
        command[command.index('-m')+1]='device_gateway_entry'
        command[command.index('--port')+1]='18504'
    elif module=='qw3_computer.app':
        command[command.index('--port')+1]='18505'
        command[command.index('--desktop')+1]='http://127.0.0.1:18504'
        command+=['--model-output-tokens',str(output_tokens)]
    return command


class MediaResponse:
    """A failed recording cannot hide a completed independent evaluation."""
    def __init__(self,content=b''):self.content=content
    def raise_for_status(self):pass


class CandidateMedia:
    def __init__(self,directory,recorder_factory=None):
        self.directory=Path(directory)
        self.recorder=None
        self.complete=False
        self.errors=[]
        self.recorder_factory=recorder_factory

    def record_error(self,stage,exc):
        self.errors.append({'stage':stage,'error':type(exc).__name__+': '+str(exc),'at':time.time()})
        runner.save(self.directory/'candidate-media-errors.json',{'errors':self.errors})

    def start(self):
        try:
            from linux_recording import LinuxRecording
            vm=json.loads((self.directory/'vm.json').read_text())
            factory=self.recorder_factory or LinuxRecording
            self.recorder=factory(self.directory,vm['container_id'],name=vm['name'])
            # The agent uses the private core package. Only the read-only
            # recorder child also needs the declared host helper/PyAV paths.
            recorder_path=os.pathsep.join(filter(None,[os.environ.get('PYTHONPATH'),
                str(ROOT/'apps/computer_use'),str(ROOT/'apps/computer_use/.runtime/media-qualification')]))
            with patch.dict(os.environ,{'PYTHONPATH':recorder_path}):self.recorder.start()
        except Exception as exc:self.record_error('start',exc)
        return MediaResponse()

    def finish(self):
        try:
            if self.recorder is None:raise RuntimeError('Recorder was not initialized')
            receipt=self.recorder.finish()
            self.complete=receipt.get('complete') is True
            return MediaResponse((self.directory/'continuous.mp4').read_bytes())
        except Exception as exc:
            self.record_error('finish',exc)
            return MediaResponse()

    def close(self):
        if self.recorder is not None and not self.recorder.collected:
            try:self.recorder.finish('interrupted.mp4')
            except Exception as exc:self.record_error('cleanup',exc)


def engine_identity(runtime):
    from osworld_linux.campaign import engine_identity as identity
    return identity(Path(runtime))


def admit(args):
    """No candidate model run overlaps the original formal campaign."""
    plan=json.loads(args.formal_campaign.joinpath('plan.json').read_text())
    if plan['status']!='finished' or len(plan['jobs'])!=40 or any(job['status']!='finished' for job in plan['jobs']):
        raise ValueError('Finish all40 frozen attempts before candidate OSWorld model trials')
    from osworld_linux.campaign import verify_frozen
    # This checks the actual owned engine PID, command, binary configuration and
    # source identity, rather than trusting a readiness response alone.
    engine_plan=dict(plan,engine_runtime=str(args.engine_runtime),engine_identity=engine_identity(args.engine_runtime))
    verify_frozen(engine_plan)
    engine=json.loads(args.engine_runtime.joinpath('manifest.json').read_text())
    with Path(engine['command'][0]).open('rb') as binary:
        if hashlib.file_digest(binary,'sha256').hexdigest()!=engine['binary_sha256']:
            raise ValueError('Owned engine binary differs from its declared hash')
    if args.arm not in {'legacy','verified'} or not 256<=args.model_output_tokens<=32768:
        raise ValueError('Invalid workflow/output budget')
    if args.seconds!=1800 or args.tools!=250:
        raise ValueError('Candidate comparison retains1800seconds and250calls')
    if not (args.framework/'device_gateway_entry.py').is_file():
        raise ValueError('Freeze the canonical entry inside the candidate package')
    acceptance=json.loads(args.acceptance_manifest.read_text())
    identity=args.output.name
    if acceptance['trials'].get(identity)!={'task':args.task,'arm':args.arm}:
        raise ValueError('Candidate identity/task/arm differs from the declared comparison')
    if (acceptance['seconds']!=args.seconds or acceptance['tools']!=args.tools
            or acceptance['model_output_tokens']!=args.model_output_tokens
            or acceptance['framework_files']!=runner.source_identity(args.framework)
            or acceptance['engine_identity']!=engine_identity(args.engine_runtime)):
        raise ValueError('Candidate source or budget differs from the declared comparison')
    expected=acceptance['candidate_runner_sha256']
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=expected:
        raise ValueError('Candidate runner changed after declaration')
    entry=hashlib.sha256((args.framework/'device_gateway_entry.py').read_bytes()).hexdigest()
    if entry!=acceptance['canonical_entry_sha256']:
        raise ValueError('Candidate canonical entry changed after declaration')
    for relative,digest in acceptance['candidate_source_sha256'].items():
        source=(ROOT/relative).resolve()
        if not source.is_relative_to(ROOT/'apps/computer_use') or hashlib.sha256(source.read_bytes()).hexdigest()!=digest:
            raise ValueError('Candidate recording/dependency source changed: '+relative)
    # Candidate services have separate ports, but share the model with the
    # device pilots. The independent app must also be idle and reconciled.
    state=runner.request(args.pilot_app,'/api/state')
    if state['status'] not in {'idle','completed','failed','stopped'} or state.get('uncertain_actions'):
        raise ValueError('Another device pilot still owns a model task')
    try:
        windows=runner.request('http://127.0.0.1:18083','/api/state',timeout=3)
    except urllib.error.URLError as exc:
        if not isinstance(exc.reason,ConnectionRefusedError):
            raise ValueError('Native Windows model task state is unconfirmed') from exc
    else:
        if windows['status'] not in {'idle','completed','failed','stopped'} or windows.get('uncertain_actions'):
            raise ValueError('Native Windows product task is still active')
    return plan


@contextlib.contextmanager
def candidate_hooks(args,media):
    import requests
    original_launch=runner.launch
    original_request=runner.request
    original_post=requests.post
    original_save=runner.save
    engine=json.loads(args.engine_runtime.joinpath('manifest.json').read_text())
    metadata={'candidate_harness':True,'model_output_tokens':args.model_output_tokens,
        'kvmem':engine['kvmem']!='off','kvmem_mode':engine['kvmem'],
        'recording_mode':'continuous_read_only_vnc_recording',
        'canonical_gateway_entry':True,'candidate_runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'canonical_entry_sha256':hashlib.sha256((args.framework/'device_gateway_entry.py').read_bytes()).hexdigest(),
        'comparison':'Declared development workflow ablation; changed output budget is not an isolated model improvement',
        'acceptance_sha256':hashlib.sha256(args.acceptance_manifest.read_bytes()).hexdigest()}

    def launch(command,log):return original_launch(candidate_command(command,args.model_output_tokens),log)
    def request(origin,path,*positional,**keywords):
        return original_request(CANDIDATE_APP if origin==ORIGINAL_APP else origin,path,*positional,**keywords)
    def post(url,*positional,**keywords):
        origin='http://127.0.0.1:'+str(CANDIDATE_PORTS['server'])
        if url==origin+'/start_recording':return media.start()
        if url==origin+'/end_recording':return media.finish()
        return original_post(url,*positional,**keywords)
    def save(path,body):
        if Path(path)==args.output.resolve()/'manifest.json':body=dict(body,**metadata)
        return original_save(path,body)

    with patch.object(environment,'PORTS',CANDIDATE_PORTS),patch.object(runner,'launch',launch),\
            patch.object(runner,'request',request),patch.object(requests,'post',post),patch.object(runner,'save',save):
        yield


def run(args):
    admit(args)
    media=CandidateMedia(args.output.resolve())
    try:
        with candidate_hooks(args,media):
            result=runner.run(args)
    finally:media.close()
    # The unchanged runner writes a terminal result first. Retain its exact
    # bytes before adding the candidate's media qualification and acceptance.
    raw=args.output/'result.json'
    args.output.joinpath('runner-result.json').write_bytes(raw.read_bytes())
    result.update(candidate_harness=True,video_collection_passed=media.complete,
                  video_collection_errors=media.errors,
                  unassisted_business_passed=result.get('unassisted_passed',False))
    result['accepted']=result['unassisted_business_passed'] and media.complete
    runner.save(raw,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=runner.TASKS,required=True)
    parser.add_argument('--arm',choices=['legacy','verified'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--framework',type=Path,required=True)
    parser.add_argument('--engine-runtime',type=Path,required=True)
    parser.add_argument('--formal-campaign',type=Path,required=True)
    parser.add_argument('--acceptance-manifest',type=Path,required=True)
    parser.add_argument('--pilot-app',default='http://127.0.0.1:18205')
    parser.add_argument('--model-output-tokens',type=int,default=16384)
    args=parser.parse_args()
    args.qualification=False;args.seconds=1800;args.tools=250
    def interrupted(*_):raise TimeoutError('Candidate watchdog requested shutdown')
    signal.signal(signal.SIGTERM,interrupted)
    import fcntl
    with open('/tmp/qw3-cross-device-candidate-model.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=run(args)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
