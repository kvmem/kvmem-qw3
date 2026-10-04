"""Reuse the frozen eight-trial runner with a declared continued-parent gate."""
import argparse
import inspect
from pathlib import Path
import signal

import kvmem_osworld_queue as base
import osworld_retired_trial_successors as successor


def patched_run_source(source):
    before='previous = json.loads((parent_dir / "queue.json").read_text())'
    after='previous = json.loads((parent_dir / "continued_queue.json").read_text())'
    if source.count(before)!=1:raise base.SafetyHalt('Frozen parent-reader shape changed')
    source=source.replace(before,after)
    before='"initial_registered_engine": initial, "planned_trials": 8}'
    after='"initial_registered_engine": initial, "planned_trials": 8, "successor_wrapper_sha256": wrapper_sha, "successor_declaration_sha256": followup_sha}'
    if source.count(before)!=1:raise base.SafetyHalt('Frozen controller-state shape changed')
    return source.replace(before,after)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--followup',type=Path,required=True)
    p.add_argument('--wait-seconds',type=int,default=172800);p.add_argument('--resource-wait-seconds',type=int,default=21600)
    p.add_argument('--min-free-mib',type=int,default=56000);args=p.parse_args()
    if args.min_free_mib<56000 or not 60<=args.wait_seconds<=172800 or not 60<=args.resource_wait_seconds<=86400:
        p.error('Keep the original bounded shared-resource profile')
    args.followup=args.followup.resolve();d,_,_,_=successor.validate_followup(args.followup)
    followup_sha=base.digest(args.followup)
    args.protocol=Path(d['memory_protocol']);args.state_dir=Path(d['memory_state_dir'])
    original_validate=base.validate_protocol
    def validate(path):
        if base.digest(args.followup)!=followup_sha:raise base.SafetyHalt('Successor declaration changed')
        successor.validate_followup(args.followup)
        return original_validate(path)
    namespace=dict(vars(base));namespace.update(validate_protocol=validate,wrapper_sha=base.digest(Path(__file__)),
        followup_sha=followup_sha,terminal_parent=lambda protocol,state:successor.terminal_parent(protocol,state,args.followup))
    code=patched_run_source(inspect.getsource(base.run));compile(code,'declared-kvmem-continuation','exec')
    exec(code,namespace)
    def interrupted(*_):raise base.SafetyHalt('Continued memory waiter interrupted; live work preserved')
    signal.signal(signal.SIGTERM,interrupted)
    namespace['run'](args)


if __name__=='__main__':main()
