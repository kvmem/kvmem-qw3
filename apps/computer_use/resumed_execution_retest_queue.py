"""Reuse SDK51b trials after the continued forty and unchanged eight terminate."""
import argparse
import json
from pathlib import Path
import signal

import execution_retest_queue as base
import execution_retest_queue_v2 as prior
import osworld_retired_trial_successors as successor


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--followup',type=Path,required=True)
    p.add_argument('--memory-controller',type=Path,required=True)
    p.add_argument('--wait-seconds',type=int,default=172800);p.add_argument('--resource-wait-seconds',type=int,default=21600)
    p.add_argument('--min-free-mib',type=int,default=56000);args=p.parse_args()
    if args.min_free_mib<56000 or not 60<=args.wait_seconds<=172800 or not 60<=args.resource_wait_seconds<=86400:
        p.error('Keep the original bounded shared-resource profile')
    args.followup=args.followup.resolve();args.memory_controller=args.memory_controller.resolve()
    d,_,_,_=successor.validate_followup(args.followup)
    followup_sha=base.owned.digest(args.followup);memory_registration_sha=base.owned.digest(args.memory_controller)
    registration=base.read(args.memory_controller);identity=registration['identity']
    if (registration.get('kind')!='actual_resumed_memory_controller_identity'
            or registration.get('followup_sha256')!=followup_sha
            or registration.get('wrapper_sha256')!=base.owned.digest(Path(__file__).with_name('resumed_kvmem_queue.py'))
            or not successor.command_path(identity,'--followup',args.followup) or 'resumed_kvmem_queue' not in identity['command']):
        raise base.SafetyHalt('Actual continued memory controller identity is unproven')
    args.protocol=Path(d['sdk_protocol']);args.state_dir=Path(d['sdk_state_dir'])
    def validate(path):
        if base.owned.digest(args.followup)!=followup_sha or base.owned.digest(args.memory_controller)!=memory_registration_sha:
            raise base.SafetyHalt('Successor or actual controller registration changed')
        successor.validate_followup(args.followup)
        original,kp=prior.validate(path);effective=dict(original)
        effective['parent_osworld_queue']=str(Path(d['continued_osworld_queue']).relative_to(base.ROOT))
        effective['parent_kvmem_queue']=str((Path(d['memory_state_dir'])/'queue.json').relative_to(base.ROOT))
        return effective,kp
    original_ready=base.predecessors_ready
    def ready(protocol,memory_protocol):
        state=base.read(Path(d['memory_state_dir'])/'queue.json')
        if state.get('pid')!=identity['pid'] or state.get('successor_declaration_sha256')!=followup_sha:
            raise base.SafetyHalt('Continued memory queue identity changed')
        if state['status']=='completed' and not base.process_retired(identity):return None
        return original_ready(protocol,memory_protocol)
    class Controller(prior.Controller):
        def __init__(self,*parameters):
            super().__init__(*parameters)
            self.state.update(successor_wrapper_sha256=base.owned.digest(Path(__file__)),
                successor_declaration_sha256=followup_sha,continued_memory_controller_registration_sha256=memory_registration_sha)
    prior.PROTOCOL,_=validate(args.protocol);prior.DECLARATION=base.read(prior.PROTOCOL['quarantine_declaration'])
    # In this new process only: old source files and halted queues are unchanged.
    base.validate=validate;base.predecessors_ready=ready;base.Controller=Controller;base.ledger_terminal=prior.scoped_ledger
    base.owned.terminal_parent=lambda protocol,state:successor.terminal_parent(protocol,state,args.followup)
    def interrupted(*_):raise base.SafetyHalt('Continued SDK waiter interrupted; live work preserved')
    signal.signal(signal.SIGTERM,interrupted);base.run(args)


if __name__=='__main__':main()
