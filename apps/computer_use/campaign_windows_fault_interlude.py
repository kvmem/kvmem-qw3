"""Reuse the frozen scheduler handoff for one separate Windows recovery pilot."""
import argparse
from pathlib import Path
import sys
from unittest.mock import patch

import campaign_interlude as interlude
from osworld_linux.runner import request


def run(args):
    args.case='windows_browser_fault';args.repair_fixture=args.acceptance_manifest=args.ffmpeg=None
    args.engine=args.engine.resolve()
    args.bundle=args.bundle.resolve()
    def specification(value):
        # The host fixture uses the outer identity, since "pilot" is shared
        # across runs. Give the evidence directory its own unique short name.
        destination=value.output/(value.output.name+'-pilot')
        return ([sys.executable,'-m','windows_fault_pilot','--app',args.app,'--runtime',str(args.app_runtime),
            '--output',str(destination),'--bundle',str(args.bundle),'--fault',args.fault,'--seconds',str(args.seconds),
            '--admission',str(args.output/'interlude.json')],destination/'integration-result.json')
    def terminal(_,result):
        state=request(args.app,'/api/state')
        return (result.get('owned_task_retired') is True and not result.get('owned_task_stop_unconfirmed')
                and state['status']=='idle' and not state.get('uncertain_actions'))
    cases=interlude.WINDOWS_CASES|{'windows_browser_fault':('browser',False)}
    with patch.object(interlude,'WINDOWS_CASES',cases),patch.object(interlude,'pilot_spec',specification),\
            patch.object(interlude,'terminal_pilot',terminal):
        return interlude.run(args)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--app-runtime',type=Path,required=True)
    parser.add_argument('--app',default='http://127.0.0.1:18205')
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--engine',type=Path,required=True)
    parser.add_argument('--fault',choices=['slow_loading','focus_dialog','no_progress'],required=True)
    parser.add_argument('--seconds',type=int,default=600)
    args=parser.parse_args()
    if not 60<=args.seconds<=900:parser.error('Use60..900seconds')
    result=run(args)
    print({key:result.get(key) for key in ['status','error','scheduler_resumed_at']},flush=True)
