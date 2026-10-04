"""Reuse the frozen scheduler's proven interlude for a separate recovery probe."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import campaign_interlude as interlude
from osworld_linux.runner import save


def run(args):
    original=interlude.pilot_spec
    def specification(value):
        command,result=original(value)
        command+=['--fault',args.fault,'--retire-task']
        return command,result
    with patch.object(interlude,'pilot_spec',specification):result=interlude.run(args)
    result.update(fault_condition=args.fault,
        fault_interlude_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save(args.output/'interlude.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--app-runtime',type=Path,required=True)
    parser.add_argument('--app',default='http://127.0.0.1:18205')
    parser.add_argument('--case',choices=['linux_browser','android_browser'],required=True)
    parser.add_argument('--fault',choices=['slow_loading','focus_dialog','no_progress'],required=True)
    parser.add_argument('--seconds',type=int,default=600)
    args=parser.parse_args()
    if not 60<=args.seconds<=900:parser.error('Use60..900seconds')
    result=run(args)
    print({key:result.get(key) for key in ['status','error','fault_condition','scheduler_resumed_at']},flush=True)
