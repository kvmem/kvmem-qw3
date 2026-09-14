"""Allow a longer model-readiness wait without editing the frozen agent harness."""
import argparse
from pathlib import Path
import sys
import types

import run_aligned_infllm_arm as aligned


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--startup-timeout-s',type=int,required=True)
    args,rest=parser.parse_known_args()
    if not 240 < args.startup_timeout_s <= 1800:
        raise ValueError('startup retry deadline must be in (240,1800] seconds')
    path=Path(__file__).resolve().parent/'run_pilot_arm.py'
    source=path.read_text()
    original='deadline = time.monotonic() + 240'
    if source.count(original)!=1:
        raise ValueError('unexpected frozen readiness loop')
    module=types.ModuleType('_infllm_readiness_retry')
    module.__file__=str(path)
    exec(compile(source.replace(original,
         f'deadline = time.monotonic() + {args.startup_timeout_s}'),str(path),'exec'),module.__dict__)
    aligned.arm=module
    sys.argv=[sys.argv[0],*rest]
    aligned.main()


if __name__=='__main__':main()
