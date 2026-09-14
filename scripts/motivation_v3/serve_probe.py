"""Launch one isolated dense-attention probe and archive its configuration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import shutil


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--binary',type=Path,required=True);ap.add_argument('--port',type=int,default=18190)
    ap.add_argument('--model',default='models/Qwen3.8-27B-Q8_0.gguf')
    ap.add_argument('--vision-model',default='models/Qwen3.8-27B-BF16')
    ap.add_argument('--vision-worker',type=Path)
    ap.add_argument('--prefill-tokens',type=int,default=64)
    ap.add_argument('--max-tokens',type=int,default=2048)
    ap.add_argument('--thinking-budget',type=int,default=512)
    args=ap.parse_args();root=Path(__file__).resolve().parents[2]
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'server_manifest.json').exists():raise FileExistsError('Use a fresh server output directory')
    env={k:v for k,v in os.environ.items() if not k.startswith('QW3_')}
    worker=out/'qw3_vision_cpu_worker.py'
    # A campaign pins this file once so later batches do not follow workspace edits.
    pinned_worker=out.parent/'vision_worker_snapshot.py'
    worker_source=args.vision_worker or (pinned_worker if pinned_worker.exists()
        else root/'scripts/qw3_vision_cpu_worker.py')
    shutil.copy2(worker_source,worker)
    selected={'QW3_ATTN_TRACE':str(out/'attention.jsonl'),
        'QW3_ATTN_TRACE_INPUT':str(out/'inputs.jsonl'),
        'QW3_ATTN_TRACE_PREFILL_TOKENS':str(args.prefill_tokens),
        'QW3_ATTN_TRACE_INTERVAL':'1','QW3_ATTN_TRACE_BLOCK_TOKENS':'128','QW3_GRAPH':'0',
        'QW3_VISION_CPU_PYTHON':str(root/'.venv/bin/python'),
        'QW3_VISION_CPU_WORKER':str(worker)}
    env.update(selected)
    cmd=[str(args.binary.resolve()),'serve','--model',str((root/args.model).resolve()),
        '--vision-model',str((root/args.vision_model).resolve()),'--vision-device','cuda',
        '--host','127.0.0.1','--port',str(args.port),'--ctx','65536','--kv-dtype','fp16',
        '--prefill-chunk','2048','--mtp-chain','0','--temp','0','--enable-thinking',
        '--thinking-budget',str(args.thinking_budget),'-n',str(args.max_tokens)]
    manifest={'command':cmd,'trace_environment':selected,'started_at':time.time(),
        'binary_sha256':hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        'vision_worker_sha256':hashlib.sha256(worker.read_bytes()).hexdigest(),
        'vision_worker_source':str(worker_source.resolve()),
        'model_size':(root/args.model).stat().st_size,
        'mode':'dense, serialized, full-visible, no KVMem; prefill-tail diagnostic'}
    (out/'server_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with (out/'server.log').open('w') as log, (out/'gpu_samples.csv').open('w') as gpu_log:
        p=subprocess.Popen(cmd,env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,memory.used,utilization.gpu',
            '--format=csv','--loop=2'],stdout=gpu_log,stderr=subprocess.STDOUT)
        (out/'server.pid').write_text(str(p.pid)+'\n')
        print(f'server pid {p.pid}; log {out}/server.log',flush=True)
        try:code=p.wait()
        except KeyboardInterrupt:
            p.terminate();code=p.wait()
        finally:
            monitor.terminate();monitor.wait()
        manifest.update(exit_code=code,finished_at=time.time())
        (out/'server_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        raise SystemExit(code)


if __name__=='__main__':main()
