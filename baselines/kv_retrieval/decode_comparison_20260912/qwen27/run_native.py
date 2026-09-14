"""Separate Qwen/KVMem deployment reference; never an upstream method backend."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import traceback
import urllib.request

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['qw3', 'kvmem'], required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--lengths', type=int, nargs='+', default=[8192, 32768, 122880, 245760])
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--port', type=int, default=18319)
    p.add_argument('--mtp', type=int, choices=[0, 4], default=0)
    p.add_argument('--tokens', type=int, default=128)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    plan = json.loads((ROOT / 'benchmark/osworld_v2/claudecode_remaining_20260912/plan.json').read_text())
    arm = plan['arms'][args.method]
    command = list(arm['command'])
    command[command.index('--model') + 1] = str(HERE / 'qwen27/native_bf16_container')
    # This throughput workload is text-only; avoid loading an unused vision tower.
    for flag in ['--vision-model', '--vision-device']:
        i = command.index(flag)
        del command[i:i+2]
    for flag, value in [('--port', str(args.port)), ('--temp', '0'), ('-n', str(args.tokens)), ('--mtp-chain', str(args.mtp))]:
        command[command.index(flag) + 1] = value
    if args.mtp == 0 and args.method == 'kvmem':
        # Every request supplies its actual query span. Full semantic replay
        # remains enabled; this control requires no privately generated query.
        command[command.index('--kvmem-guided-reselect') + 1] = 'off'
    env = dict(os.environ)
    for key in list(env):
        if key.lower() in ['http_proxy', 'https_proxy', 'all_proxy'] or key.startswith('QW3_'):
            env.pop(key)
    env.update(arm['environment'])
    if args.mtp == 0:
        env['QW3_KVMEM_GUIDED_MTP'] = '0'
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', args.port))
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('Exclusive GPU required')
    data = json.loads((HERE / 'inputs.json').read_text())
    source = data['evaluation']['input_ids']
    suffix = data['suffix']
    extra = suffix['before'] + suffix['query'] + suffix['after']
    report = {'status': 'starting', 'command': command, 'environment': arm['environment'], 'results': [],
              'binary_sha256': hashlib.sha256(Path(command[0]).read_bytes()).hexdigest(),
              'input': {k: v for k, v in data['evaluation'].items() if k != 'input_ids'},
              'scope': f'Qwen3.8-27B original BF16 safetensors / FP8; same weights as HF external methods; native FP32 residual differs from HF BF16',
              'mtp_chain': args.mtp, 'query_mode': 'explicit current query span; full query replay',
              'timing': 'Native decoder accounting; prefill/private query/replay reported separately. Fixed 128 output tokens, greedy, ignore EOS.',
              'started': time.time()}
    def save():
        (args.out / 'result.json').write_text(json.dumps(report, indent=2))
    report['cpu_affinity'] = sorted(os.sched_getaffinity(0))
    report['weight_container'] = json.loads((HERE / 'qwen27/native_bf16_container_manifest.json').read_text())
    report['harness_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save()
    log = (args.out / 'server.log').open('w')
    server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=log)
    report['server_pid'] = server.pid
    save()
    http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base = f'http://127.0.0.1:{args.port}'
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f'server exited: {server.returncode}')
            try:
                with http.open(base + '/health', timeout=2) as r:
                    if r.status == 200:
                        break
            except OSError:
                pass
            time.sleep(.5)
        else:
            raise TimeoutError('server readiness')
        for length in [1024] + args.lengths:
            assert length < len(source)
            ids = source[:length-len(extra)] + extra
            begin = length - len(suffix['query']) - len(suffix['after'])
            for rep in range(1 if length == 1024 else args.reps):
                body = {'model': 'qw3', 'prompt': ids, 'max_tokens': args.tokens, 'temperature': 0,
                        'ignore_eos': True, 'stream': False, 'seed': 0}
                if args.method == 'kvmem':
                    body['kvmem_query_token_span'] = {'begin': begin, 'end': begin + len(suffix['query'])}
                offset = (args.out / 'server.log').stat().st_size
                started = time.perf_counter()
                req = urllib.request.Request(base + '/v1/completions', data=json.dumps(body).encode(),
                                             headers={'Content-Type': 'application/json'})
                with http.open(req, timeout=900) as r:
                    response = json.load(r)
                wall_s = time.perf_counter() - started
                (args.out / f'response_{length}_{rep}.json').write_text(json.dumps(response, indent=2))
                log.flush()
                with (args.out / 'server.log').open('rb') as f:
                    f.seek(offset)
                    lines = f.read().decode(errors='replace').splitlines()
                phases = {}
                for line in lines:
                    nums = {k: float(v) for k, v in re.findall(r'\b([A-Za-z_][A-Za-z_0-9]*)=(-?\d+(?:\.\d+)?)', line)}
                    if '[qw3-native-accounting]' in line or 'native generate: prompt_tokens=' in line:
                        phases.update(nums)
                        if 'native generate: prompt_tokens=' in line:
                            phases.setdefault('prefill_ms', nums['prefill'] * 1000)
                            phases.setdefault('decode_ms', nums['decode'] * 1000)
                    elif 'native continuous_batch: request=' in line:
                        phases.update(nums)
                        phases.update(prefill_ms=nums['prefill']*1000, decode_ms=nums['decode']*1000)
                    elif 'native mtp_spec_summary:' in line:
                        phases.update({f'mtp_{k}': v for k, v in nums.items()})
                row = {'prompt_tokens': length, 'rep': rep, 'warmup': length == 1024, 'input_ids_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(), 'request_wall_s': wall_s,
                       'usage': response.get('usage'), 'phases': phases}
                report['results'].append(row)
                save()
                print(json.dumps(row), flush=True)
                if response.get('usage', {}).get('completion_tokens') != args.tokens:
                    raise AssertionError('Did not generate the full requested length')
        report['status'] = 'complete'
    except BaseException as exc:
        report.update(status='error', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        log.close()
        report['ended'] = time.time()
        save()


if __name__ == '__main__':
    main()
