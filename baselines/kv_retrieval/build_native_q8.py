"""Compile the existing Q8 kernels behind a small ABI; no GPU initialization."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    source = ROOT / 'native_source/src'
    build = ROOT / 'compat_build/native_q8'
    build.mkdir(parents=True, exist_ok=True)
    inputs = [ROOT / 'alignment/native_q8_bridge.cu', source / 'mmvq_q8.cu',
              source / 'mmq_q8.cu']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in inputs + [source / 'cuda_helpers.cuh']}
    # Source snapshot already validated/frozen for previous calibration.
    frozen = json.loads((ROOT / 'reports/native_source_manifest.json').read_text())
    for path in [source / 'mmvq_q8.cu', source / 'mmq_q8.cu', source / 'cuda_helpers.cuh']:
        key = str(path.relative_to(ROOT / 'native_source'))
        if hashes[str(path.relative_to(ROOT))] != frozen['original_sha256'][key]:
            raise RuntimeError(f'frozen native kernel changed: {key}')
    record = dict(started=time.time(), component_only=True, source_sha256=hashes,
                  frozen_source_manifest_sha256=hashlib.sha256(
                      (ROOT / 'reports/native_source_manifest.json').read_bytes()).hexdigest())
    command = ['/usr/local/cuda-13.0/bin/nvcc', '-std=c++17', '-O3', '-arch=sm_120a',
               '--shared', '-Xcompiler', '-fPIC', '-I', str(source),
               *map(str, inputs), '-o', str(build / 'libaligned_q8.so')]
    record['command'] = command
    result = subprocess.run(command, text=True, capture_output=True)
    record.update(finished=time.time(), returncode=result.returncode,
                  compiler_stdout=result.stdout, compiler_stderr=result.stderr,
                  status='compiled' if result.returncode == 0 else 'failed')
    if result.returncode == 0:
        record['library_sha256'] = hashlib.sha256((build / 'libaligned_q8.so').read_bytes()).hexdigest()
    args.out.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k not in ['source_sha256', 'command']}))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
