"""Isolated arithmetic variant: same reduction tree for decode and verify."""
from common import HERE, BASE
import hashlib
import json
import subprocess
import time

source = BASE/'native_source/src'
out = HERE/'q8_consistent'
out.mkdir(exist_ok=True)
hashes = {}
for name in ('mmvq_q8.cu','mmq_q8.cu','cuda_helpers.cuh'):
    data = (source/name).read_text()
    hashes[name] = hashlib.sha256(data.encode()).hexdigest()
    if name == 'mmvq_q8.cu':
        before = 'return NCOLS_DST <= 4 ? 4 : (NCOLS_DST <= 8 ? 2 : 1);'
        assert data.count(before) == 1
        data = data.replace(before,'return 4; // external MTP: identical reduction across 1..8 columns')
        # Prevent template-dependent multiply/add contraction at rounding ties.
        before = 'tmp[j][i] += vec_dot_q8_0_q8_1(qs_u8, d_w, ya, kqs);'
        assert data.count(before) == 1
        data = data.replace(before,'tmp[j][i] = __fadd_rn(tmp[j][i], vec_dot_q8_0_q8_1(qs_u8, d_w, ya, kqs));')
        before = 'return d_w * d_x * static_cast<float>(sumi);'
        assert data.count(before) == 1
        data = data.replace(before,'return __fmul_rn(__fmul_rn(d_w, d_x), static_cast<float>(sumi));')
    (out/name).write_text(data)
(out/'bridge.cu').write_bytes((BASE/'alignment/native_q8_bridge.cu').read_bytes())
command = ['/usr/local/cuda-13.0/bin/nvcc','-std=c++17','-O3','-arch=sm_120a',
           '--shared','-Xcompiler','-fPIC','-I',str(out),str(out/'bridge.cu'),
           str(out/'mmvq_q8.cu'),str(out/'mmq_q8.cu'),'-o',str(out/'libconsistent_q8.so')]
tick = time.time()
result = subprocess.run(command,capture_output=True,text=True)
(out/'build.json').write_text(json.dumps(dict(command=command,started=tick,ended=time.time(),
    original_sha256=hashes,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr),indent=2)+'\n')
print(result.stderr)
raise SystemExit(result.returncode)
