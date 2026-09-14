"""Build unchanged author CUDA sources for the installed GPU architecture only."""
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('MAX_JOBS', '4')
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '12.0')
import torch
from torch.utils.cpp_extension import load

HERE = Path(__file__).resolve().parent
source = HERE / 'vendor/causal-conv1d'
build = HERE / 'vendor/conv_build'
build.mkdir(exist_ok=True)
files = [source / 'csrc' / n for n in ('causal_conv1d.cpp', 'causal_conv1d_fwd.cu',
                                      'causal_conv1d_bwd.cu', 'causal_conv1d_update.cu')]
start = time.time()
module = load(name='causal_conv1d_cuda', sources=[str(p) for p in files],
              extra_include_paths=[str(source / 'csrc')], build_directory=str(build),
              extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '--use_fast_math',
                  '--expt-relaxed-constexpr', '--expt-extended-lambda',
                  '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__',
                  '-U__CUDA_NO_BFLOAT16_OPERATORS__', '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                  '-U__CUDA_NO_BFLOAT162_OPERATORS__', '-U__CUDA_NO_BFLOAT162_CONVERSIONS__'],
              verbose=True)
(HERE / 'conv_build.json').write_text(json.dumps({'passed': True, 'seconds': time.time()-start,
    'torch': torch.__version__, 'cuda': torch.version.cuda, 'module': module.__file__,
    'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    'scope': 'Unchanged causal-conv1d 1.6.2.post1; SM120 compilation, no global environment install'}, indent=2)+'\n')
