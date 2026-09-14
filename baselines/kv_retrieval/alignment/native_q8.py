"""Reuse native packed Q8_0 weights and kernels from the existing qw3 backend.

The existing BF16 InfLLM server is unchanged. This module is a separately
validated candidate for matching its weights to the previous agent runs.
"""
import ctypes
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def pack_native_rows(interleaved, cols):
    """Same lossless split-plane rearrangement as qw3's weight upload."""
    if cols <= 0 or cols % 32:
        raise ValueError('Q8_0 columns must be a positive multiple of 32')
    raw = np.asarray(interleaved)
    if raw.dtype != np.uint8 or raw.size % (cols // 32 * 34):
        raise ValueError('invalid GGUF Q8_0 row bytes')
    blocks = raw.reshape(-1, cols // 32, 34)
    return np.concatenate((blocks[..., :2].reshape(len(blocks), -1),
                           blocks[..., 2:].reshape(len(blocks), -1)), axis=1)


class NativeQ8:
    def __init__(self, library=None):
        # The original dispatch reads these once per process. Reject nondefault
        # layouts rather than hand it a scratch buffer with the wrong format.
        for key, allowed in [('QW3_MMQ_VERSION', ('', 'auto')),
                             ('QW3_VMED', ('', '1')),
                             ('QW3_VMED_MIN_BATCH', ('', '12'))]:
            if os.environ.get(key, '') not in allowed:
                raise ValueError(f'{key} must use the validated default dispatch')
        self.lib = ctypes.CDLL(str(library or ROOT / 'compat_build/native_q8/libaligned_q8.so'))
        pointer, u32 = ctypes.c_void_p, ctypes.c_uint32
        self.lib.aligned_q8_matmul.argtypes = [pointer] * 4 + [u32] * 3 + [ctypes.c_int, pointer]
        self.lib.aligned_q8_matmul.restype = ctypes.c_int
        self.lib.aligned_q8_embedding.argtypes = [pointer] * 3 + [u32] * 3 + [pointer]
        self.lib.aligned_q8_embedding.restype = ctypes.c_int

    @staticmethod
    def check_weight(weight, cols):
        if (weight.dtype != torch.uint8 or not weight.is_cuda or weight.ndim != 2
                or not weight.is_contiguous() or cols % 32 or cols <= 0
                or weight.shape[1] != cols // 32 * 34 or weight.shape[0] == 0):
            raise ValueError('expected contiguous CUDA split-plane Q8_0 rows')

    def matmul(self, weight, x, cols):
        self.check_weight(weight, cols)
        if x.device != weight.device or x.shape[-1] != cols:
            raise ValueError('activation device/shape mismatch')
        if torch.is_grad_enabled() and x.requires_grad:
            raise ValueError('inference only')
        shape = x.shape[:-1]
        x = x.reshape(-1, cols).float().contiguous()
        batch, rows = x.shape[0], weight.shape[0]
        if not batch:
            return torch.empty((*shape, rows), device=x.device, dtype=torch.float32)
        # Native's column-batched MMVQ supports 1..8. For uncommon 9..11 or
        # tiny-row projections, split into <=8 rather than disable its fast path.
        if batch > 8 and (batch < 12 or rows < 64 or cols % 128):
            return torch.cat([self.matmul(weight, chunk, cols)
                              for chunk in x.split(8)]).reshape(*shape, rows)
        mode = 0 if batch <= 8 else (2 if batch <= 64 else 1)
        if mode == 1 and rows < 128:
            # Auto v7 uses ordinary staging below batch 64; this branch is >64.
            assert batch >= 64 and cols % 128 == 0
        scratch = torch.empty(batch * (cols // 32) * 36, dtype=torch.uint8, device=x.device)
        out = torch.empty((batch, rows), device=x.device, dtype=torch.float32)
        with torch.cuda.device(x.device):
            stream = torch.cuda.current_stream(x.device)
            code = self.lib.aligned_q8_matmul(weight.data_ptr(), x.data_ptr(),
                scratch.data_ptr(), out.data_ptr(), rows, cols, batch, mode, stream.cuda_stream)
        if code:
            raise RuntimeError(f'native Q8_0 matmul launch failed: {code}')
        return out.reshape(*shape, rows)

    def embedding(self, weight, ids, cols):
        self.check_weight(weight, cols)
        if ids.device != weight.device or ids.dtype != torch.int64:
            raise ValueError('token IDs must be CUDA int64 on the weight device')
        ids = ids.contiguous()
        out = torch.empty((*ids.shape, cols), device=ids.device, dtype=torch.float32)
        if not ids.numel():
            return out
        with torch.cuda.device(ids.device):
            stream = torch.cuda.current_stream(ids.device)
            code = self.lib.aligned_q8_embedding(weight.data_ptr(), ids.data_ptr(),
                out.data_ptr(), weight.shape[0], cols, ids.numel(), stream.cuda_stream)
        if code:
            raise RuntimeError(f'native Q8_0 embedding launch failed: {code}')
        return out
