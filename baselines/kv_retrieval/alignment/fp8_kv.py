"""FP8 storage boundary around the pinned author InfLLM implementation.

Raw E4M3FN, scale=1 and saturate-finite, matching qw3's KV format. Author
selection, positional rules, cache policy and fused attention are unchanged.
Local persistent KV, CPU history and GPU block cache use FP8. The author's
assembled attention buffer and transient computation views remain FP16.
"""
import torch
from kv_baselines.upstream import infllm_module

DTYPE = torch.float8_e4m3fn


def encode(x):
    if x.dtype == DTYPE:
        return x
    return x.clamp(-448, 448).to(DTYPE)


def install_fp8_storage():
    cm = infllm_module('context_manager')
    if getattr(cm, '_baseline_fp8_storage', False):
        return
    original_cache, original_unit, original_append = cm.CudaCache, cm.MemoryUnit, cm.ContextManager.append

    class FP8CudaCache(original_cache):
        def __init__(self, num_units, unit_size, dtype):
            super().__init__(num_units, unit_size, DTYPE)

    class FP8MemoryUnit(original_unit):
        def __init__(self, kv, cache, load_to_cache=False, pin_memory=False):
            super().__init__(tuple(encode(t) for t in kv), cache, load_to_cache, pin_memory)

    def append(self, lq, lk, lv, gq, gk, gv):
        # Unrotated raw K is InfLLM's authoritative historical representation.
        # Round only once; subsequent FP8 -> FP16 -> FP8 is exact.
        if lk.data_ptr() != gk.data_ptr() or lv.data_ptr() != gv.data_ptr():
            raise ValueError('FP8 adapter requires identical local/global raw KV')
        lk, lv = (encode(t).to(lq.dtype) for t in (lk, lv))
        gk, gv = lk, lv  # validated adapter passes identical local/global KV
        if self.initialized:
            for name in ('local_k', 'local_v', 'init_k', 'init_v'):
                setattr(self, name, getattr(self, name).to(lq.dtype))
            self.global_remainder = tuple(t.to(lq.dtype) for t in self.global_remainder)
        output = original_append(self, lq, lk, lv, gq, gk, gv)
        if self.async_global_stream:
            torch.cuda.current_stream().wait_stream(cm.GLOBAL_STREAM)
        for name in ('local_k', 'local_v', 'init_k', 'init_v'):
            setattr(self, name, encode(getattr(self, name)))
        self.global_remainder = tuple(encode(t) for t in self.global_remainder)
        return output

    cm.CudaCache, cm.MemoryUnit = FP8CudaCache, FP8MemoryUnit
    cm.ContextManager.append = append
    cm._baseline_fp8_storage = True
