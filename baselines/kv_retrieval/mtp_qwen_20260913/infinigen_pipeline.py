"""Pipeline per-token author forecasts over a causally bounded speculative tail.

The next layer's K/V projections are already available for the whole verify
batch. Stage them once, then forecast i with index[:base+i]. Future candidates
are never visible to its selector. One author worker/stream retains ordering.
"""
import copy
import time
import torch
from qwen.backend import encode, attention, FP8


def enqueue(backend, module, state, previous_norm, rope, k, v):
    assert state.finalized and not state.config.dense and state.future is None
    base, count = state.length, k.shape[-2]
    assert base+count <= state.config.capacity
    quant_k, quant_v = encode(k), encode(v)
    packed = torch.stack((quant_k,quant_v)).view(torch.uint8)[:,0].transpose(1,2)
    state.cpu_kv[:,base:base+count].copy_(packed,non_blocking=True)
    for pos in range(count):
        state.partial_keys[:,base+pos:base+pos+1].copy_(
            state.project_keys(quant_k[0,:,pos:pos+1].transpose(0,1)))
    ready = torch.cuda.Event()
    ready.record(torch.cuda.current_stream())
    futures = []
    for pos in range(count):
        view = copy.copy(state)
        view.length = base+pos
        # The single executor and single prefetch stream serialize forecast
        # kernels. The following selector's CPU synchronization also completes
        # the previous staging H2D before that pinned buffer is reused.
        futures.append(backend.executor.submit(backend._forecast,module,view,
            previous_norm[:,pos:pos+1],tuple(t[:,pos:pos+1] for t in rope),ready))
    return futures, quant_k, quant_v


def consume(state, future, q, quant_k, quant_v, scaling):
    tick = time.perf_counter()
    ready, selected, stats = future.result()
    stream = torch.cuda.current_stream()
    stream.wait_event(ready)
    selected.record_stream(stream)
    full = selected.view(FP8).permute(0,2,1,3).unsqueeze(1)
    groups = state.qh//state.kh
    k = torch.cat((full[0].half(),quant_k.half().repeat_interleave(groups,dim=1)),dim=2)
    v = torch.cat((full[1].half(),quant_v.half().repeat_interleave(groups,dim=1)),dim=2)
    output = attention(q,k,v,scaling)
    state.length += 1
    stats['consume_host_s'] = time.perf_counter()-tick
    state.metrics.append(stats)
    return output
