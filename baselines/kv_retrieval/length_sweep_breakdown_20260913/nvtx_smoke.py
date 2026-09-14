"""CPU-only check that the capture-trigger NVTX range survives export."""
import time
import torch
assert not torch.cuda.is_initialized()
torch.cuda.nvtx.range_push('measured_query')
time.sleep(.1)
torch.cuda.nvtx.range_pop()
assert not torch.cuda.is_initialized()
