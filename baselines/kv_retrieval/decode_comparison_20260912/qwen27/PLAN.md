# Qwen3.8-27B matched-model throughput

All primary arms use the original local Qwen3.8-27B BF16 safetensors, FP8 E4M3FN
KV storage, batch one, greedy fixed-length generation, no MTP, no vision tower,
no NVMe, and exclusive GPU ownership. Bind all processes to GPU-local NUMA 0
CPUs (0-15,32-47). Do not alter the production engine or OSWorld settings.

External arms use HF/FLA/PyTorch model computation. InfLLM imports the original
ContextManager, block selection/replacement and Triton attention. Its Qwen
boundary supports hybrid recurrence, head normalization/gate, partial RoPE and
256-dimensional heads; FP8 storage is an explicitly documented extension.
InfiniGen uses the isolated Qwen port of the original threshold/mean-count/top-k
selector, cross-layer partial-query forecast and real CPU KV gathering/prefetch.
Calibration is repeated on BF16 weights using the author's PG19 setup text.
These are Qwen adaptations, not untouched upstream end-to-end implementations.

The native BF16 container has a copied config with dense compressed-tensors
format metadata and symlinks to the exact original safetensors. The frozen qw3
loader supports those tensor types but requires this format tag. No tensor is
converted or quantized and the original model directory remains unchanged.
Loader validation and short generation checks must pass before formal timing.

Measurements: 8K, 32K, 120K and 240K histories from the same real DeepSWE request;
identical token IDs and current query suffix. Primary output length 128, three
repetitions after warmup. Include selection/prefetch/transfers during decode;
report loading, calibration, prefill and initial KVMem selection separately.
KVMem budgets remain 128K selection + 128K generation, block 32, CPU history and
index, full query replay. It receives an explicit current-query span; this
decode benchmark does not measure private guided-query generation stalls.

Report InfLLM's native small attention set (~6.3K) separately from a bounded
128K-budget check. InfiniGen's adaptive cap does not guarantee that many tokens
are actually selected; report actual selected counts and transferred bytes.
Retain a same-model HF dense GPU/full CPU-offload control. Backend, residual and
attention-accumulation precision differences remain explicit; a same-model
system throughput ratio is not a proof of isolated KV algorithm speedup.

Prerequisites: full-selection/dense logits and CPU KV agreement for InfiniGen;
original attention oracle agreement for InfLLM; graph output AND recurrent
state agreement; no qw3 Q8 projection library in the external process.

After exporting results, resume OSWorld from interrupted 002/KVMem and verify
actual progress. Preserve all completed arms and archived interrupted data.
