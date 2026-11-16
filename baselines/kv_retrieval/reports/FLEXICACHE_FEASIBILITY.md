# FlexiCache feasibility at the pinned revision

Audited author revision: `ab0f495b10fc6509c1637af35779b29121fb7d3f`.
This is a compatibility finding, not a task result or a performance comparison.

The author implementation is a vLLM 0.8.2 fork. Its model registry contains
Qwen2 but not this checkpoint's Qwen3.5-style hybrid architecture. This model
has 48 recurrent layers and 16 full-attention layers; relabeling it as Qwen2
would execute the wrong model. A valid port must preserve recurrent state,
gated full attention, partial rotary dimensions and safe cross-turn restore.
See `upstream/FlexiCache/vllm/model_executor/models/registry.py` and the local
model manifest.

The head profile is part of the method. In
`upstream/FlexiCache/vllm/v1/flexicache/config.py:82`, initialization rejects a
model absent from `model_data`; for nonzero unstable heads, the profile key
also depends on unstable-head count, profiling task and top-k page budget.
The bundled profiles cover Llama 3.1 8B, Mistral 7B, Mistral Small 24B and
Qwen2.5 32B. None is a profile for this checkpoint. Setting every head stable
or borrowing Qwen2.5's profile would remove an essential validated choice and
cannot stand in for the named method's best supported configuration.

The default host pool is 180 GiB (`config.json`; `config.py:73` multiplies by
1024 cubed), larger than this host's 125 GiB RAM. This is configurable and
is not by itself a fundamental obstacle. A 40 GiB pool could be calibrated
after porting, together with GPU storage for unstable heads and staging/index
overhead. The author's documented H100 94 GB / 256 GB host setup also differs
from this Blackwell workstation. Kernel and framework compatibility would
need separate verification.

The shortest valid extension is therefore: port the original full-attention
and offload path, generate a head-stability profile on held-out text, validate
logits/session restoration, then run the same four tasks and memory ceiling.
The needed architecture and profile work is materially larger than launching
an existing executable. We retain this as a potential follow-up rather than
replacing it with an internal KVMem ablation or an unprofiled approximation.
The first direct runnable named-method comparison is InfLLM.

For interpretation, InfLLM already includes CPU offload, an LRU GPU block cache,
asynchronous transfers and batched block scoring; these are enabled in our
adapter. Its author's `context_manager.py:get_batched_topk` computes block
scores/top-k for each execution chunk when global history exceeds the retrieval
budget. With one-token decode calls this occurs per token, in each adapted
full-attention layer. KVMem amortizes selection across a working-set epoch.
FlexiCache directly addresses selection frequency through profiled head
stability, so InfLLM alone cannot establish superiority to every method that
already reuses retrieved KV across tokens. The report must retain this limit.

## Component feasibility experiment (2026-09-12)

Four original CUDA extensions now compile unchanged under torch 2.11 / CUDA
13.0 / SM120. The compile process had no GPU visible. Numerical probes used
tiny allocations with shared-GPU timings excluded:

- The original dispatch chooses a non-padded scoring kernel for GQA group 6;
  Triton rejects its `arange(6)`. Calling the author's existing padded kernel
  works. With 24 query heads, 4 KV heads and dimension 256, scores match a CPU
  float32 reference (maximum absolute error 0.00006103515625). Stable-head
  rerank cadence, unstable-head updates and the prefill gate also pass.
- Original pinned-host H2D and D2H kernels pass bitwise KV-copy checks, including
  chunking and background-stream ordering. Dirty block-table copies preserve
  untouched ranges. The top-k swap-map extension compiles but is not yet
  numerically validated. Sparse attention parity is also still pending.

The source audit exposes two additional integration contracts. The author
scores stable heads at a rerank step, updates top-k after the last layer, and
reloads through the model runner for subsequent execution. Updating/reloading
each layer immediately before its current attention would change this timing.
Also, the multi-token prefill branch reads the full context block table;
simply giving it the sparse resident decode table is incorrect. Incremental
tool observations therefore need an explicit supported prefill/residency path.
`topk_swap_map.cu` enforces at most 256 selected pages, so a 32K token budget
would need a validated larger page size instead of passing 1024 32-token pages.

These checks show usable original components, but do not constitute a runnable
hybrid-model baseline. Raw results are linked in
`reports/flexicache_component_summary_20260912.json`; the exact probe source
versions are archived under `reports/component_sources/`.
