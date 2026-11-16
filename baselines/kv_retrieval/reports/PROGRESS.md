# Baseline implementation progress — 2026-09-12

The goal remains active. **No completed agent-level comparison and no formal
latency result yet.** Current runs share the GPU with an unrelated workload;
their times must not be used as baseline performance claims.

Validated so far:

- Exact pinned InfLLM ContextManager, representative selection, fused two-stage
  attention, LRU GPU cache, CPU offload, async stream and pinned-memory path.
- Qwen3.8 attention boundary: partial rotary channels, GQA, Q/K normalization,
  output gating, 16 full-attention layers and unchanged recurrent layers.
- Safe session state: recent tensors and recurrence are snapshotted; immutable
  CPU blocks are shared. Restores discard later checkpoints/GPU residency. A
  short append from an unaligned live prefix retains an older safe checkpoint.
- Eleven module/session/precision checks pass in the isolated FLA environment.
- The real BF16 27B model completed short generation and 8,192-token prefill;
  all 16 attention layers reached 31 historical blocks and performed top-16
  retrieval. Peak reserved memory was 51.54 GiB in this shared-GPU check.
- The isolated native BF16 build passes source geometry and conversion checks.
  Its CPU-only native prompt/tool bridge now passes seven tests, including
  first-content emission, UTF-8/token boundaries, multiple tools, and explicit
  truncated-tool failure. Three generation/validation checks also pass.
- Short native/InfLLM numerical investigation: with BF16 output logits, one
  close pair became tied (native scores 25.698 vs 25.6557). The adapter now emits
  FP32 LM-head logits directly from BF16 weights without copying weights to
  FP32. All four combinations of reference/FLA recurrence and gated norm then
  match the native argmax on all 12 identical teacher-forced positions. This
  is a bounded sanity check, not proof of universal token identity.

New integration evidence:

- The production native BF16 server completed four requests with no speculative
  proposals. Its last request generated a 79-token guided query and performed
  selection plus query replay. This is a synthetic observation fixture, not a
  task result. See `native_guided_smoke_20260912_v4/audit.json`.
- Real-model InfLLM HTTP validation completed five requests. A 13K-token history
  reached 68 global blocks in every attention layer. The next request reused
  12,980 tokens and prefills 30; a small tail rewrite then restored the 12,800
  checkpoint and prefills 217. See `http_smoke_shared_gpu_20260912_v2/result.json`.
  The earlier failed report is retained: its 16-token limit cut a tool call in
  half. Validation now allows complete tool generation, and generation continues
  to EOS/output budget rather than stopping at the first tool closing tag.
- Real Claude Code 2.1.148 used Read, received the fixture content, and returned
  the correct line. Its second model request reused 7,201 tokens and prefills
  27. See `claude_smoke_shared_gpu_20260912/result.json`. This checks protocol
  integration, not DeepSWE success.
- Captured the current client's default 23-tool schema, with Agent disabled.
  A separate termenv transcript reconstructs 62 fixed text requests spanning
  19,992 to 130,655 tokens. System/tool scaffolding comes from the current frozen
  capture; **these are reconstructed inputs, not original wire request bodies**.
  No repeated padding or tool execution is added. The task is outside the four
  frozen pilot tasks. A shorter six-request source is retained but will not be
  the primary calibration trace. All 62 primary requests pass tool-reference,
  body-hash, token-count and prefix-integrity checks.

Next gates, in order:

1. Complete paired capacity calibration and component/decode instrumentation.
   Fixed TTFT probes generate one token and do not claim decode throughput.
   Keep truncated-tool errors explicit in any longer-output probes.
2. Calibrate two physically matched memory budgets on held-out fixed text
   histories. Include frontend and dense-backend controls; do not attribute
   HF/native implementation differences solely to retrieval algorithms.
3. Run `tasks_pilot4.json`, one seed, original tools/judges, without injected
   history. All four verifier images are already local. Three main conditions:
   native compaction, adapted InfLLM, full guided KVMem. No pilot outcomes exist.
4. FlexiCache requires Qwen3.8 integration and independent head profiling; the
   author source lacks a profile for this checkpoint. Audit is recorded, and
   no synthetic approximation is presented as a FlexiCache result.

All experimental changes stay in `baselines/kv_retrieval/`. The shared source,
Python environment and prior experiment artifacts are preserved. Downloads
removed proxy variables. Package wheel hashes and upstream commits are recorded.

Capacity calibration update:

- InfLLM completed all 62 reconstructed fixed-input probes, through 130,655
  tokens, using the author default retrieval/local/cache budgets. There was
  one initial cold start and zero unexpected cold prefills; all later requests
  used the live prefix. Each attention layer reached 987 global blocks. Peak
  PyTorch reserved memory was 54.457 GiB (model included). These one-token
  shared-GPU probes do not measure sustained decode or establish task quality.
- Native guided KVMem capacity calibration is now running on the same inputs,
  in `results/native_kvmem_capacity_trace_shared_20260912_v3`. The first launch
  stopped at CLI validation because the default middecode trigger exceeded the
  small test generation budget; the rerun explicitly sets a 1,024-token trigger.
- No matched-budget performance ratio and no full pilot task result yet.

The native v2 preflight also rejected 16 GiB CPU capacity for a full 256K
inclusive K+V backing. v3 raises the CPU cap to 24 GiB; proactive stage-out
remains enabled. Both rejected startup logs are retained; neither processed
a benchmark request. Actual resident bytes, not this cap, must be matched.

The native v3 run completed two requests, then the 4K selection budget hit
the mandatory-block capacity guard after guided-query generation. This budget
was only valid for the smaller smoke fixture, not the 20K Claude scaffolding.
The current v4 capacity run uses 32K selection + 8K generation reserve, leaving
all safety guards and query/replay computations intact. It is not a matched
budget comparison with the completed author-default InfLLM trace.

The 32K/8K v4 startup hit the conservative 0.96 VRAM planner ceiling
(1,246 available blocks versus 1,280 required). v5 retains the same requested
KV pool and raises the planning ceiling to 0.98 after a free-memory check.
This remains a shared-GPU capacity validation, not a formal latency run.


Latest calibration and harness update:

- Native v5 stopped on request 10: end-of-turn promotion replaced an actual
  newer aligned prompt checkpoint with an older postselection origin. The
  isolated patch chooses the newest aligned, current-generation actual state
  before source-index/MTP validation. A CPU regression fixture and rebuild pass.
- Native v6 then completed 49 requests, with 49 subsequent safe reuse attempts
  and zero logged unexpected checkpoint misses. Request 50 exposed a separate
  private-query capacity error: the existing mutation guard disables pressure
  handling inside a reversible probe, but its prompt/query had not been admitted
  beforehand. The isolated capacity patch moves admission before guard/snapshot
  using the actual private prompt length and the full query/thinking limits.
  Full query generation and replay remain intact. v7 is validating that patch.
- InfLLM's larger candidate (4096 local, 224 retrieved blocks, 256 cache blocks,
  block size 128) completed all 62 requests through 130655 tokens. One initial
  cold prefill, zero unexpected cold prefills. Peak allocator reservation was
  57.994 GiB. Final unique live GPU cache storage was about 4.67 GiB, with about
  1.77 GiB additional checkpoint storage and 7.71 GiB pinned CPU history.
  The earlier 57 GiB allocator-cap failure is preserved; 58 GiB passed.
- Storage inventory counts backing allocations once across views and snapshots;
  its optional CPU traversal runs outside request timing and is off for agents.
- The original DeepSWE cell/Claude scripts are snapshotted under `harness/`,
  with source hashes. `run_pilot_arm.py` owns its server/network/container run;
  `trace_proxy.py` archives exact request bytes and records first meaningful
  streamed content and first action. The protocol and parameter tests pass.
  `configs/pilot_candidate.json` is deliberately not frozen until calibration
  and sustained decode checks finish. No full pilot task has started.
- `profile_decode.py` prepares equal recorded teacher-forced continuations for
  dense HF and two original InfLLM configurations, with a separate intrusive
  profiler pass. Dense HF uses BF16 KV and is explicitly an implementation
  control, not a matched-precision main baseline. This diagnostic has not run yet.

Native v7 completed **62/62** requests. All 61 resumed requests chose safe
origins, with zero unexpected checkpoint misses; the previous request-50
private-query exhaustion now invokes capacity recovery before the reversible
probe and completes. Audit: `native_capacity_regression_20260912.json`.
The isolated source manifest marks both fixes validated by this trace.
Teacher-forced decode/component controls are now running at a held-out 38K
prefix, above both InfLLM retrieval budgets. Still no full agent outcomes or
formal shared-GPU latency claims.

A sequential calibration queue follows the current decode diagnostic: InfLLM
24K at a 58 GiB process cap; native 32K selection + 32K generation, 16K
refresh trigger, 512K logical context and CPU-only backing at a 60 GiB cap;
then a 64K dense-native prefix control. Each checks remaining GPU memory,
monitors process occupancy, and stops on failure. Only experimental processes
are terminated. These are still capacity/diagnostic runs, not formal latency.


Complete-task launch preparation:

- Three held-out teacher-forced decode controls completed (64 measured tokens,
  plus a separate 16-token intrusive CPU/CUDA profiler pass). Raw shared-GPU
  token times were 107.5 ms (dense HF, BF16 KV), 126.8 ms (InfLLM defaults), and
  205.2 ms (larger InfLLM candidate). These are **not formal speed ratios**.
  The conv1d CPU operator itself is about 1.5 ms/token in the profiler; attention,
  CPU/GPU transfers and dispatch account for larger differences in this fixture.
- The 24K InfLLM candidate completed 62/62 under a 58 GiB process cap (peak
  57.512 GiB). The 32K+32K KVMem pilot candidate completed 62/62 under 60 GiB
  (peak 58.959 GiB), with four valid guided queries and no original-query fallback.
- Dense native's first command accidentally used the plain route without prefix
  caching. That run was stopped and is excluded. The corrected continuous,
  single-active, prefix-cache-enabled route completed all 19 trace requests
  below 64K, with reuse on all requests after the initial one (peak 58.816 GiB).
- Sequential startup now waits for actual free GPU memory, since a profiler can
  write its final result before CUDA cleanup finishes. Port preflight also waits
  for reuse after server shutdown. All unsuccessful attempts are retained.
- Original model shards and configuration/tokenizer JSON files have SHA256
  provenance. The four-task/three-condition runner and exporter are ready. The
  exporter identifies summary requests via Claude compact-boundary/response IDs,
  and keeps their first tokens out of observation-to-resumed-action latency.
- A bounded native generation fixture is now exercising mid-decode guided refresh
  before freezing and launching the full pilot. Full agent tasks completed: zero.

## Frozen pilot launch (2026-09-12)

All three budget and frontend gates pass. Dense uses continuous batching and prefix cache; the plain-route cold-prefill calibration is excluded. Bounded mid-decode fixtures ended in short tool calls and did not cover their target, so they are explicitly incomplete rather than marked passed. Four-task / three-condition seed-73 pilot is frozen in `configs/pilot_shared_frozen_20260912.json`. Shared-GPU times are diagnostic only. Three proxy/generation/compaction alignment CPU tests pass.

The first full task is now executing (`superjson-error-stack-serialization`, InfLLM). Original Claude tool calls and incremental prefix reuse are observed. No complete task outcome yet. Formal 60 GiB InfLLM/KVMem TTFT measurement is queued behind all known GPU campaigns, with two repetitions in reversed method order. It uses the same 62 held-out prefixes and one generated token; it makes no agent/compaction/decode claim. CPU PSS/RSS is sampled externally every 10 seconds, starting during the first task (early startup coverage is incomplete); GPU NVML sampling covers each full arm.

## Formal decode measurement extension

The initial timing queue was intentionally stopped before any GPU work to add bounded decode measurement. Replacement queue: `results/formal_timing_60g_20260912_v2` (PID 1777169 at launch). It keeps the same 62 prompts and resource budgets, uses at most 64 greedy generated tokens, preserves natural EOS, and reports actual output counts plus engine phases. Two repetitions reverse the method order. Three CPU checks pass, including rejection of mismatched output counts. No formal GPU measurement has started. The complete agent campaign remains unchanged and is still generating its first task. An explicit bounded counting fixture is prepared for the unexercised middecode branch; it has not yet run.

## FlexiCache component feasibility progressed

All four original CUDA extensions compile unchanged on torch 2.11/CUDA 13/SM120. The padded original score kernel passes dimension-256/GQA-6 scoring and update-cadence checks; the original even-GQA dispatch has a verified arange(6) compilation failure. Original UVA pinned H2D/D2H and dirty table kernels pass bitwise/cross-stream checks. These tiny shared-GPU probes use about 2 MiB of PyTorch reserved memory and make no performance or full-model claim. Exact probe source versions and raw results are retained. Full sparse attention, top-k mapping, held-out head profiles and hybrid sessions remain unfinished. LouisKV was also checked through its official paper, a coauthor publication page and a direct GitHub repository search; no verified author repository was located in this audit. The original InfLLM complete task continues.

## 2026-09-12: existing agent-setting alignment, native Q8 primitives

User priority is InfLLM-only additions against completed KVMem/dense agent runs; see LONG_RUNNING_AGENT_REUSE_PLAN.md. Do not resume the BF16 three-condition campaign or FlexiCache work automatically. Current InfLLM arm remains integration-only.

Compiled the frozen original qw3 MMVQ/MMQ/medium-batch CUDA kernels behind a small ctypes ABI, without modifying native or InfLLM source. Losslessly repacks existing GGUF Q8_0 rows into the backend split-plane layout; does not expand all weights into BF16. Real 5120-column GGUF weight-row probes passed batches 1, 8, 11, 17, 64, 65, 512 and the 48-output projection fallback against CPU quantized-dot oracles. Maximum observed absolute error 1.20e-6. Embedding gather was bitwise equal and non-default stream execution passed. Peak Torch reservation 24 MiB; shared GPU component evidence only. First launch was rejected by a preflight that counted CUDA context space twice; corrected checks retain 2 GiB before context initialization plus a 256 MiB tensor cap and 768 MiB required after context initialization.

Evidence: reports/native_q8_compile_20260912.json; reports/native_q8_components_shared_20260912_v2.json. All three reused kernel/header hashes match the frozen native source manifest. This does not validate a complete quantized InfLLM model, FP8 KV, speculation, or agent performance. Next implement and verify GGUF-to-Qwen module mapping (including inverse recurrent head transforms), then complete FP8/state and budget alignment.
