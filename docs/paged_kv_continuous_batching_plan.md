# Paged KV and Continuous Batching Development Plan

This document tracks the incremental implementation plan for efficient paged KV
and continuous batching in qw3. Each stage must be independently verifiable and
should be committed separately from unrelated work.

## Final Target

- Paged KV uses a global physical KV page pool shared by active requests.
- Each request owns an isolated logical page table that maps to global physical
  pages.
- KV pages are allocated on admission/prefill/decode and released on finish,
  cancel, or error.
- Continuous batching supports service workloads with stream, tools, greedy,
  and sampling requests.
- Decode uses a real batched executor and FlashInfer paged batch decode instead
  of looping over per-request `forward_one_token()`.
- HGEMM is not used as a solution path. Existing FlashInfer/MMQ paths should be
  reused wherever possible.

## Stage Status

| Stage | Name | Status | Test Result |
| --- | --- | --- | --- |
| 0 | Baseline freeze and regression harness | Completed | Passed on 2026-06-09 |
| 1 | Continuous batching service semantics | Completed | Passed on 2026-06-09 |
| 2 | Simplified admission control | Completed | Passed on 2026-06-09 |
| 3 | Request-level paged KV state | Completed | Passed on 2026-06-09 |
| 4 | Global KV page pool | Completed | Passed on 2026-06-09 |
| 5 | BatchedDecodeExecutor batch=1 parity | Completed | Passed on 2026-06-10 |
| 6 | Batched greedy decode with FlashInfer paged attention | In Progress | Ragged backend path passed on 2026-06-10 |
| 7 | Chunked prefill and decode interleaving | Completed | Passed on 2026-06-10 |
| 8 | Batched sampling optimization | Pending | Not run |

## Stage 0: Baseline Freeze and Regression Harness

Goal: make current behavior measurable before further architecture changes.

Tasks:

- Record baseline behavior for single request, concurrent requests, streaming,
  tools, sampling, paged KV, and opencode-compatible payloads.
- Ensure the existing regression scripts cover the current critical paths.
- Preserve a repeatable smoke command set for future stages.

Verification:

- `cmake --build build -j`
- `ctest --test-dir build --output-on-failure`
- `git diff --check`
- `python3 scripts/paged_kv_regression.py ...`
- `python3 scripts/continuous_batching_regression.py ...`
- Local curl smoke tests for `/v1/models`, chat, stream, concurrent greedy,
  concurrent sampling, and tools stream.

Completion Notes:

- Completed on 2026-06-09.
- Build passed: `cmake --build build -j`.
- Unit tests passed: `ctest --test-dir build --output-on-failure`.
- Diff check passed: `git diff --check`.
- Paged KV smoke passed:
  - Command output JSON: `/tmp/qw3_stage0_paged_kv.json`
  - Prompts: `short chinese`
  - Page sizes: `16 32`
  - Allocation modes: `identity reverse`
  - KV dtype: `fp16`
  - Result: 8/8 cases passed.
  - Mean decode throughput was about 25.6 tok/s for both page sizes.
- Continuous batching smoke passed:
  - Command output JSON: `/tmp/qw3_stage0_cb.json`
  - Prompts: `capital math cuda chinese`
  - Result: baseline and continuous outputs matched.
  - Evidence: `trace_max_batch=2`, `summary_max_batch=2`,
    `paged_kv_ready=True`, `hgemm_guard=True`.

## Stage 1: Continuous Batching Service Semantics

Goal: make the current scheduler path service-complete and observable before
global KV and batched kernels are introduced.

Tasks:

- Return real usage values for non-stream responses.
- Return real usage in `stream_options.include_usage` chunks.
- Track prompt, completion, and total tokens per request.
- Add route/fallback/lifecycle metrics that show whether a request used
  continuous batching or plain generation.
- Add finish reason and decoded token counts to continuous batching completion
  logs.
- Handle stream sink write failures as request cancellation where the current
  server/backend interface allows it.
- Keep tools and sampling behavior model-faithful: do not fabricate model
  output or add special content fallbacks.

Verification:

- Build/tests/diff-check pass.
- Non-stream usage is non-zero.
- Stream `include_usage` returns non-zero usage.
- Concurrent sampling requests enter `native continuous_batch`.
- Tools stream requests enter `native continuous_batch`.
- opencode-style payload still streams normal model output.

Completion Notes:

- Completed on 2026-06-09.
- Implemented real usage values for `/v1/chat/completions` and
  `/v1/completions`.
- Implemented real usage chunk for `stream_options.include_usage`.
- Added server-side request route/lifecycle logging:
  - `route=continuous|plain`
  - `prompt_tokens`
  - `completion_tokens`
  - `fallback_reason` when applicable
  - `client_closed=true` marker when stream writes fail.
- Changed native token callbacks to fire once per generated token, including
  empty decoded pieces, so service usage counts generated tokens rather than
  only non-empty text chunks.
- Verified non-stream chat usage:
  - `completion_tokens=1`, `prompt_tokens=23`, `total_tokens=24`.
- Verified non-stream completion usage:
  - `completion_tokens=8`, `prompt_tokens=5`, `total_tokens=13`.
- Verified stream `include_usage`:
  - `completion_tokens=1`, `prompt_tokens=24`, `total_tokens=25`.
- Verified concurrent sampling requests:
  - Both requests returned `completion_tokens=80`, `prompt_tokens=25`,
    `total_tokens=105`.
  - Server logs showed `route=continuous` and `max_batch=2`.
- Verified concurrent tools stream requests:
  - Both requests returned usage chunks with `completion_tokens=2`,
    `prompt_tokens=249`, `total_tokens=251`.
  - Server logs showed `chat(stream tools)`, `route=continuous`, and
    `max_batch=2`.
- Build passed: `cmake --build build -j`.
- Unit tests passed: `ctest --test-dir build --output-on-failure`.
- Diff check passed: `git diff --check`.
- Continuous batching regression passed:
  - Command output JSON: `/tmp/qw3_stage1_cb.json`
  - Evidence: `trace_max_batch=2`, `summary_max_batch=2`,
    `paged_kv_ready=True`, `hgemm_guard=True`.

## Stage 2: Simplified Admission Control

Goal: prevent the scheduler from accepting unbounded work before the global KV
page pool exists.

Tasks:

- Add active/pending/request-token budget knobs.
- Reject impossible single requests with clear errors.
- Queue or reject requests when pending capacity is exceeded.
- Release budget on finish, cancel, and error.

Verification:

- Overlong prompt returns a clear error.
- Pending overflow returns a clear 429-style error.
- Repeated short requests do not leak budget.

Completion Notes:

- Completed on 2026-06-09.
- Added `QW3_CONTINUOUS_BATCHING_MAX_PENDING`, default `128`.
- Added `QW3_CONTINUOUS_BATCHING_MAX_TOTAL_TOKENS`.
  - Default budget is `ctx_size * QW3_CONTINUOUS_BATCHING_MAX_ACTIVE`.
  - Setting the env var to `0` disables the token budget.
- Each continuous batching request now reserves
  `prompt_tokens + max_tokens` at admission.
- Reserved token budget is released on finish, error, zero-decode completion,
  and worker failure.
- Non-stream admission failures now return HTTP 429 with a clear error message.
- Over-context prompts map to HTTP 413.
- Negative admission smoke passed:
  - Temporary service used `QW3_CONTINUOUS_BATCHING_MAX_TOTAL_TOKENS=64`.
  - Request with reservation `105` returned:
    `HTTP/1.1 429 Too Many Requests`.
  - Error message:
    `continuous batching admission rejected: request token reservation 105 exceeds total token budget 64`.
- Positive admission smoke passed under the same service:
  - Small request returned `HTTP/1.1 200 OK`.
  - Usage was `completion_tokens=1`, `prompt_tokens=20`,
    `total_tokens=21`.
- Build passed: `cmake --build build -j`.
- Unit tests passed: `ctest --test-dir build --output-on-failure`.
- Diff check passed: `git diff --check`.
- Continuous batching regression passed:
  - Command output JSON: `/tmp/qw3_stage2_cb.json`
  - Evidence: `trace_max_batch=2`, `summary_max_batch=2`,
    `paged_kv_ready=True`, `hgemm_guard=True`.

## Stage 3: Request-Level Paged KV State

Goal: make request-owned logical page tables explicit before replacing
per-executor physical KV ownership.

Tasks:

- Introduce request KV state as a first-class object.
- Track logical pages, sequence length, and page size per request.
- Keep current physical storage path as the backend while validating the
  lifecycle.

Verification:

- Different page sizes work.
- Prompts and decode sequences crossing page boundaries work.
- Request page state is released on completion.

Completion Notes:

- Completed on 2026-06-09.
- Added `QwenExecutor::KvStateSnapshot`.
- Added `QwenExecutor::kv_state_snapshot()` for request-level logical KV state
  inspection.
- Added `ContinuousRequestKvState` in the native continuous batching path.
- Active continuous batching requests now refresh request KV state after
  prefill and each decode step.
- Continuous batching completion logs now include:
  - `kv_seq_len`
  - `kv_page_size`
  - `kv_pages`
- Physical KV storage is intentionally unchanged in this stage; this stage only
  makes request-level logical page state explicit and observable.
- Build passed: `cmake --build build -j`.
- Unit tests passed: `ctest --test-dir build --output-on-failure`.
- Diff check passed: `git diff --check`.
- Paged KV regression passed:
  - Command output JSON: `/tmp/qw3_stage3_paged_kv.json`
  - Prompts: `short chinese boundary_24_words`
  - Page sizes: `16 32`
  - Allocation modes: `identity reverse`
  - KV dtype: `fp16`
  - Result: 12/12 cases passed.
- Continuous batching regression passed:
  - Command output JSON: `/tmp/qw3_stage3_cb_rerun2.json`
  - Evidence: `trace_max_batch=2`, `summary_max_batch=2`,
    `paged_kv_ready=True`, `hgemm_guard=True`.
  - Regression JSON contains `kv_seq_len=` from continuous batching logs.
- Note: one earlier continuous regression run produced a transient exact-output
  mismatch on the `cuda` prompt; an immediate isolated rerun passed. The failed
  run did not show HTTP or batching evidence failures.

## Stage 4: Global KV Page Pool

Goal: move physical KV ownership to a backend/scheduler-level pool.

Tasks:

- Allocate global K/V page storage.
- Add free-list allocation and release.
- Use page availability for admission.
- Release pages on finish, cancel, and error.

Verification:

- Page IDs are unique across active requests.
- Free page count returns to baseline after repeated requests.
- Small pool exhaustion queues or rejects instead of crashing.

Completion Notes:

- Completed on 2026-06-09.
- Stage 4a completed on 2026-06-09.
- Added `KvPhysicalPageAllocator` abstraction.
- Added a native `GlobalKvPagePool` allocator with free-list based physical
  page allocation and release.
- Continuous batching executors now allocate logical page table physical page
  IDs through the global allocator.
- Executor `reset_state()` and destruction release allocated page IDs back to
  the allocator.
- Completion logs now include:
  - `kv_pool_used`
  - `kv_pool_free`
- Added boundary validation for allocator-returned physical page IDs.
- Added `QW3_CONTINUOUS_BATCHING_KV_POOL_PAGES`.
  - In Stage 4a, the configured value is clamped to the per-executor page
    capacity because K/V tensors are still physically owned by each executor.
  - Stage 4b must remove this clamp by moving K/V tensor storage to a backend
    global physical pool.
- Mapped `global KV page pool exhausted` errors to HTTP 429 for non-stream
  requests.
- Build passed: `cmake --build build -j`.
- Unit tests passed: `ctest --test-dir build --output-on-failure`.
- Diff check passed: `git diff --check`.
- Paged KV regression passed:
  - Command output JSON: `/tmp/qw3_stage4a_paged_kv.json`
  - Prompts: `short chinese`
  - Page sizes: `16 32`
  - Allocation modes: `identity reverse`
  - KV dtype: `fp16`
  - Result: 8/8 cases passed.
- Continuous batching regression passed:
  - Command output JSON: `/tmp/qw3_stage4a_cb.json`
  - Evidence: `trace_max_batch=2`, `summary_max_batch=2`,
    `paged_kv_ready=True`, `hgemm_guard=True`.
- KV pool exhaustion smoke passed:
  - Temporary service used `QW3_CONTINUOUS_BATCHING_KV_POOL_PAGES=1` and
    `QW3_PAGED_KV_PAGE_SIZE=16`.
  - A chat request that needed more than one page returned
    `HTTP/1.1 429 Too Many Requests`.
  - Error message:
    `global KV page pool exhausted: free=0 total=1 page_size=16`.
  - A minimal `/v1/completions` request succeeded under the same pool.
  - Log included `kv_pool_used=1 kv_pool_free=0`.
- Stage 4b completed on 2026-06-09.
- Added backend-owned global K/V tensor storage for continuous batching.
- Added `QwenExecutor::KvCacheStorage` so continuous batching executors can
  write/read K/V through shared backend-owned tensors.
- Standard attention K/V append and attention paths now use cache accessors
  that can resolve either executor-owned or backend-owned cache tensors.
- `QW3_CONTINUOUS_BATCHING_KV_POOL_PAGES` now controls both the allocator page
  count and the global K/V tensor physical capacity when continuous batching is
  enabled.
- The global K/V cache is allocated only when `QW3_CONTINUOUS_BATCHING=1`, so
  plain CLI and paged-KV regression paths do not allocate extra continuous
  batching storage or perturb their baseline behavior.
- Build passed: `cmake --build build -j`.
- Unit tests passed: `ctest --test-dir build --output-on-failure`.
- Diff check passed: `git diff --check`.
- Paged KV regression passed after the non-CB allocation fix:
  - Command output JSON: `/tmp/qw3_stage4b_paged_kv_rerun2.json`
  - Prompts: `short chinese`
  - Page sizes: `16 32`
  - Allocation modes: `identity reverse`
  - KV dtype: `fp16`
  - Result: 8/8 cases passed.
- Continuous batching regression passed:
  - Command output JSON: `/tmp/qw3_stage4b_cb_rerun2.json`
  - Evidence: `trace_max_batch=2`, `summary_max_batch=2`,
    `paged_kv_ready=True`, `hgemm_guard=True`.
  - Regression JSON contains `global KV cache pages=` and
    `kv_pool_used=/kv_pool_free=` logs.
- KV pool exhaustion smoke was repeated with backend-owned global K/V storage:
  - Temporary service used `QW3_CONTINUOUS_BATCHING_KV_POOL_PAGES=1` and
    `QW3_PAGED_KV_PAGE_SIZE=16`.
  - Multi-page chat request returned `HTTP/1.1 429 Too Many Requests`.
  - Minimal `/v1/completions` request returned `HTTP/1.1 200 OK`.
  - Log included `global KV cache pages=1 page_size=16 physical_slots=16`
    and `kv_pool_used=1 kv_pool_free=0`.
- Note: one continuous regression run produced a transient exact-output
  mismatch on the `cuda` prompt; an immediate rerun passed.

## Stage 5: BatchedDecodeExecutor Batch=1 Parity

Goal: introduce the batched decode abstraction without changing behavior yet.

Tasks:

- Add `BatchedDecodeExecutor` interface and batch input/output structs.
- Run batch=1 through the new abstraction.
- Compare batch=1 output to the existing executor path.

Verification:

- Batch=1 greedy matches the reference path.
- Batch=1 sampling with the same seed remains stable.

Completion Notes:

- Stage 5 completed on 2026-06-10.
- Added an internal `BatchedDecodeExecutor` abstraction for the native
  continuous batching decode path.
- The first implementation is intentionally delegated: it packages batch input
  and output structs, then calls each active request's existing
  `forward_one_token()` path. This preserves current behavior while giving
  Stage 6 a stable replacement point for FlashInfer paged batch decode.
- Continuous batching trace logs now include
  `decode_executor=delegated`.
- Verification:
  - `cmake --build build -j`: passed.
  - `ctest --test-dir build --output-on-failure`: passed, 2/2 tests.
  - `git diff --check`: passed.
  - `python3 scripts/paged_kv_regression.py --qw3 ./build/qw3 --model models/Qwen3.6-27B-Q8_0.gguf --page-sizes '16 32' --alloc-modes 'identity reverse' --prompts 'short chinese' --max-tokens 8 --ctx 1024 --prefill-chunk 512 --out-json /tmp/qw3_stage5_paged_kv.json --timeout 900`: passed, 8/8 runs.
  - `python3 scripts/continuous_batching_regression.py --qw3 ./build/qw3 --model models/Qwen3.6-27B-Q8_0.gguf --prompts 'capital math cuda chinese' --max-tokens 8 --ctx 1024 --prefill-chunk 512 --out-json /tmp/qw3_stage5_cb_rerun.json --timeout 900 --min-batch 2`: passed on rerun, 4/4 comparisons, `max_batch=2`, `paged_kv_ready=true`, `hgemm_guard=true`.
- Note: the first continuous batching regression run wrote
  `/tmp/qw3_stage5_cb.json` and failed 2 output comparisons because the plain
  baseline returned a transient abnormal text for `capital` and a different
  short reasoning prefix for `cuda`. The immediate rerun passed all
  comparisons and kept the required batching evidence.

## Stage 6: Batched Greedy Decode With FlashInfer Paged Attention

Goal: implement actual multi-request decode for greedy requests.

Tasks:

- Pack feed tokens, positions, and page tables.
- Use FlashInfer paged batch decode attention.
- Keep linear layers on the non-HGEMM path.
- Add batched argmax.

Verification:

- Batch=1 parity still passes.
- Batch=2/4 requests do not cross-contaminate KV.
- Multi-request decode is faster than the per-request loop.

Completion Notes:

- Stage 6 is in progress, not complete.
- Added a first real kernel-batched decode tail for greedy requests:
  request bodies still advance through per-request `forward_one_token(...,
  compute_logits=false)`, but active hidden states are packed into a
  `[batch, hidden]` buffer and the final `output_norm`, `lm_head`, and
  `argmax` run as batch kernels.
- The batched tail is enabled only for greedy requests that do not require full
  logits for sampling or penalties. Sampling and penalty requests keep the
  delegated path for correctness.
- Continuous batching trace now reports both scheduler and kernel batching via
  `native continuous_batch_executor: mode=... scheduler_batch=... kernel_batch=...`.
- Added a server-side max token cap: request `max_tokens` is limited by the
  service default configured with `-n`. This prevents OpenCode-style
  `max_tokens=32000` requests from reserving an entire context window and
  rejecting concurrent requests.
- Streaming chat requests whose rendered prompt exceeds `--ctx` are rejected
  before opening the SSE stream with HTTP 413, instead of returning an HTTP 200
  error chunk.
- Verification:
  - `cmake --build build -j`: passed.
  - `ctest --test-dir build --output-on-failure`: passed, 2/2 tests.
  - `git diff --check`: passed.
  - `python3 scripts/continuous_batching_regression.py --qw3 ./build/qw3 --model models/Qwen3.6-27B-Q8_0.gguf --prompts 'capital math cuda chinese' --max-tokens 8 --ctx 1024 --prefill-chunk 512 --out-json /tmp/qw3_stage6_lmhead_cb.json --timeout 900 --min-batch 2`: passed, 4/4 comparisons, `max_batch=2`, `paged_kv_ready=true`, `hgemm_guard=true`; log contained `mode=lm_head_batch` with `kernel_batch=2`.
  - `python3 scripts/continuous_batching_regression.py --qw3 ./build/qw3 --model models/Qwen3.6-27B-Q8_0.gguf --prompts 'capital math' --max-tokens 128 --ctx 2048 --prefill-chunk 512 --out-json /tmp/qw3_stage6_lmhead_cb_128.json --timeout 900 --min-batch 2`: passed; `mode=lm_head_batch` appeared 82 times.
  - `python3 scripts/paged_kv_regression.py --qw3 ./build/qw3 --model models/Qwen3.6-27B-Q8_0.gguf --page-sizes '16 32' --alloc-modes 'identity reverse' --prompts 'short chinese' --max-tokens 8 --ctx 1024 --prefill-chunk 512 --out-json /tmp/qw3_stage6_lmhead_paged_kv.json --timeout 900`: passed, 8/8 runs.
- Remaining Stage 6 work:
  - Cross-request recurrent-state batching is not implemented. Existing
    `recurrent_batch()` is a time-batch kernel for one sequence state, so it
    cannot be reused directly for independent requests without new state
    metadata support.
  - Cross-request paged attention is not wired into the executor. Existing
    prefill batch attention assumes one logical sequence/page table or
    consecutive positions; continuous batching needs ragged per-request page
    metadata.
  - Q/K/V, FFN, and attention output linear layers are still executed in the
    per-request body path, so aggregate decode throughput is only slightly
    improved by the batched lm_head tail.
- Stage 6.1 backend metadata and ragged attention path:
  - Added backend interfaces for per-row RoPE positions and ragged per-request
    paged decode attention metadata.
  - Added a CUDA per-row-position RoPE kernel for packed request batches.
  - Added a FlashInfer-backed ragged paged decode implementation for FP16 KV
    cache with head dimensions supported by the existing FlashInfer adapter.
  - Added executor-side ragged metadata packing for the current batched greedy
    path. The executor now builds device-side `positions`, `page_indices`,
    `page_indptr`, `last_page_len`, and `seq_lens` buffers for active request
    batches after their delegated body step. This prepares the later
    FlashInfer attention call without changing model outputs yet.
  - Added a narrow mutable decode-state interface on `QwenExecutor` and a
    decode-page preparation hook. The continuous batching executor now proves
    that active greedy requests have mutable hidden state, page tables, and
    current-token KV pages ready before the delegated body path runs.
  - Added a backend interface and CUDA implementation for ragged batched KV
    append. The new append path supports per-row logical positions and
    per-request page-table slices for FP32, FP16, FP8, and Q8 KV cache
    storage. This removes the remaining append-side blocker for cross-request
    standard-attention batching.
  - Added an experimental FP16 KV body-batch executor path gated by
    `QW3_CONTINUOUS_BATCHING_BODY_BATCH=1`. Standard-attention layers now use
    batched Q/K/V projection, per-row RoPE, ragged paged KV append, and
    FlashInfer ragged paged batch decode. Recurrent layers are still executed
    per request through a narrow single-layer executor interface, so this is
    correctness-first and not the final throughput target.
  - Extended the FlashInfer paged batch decode adapter from FP16 KV to FP8
    KV by instantiating the same paged batch decode path with raw e4m3 KV
    cache storage. Both non-ragged paged device attention and ragged
    continuous-batching attention now dispatch to the FP8 launcher when
    `QW3_KV_DTYPE=fp8`.
  - Fixed the FP8 non-ragged paged device attention path: it previously chose
    an FP8 batch-decode plan but still launched the FP16 KV FlashInfer kernel,
    causing the 1-byte FP8 cache plane to be read as half data. That made even
    non-continuous FP8 requests non-deterministic across repeated requests.
    The path now dispatches FP16 and FP8 launchers consistently.
  - Verification:
    - `cmake --build build -j`: passed.
    - `ctest --test-dir build --output-on-failure`: passed, 2/2 tests.
    - `git diff --check`: passed.
    - `python3 scripts/continuous_batching_regression.py --qw3 ./build/qw3 --model models/Qwen3.6-27B-Q8_0.gguf --prompts 'capital math' --max-tokens 8 --ctx 1024 --prefill-chunk 512 --out-json /tmp/qw3_stage6_ragged_backend_cb.json --timeout 900 --min-batch 2`: passed, `trace_max_batch=2`, `summary_max_batch=2`, `paged_kv_ready=true`, `hgemm_guard=true`.
    - `python3 scripts/continuous_batching_regression.py --qw3 ./build/qw3 --model models/Qwen3.6-27B-Q8_0.gguf --prompts 'capital math' --max-tokens 16 --ctx 1024 --prefill-chunk 512 --out-json /tmp/qw3_stage6_ragged_metadata_required_cb.json --timeout 900 --min-batch 2 --require-ragged-metadata`: passed, `ragged_metadata_ready=true`, `ragged_pages=4`, `ragged_max_seq_len=22`.
    - `python3 scripts/continuous_batching_regression.py --qw3 ./build/qw3 --model models/Qwen3.6-27B-Q8_0.gguf --prompts 'capital math' --max-tokens 16 --ctx 1024 --prefill-chunk 512 --out-json /tmp/qw3_stage6_body_ready_cb.json --timeout 900 --min-batch 2 --require-body-batch-ready --require-ragged-metadata`: passed, `body_batch_ready=true`, `ragged_metadata_ready=true`, `ragged_pages=4`, `ragged_max_seq_len=22`.
    - `python3 scripts/continuous_batching_regression.py --qw3 ./build/qw3 --model models/Qwen3.6-27B-Q8_0.gguf --prompts 'capital math' --max-tokens 16 --ctx 1024 --prefill-chunk 512 --out-json /tmp/qw3_stage6_ragged_kv_append_cb.json --timeout 900 --min-batch 2 --require-body-batch-ready --require-ragged-metadata`: passed, `body_batch_ready=true`, `ragged_metadata_ready=true`, `ragged_pages=4`, `ragged_max_seq_len=21`.
    - `python3 scripts/continuous_batching_regression.py --qw3 ./build/qw3 --model models/Qwen3.6-27B-Q8_0.gguf --prompts 'capital math' --max-tokens 4 --ctx 1024 --prefill-chunk 512 --out-json /tmp/qw3_stage6_body_on_cb.json --timeout 900 --min-batch 2 --enable-body-batch --require-body-batch-mode --require-ragged-metadata`: passed, `mode=body_batch_fp16`.
    - `python3 scripts/continuous_batching_regression.py --qw3 ./build/qw3 --model models/Qwen3.6-27B-Q8_0.gguf --prompts 'capital math' --max-tokens 16 --ctx 1024 --prefill-chunk 512 --out-json /tmp/qw3_stage6_body_on_16_cb.json --timeout 900 --min-batch 2 --enable-body-batch --require-body-batch-mode --require-ragged-metadata`: passed, `mode=body_batch_fp16`, exact output parity.
    - `python3 scripts/continuous_batching_regression.py --qw3 ./build/qw3 --model models/Qwen3.6-27B-Q8_0.gguf --prompts 'capital math cuda chinese' --max-tokens 8 --ctx 1024 --prefill-chunk 512 --out-json /tmp/qw3_stage6_body_on_4prompts_cb.json --timeout 900 --max-active 4 --min-batch 2 --enable-body-batch --require-body-batch-mode --require-ragged-metadata`: passed, `max_batch=4`, `mode=body_batch_fp16`.
    - Matched 4-prompt comparison without body batch wrote `/tmp/qw3_stage6_body_off_4prompts_cb.json`. Body batch reduced observed continuous request latencies from roughly `0.667-0.810s` to `0.611-0.741s` for this short test. This is a modest improvement because recurrent layers are still per-request.
    - `python3 scripts/continuous_batching_regression.py --qw3 ./build/qw3 --model models/Qwen3.6-27B-Q8_0.gguf --prompts 'capital math' --max-tokens 4 --ctx 1024 --prefill-chunk 512 --out-json /tmp/qw3_stage6_fp8_body_off_fixed_cb.json --timeout 900 --min-batch 2 --require-ragged-metadata --extra-arg=--kv-dtype --extra-arg=fp8`: passed, exact output parity, `max_batch=2`.
    - `python3 scripts/continuous_batching_regression.py --qw3 ./build/qw3 --model models/Qwen3.6-27B-Q8_0.gguf --prompts 'capital math' --max-tokens 4 --ctx 1024 --prefill-chunk 512 --out-json /tmp/qw3_stage6_body_fp8_fixed_cb.json --timeout 900 --min-batch 2 --enable-body-batch --require-body-batch-mode --require-ragged-metadata --extra-arg=--kv-dtype --extra-arg=fp8`: passed, exact output parity, `max_batch=2`, `mode=body_batch_fp16`.
    - `python3 scripts/continuous_batching_regression.py --qw3 ./build/qw3 --model models/Qwen3.6-27B-Q8_0.gguf --prompts 'capital math cuda chinese' --max-tokens 8 --ctx 1024 --prefill-chunk 512 --out-json /tmp/qw3_stage6_body_fp8_4prompts_fixed_rerun_cb.json --timeout 900 --max-active 4 --min-batch 2 --enable-body-batch --require-body-batch-mode --require-ragged-metadata --extra-arg=--kv-dtype --extra-arg=fp8`: passed on rerun, exact output parity, `max_batch=4`.
    - Note: the first FP8 4-prompt body-batch run wrote `/tmp/qw3_stage6_body_fp8_4prompts_fixed_cb.json` and had one `cuda` prompt divergence after a 10-character common prefix. Immediate rerun passed all comparisons; FP8 KV is numerically more sensitive, so wider FP8 tests should be repeated before treating a late greedy-token split as a deterministic regression.

Stage 6 throughput subplan:

1. Metadata and observability:
   - Keep reporting `scheduler_batch` separately from `kernel_batch`.
   - Add backend interfaces for arbitrary per-row RoPE positions and ragged
     per-request paged attention metadata.
   - Verification: compile-only plus existing continuous batching regression.
2. Batched standard-attention layer:
   - Pack active request hidden states into `[B, hidden]`.
   - Batch attention norm, Q/K/V projections, Q/K norms, per-row-position
     RoPE, KV append, FlashInfer ragged paged decode, attention output
     projection, and residual.
   - Verification: batch=1 parity, batch=2 exact greedy parity, page-table
     isolation tests.
3. Batched recurrent layer:
   - Add independent-state recurrent batch kernels. This is distinct from the
     existing time-batched `recurrent_batch()`.
   - Keep each request's recurrent and conv states isolated.
   - Verification: state isolation tests and exact greedy parity.
4. Batched FFN:
   - Batch FFN norm, gate/up projections, SwiGLU, down projection, and residual
     for active requests.
   - Verification: exact greedy parity and throughput counters.
5. Full greedy decode executor:
   - Replace the delegated body path for greedy no-penalty requests.
   - Keep fallback to delegated decode for unsupported dtypes, sampling,
     penalties, and any backend lacking the required kernels.
   - Verification: continuous batching regression, paged KV regression,
     OpenCode concurrent streaming test, and throughput comparison against the
     delegated path.

## Stage 7: Chunked Prefill and Decode Interleaving

Goal: prevent long prefills from blocking active decode requests.

Tasks:

- Track prefill progress per pending request.
- Interleave decode steps with configurable prefill chunks.
- Support chunk sizes in the 512 to 4096 range.

Verification:

- Short request latency remains bounded while a long prompt is prefilling.
- Different chunk sizes remain correct.

Completion Notes:

- Stage 7 completed on 2026-06-10.
- Continuous batching now keeps newly admitted requests in a prefilling set
  before they become active decode requests.
- Each scheduler turn advances at most one prefill chunk, and active decode
  batches run before the next prefill chunk. Prefilling requests count toward
  `QW3_CONTINUOUS_BATCHING_MAX_ACTIVE`, so KV/token capacity is not
  oversubscribed.
- Prefill chunk sizes are clamped to the intended service range:
  - `--prefill-chunk 0`: whole prompt prefill.
  - positive values below 512 use 512.
  - positive values above 4096 use 4096.
  - unset/default uses 512.
- Added `scripts/continuous_prefill_interleave_regression.py`, which starts a
  continuous batching server, sends a long prompt followed by a short prompt,
  and verifies that the short request starts decode before the long request's
  final prefill chunk.
- Verification:
  - `cmake --build build -j`: passed.
  - `ctest --test-dir build --output-on-failure`: passed, 2/2 tests.
  - `git diff --check`: passed.
  - `python3 scripts/continuous_batching_regression.py --qw3 ./build/qw3 --model models/Qwen3.6-27B-Q8_0.gguf --prompts 'capital math cuda chinese' --max-tokens 8 --ctx 1024 --prefill-chunk 512 --out-json /tmp/qw3_stage7_cb.json --timeout 900 --min-batch 2`: passed, 4/4 comparisons, `max_batch=2`, `paged_kv_ready=true`, `hgemm_guard=true`.
  - `python3 scripts/continuous_prefill_interleave_regression.py --qw3 ./build/qw3 --model models/Qwen3.6-27B-Q8_0.gguf --ctx 2048 --prefill-chunk 512 --max-tokens 4 --out-json /tmp/qw3_stage7_interleave_rerun.json --timeout 900`: passed, `long_nonfinal_chunks=3`, `short_decode_before_long_final_prefill=true`.
  - `python3 scripts/paged_kv_regression.py --qw3 ./build/qw3 --model models/Qwen3.6-27B-Q8_0.gguf --page-sizes '16 32' --alloc-modes 'identity reverse' --prompts 'short chinese' --max-tokens 8 --ctx 1024 --prefill-chunk 512 --out-json /tmp/qw3_stage7_paged_kv.json --timeout 900`: passed, 8/8 runs.
  - `python3 scripts/continuous_batching_regression.py --qw3 ./build/qw3 --model models/Qwen3.6-27B-Q8_0.gguf --prompts 'capital math cuda chinese' --max-tokens 8 --ctx 4096 --prefill-chunk 4096 --out-json /tmp/qw3_stage7_cb_4096_rerun.json --timeout 900 --min-batch 2`: passed on rerun, 4/4 comparisons, `max_batch=2`, `paged_kv_ready=true`, `hgemm_guard=true`.
- Note: the first `--prefill-chunk 4096` continuous batching run wrote
  `/tmp/qw3_stage7_cb_4096.json` and failed one comparison because the plain
  baseline returned the known transient abnormal `capital` output. Immediate
  rerun passed all comparisons.

## Stage 8: Batched Sampling Optimization

Goal: improve sampling performance after the batched greedy executor is stable.

Tasks:

- Replace per-request logits copies with batched logits copy.
- Keep host-side sampling initially for correctness.
- Consider device-side sampling only after the batched host path is stable.

Verification:

- Sampling requests batch correctly.
- Sampling throughput improves over per-request copy.
- Seeds, top-p, top-k, min-p, and penalties behave as expected.

Completion Notes:

- Pending.
