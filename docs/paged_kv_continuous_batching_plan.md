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
| 3 | Request-level paged KV state | Pending | Not run |
| 4 | Global KV page pool | Pending | Not run |
| 5 | BatchedDecodeExecutor batch=1 parity | Pending | Not run |
| 6 | Batched greedy decode with FlashInfer paged attention | Pending | Not run |
| 7 | Chunked prefill and decode interleaving | Pending | Not run |
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

- Pending.

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

- Pending.

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

- Pending.

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

- Pending.

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

- Pending.

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
