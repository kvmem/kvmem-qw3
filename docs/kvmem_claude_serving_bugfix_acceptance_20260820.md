# KVMem × Claude Code serving bugfix acceptance (2026-08-20)

## Scope

This change set fixes the logged serving and harness failures except for
multi-trajectory KVMem state. The controlled Claude Code runner remains
single-trajectory and disables the `Agent` tool.

## Implemented fixes

- Guided reselection no longer permanently falls back from MTP to plain
  decoding. One authoritative target token rebuilds the MTP prefix in the new
  compact window before speculative decoding resumes.
- M/P checkpoint admission now validates MTP prefix length and required MTP V
  page durability. Missing MTP V during spill produces a main-model-only
  record and invalidates later MTP checkpoints instead of terminating the
  request.
- A prefix checkpoint hole no longer promotes an old task query into an
  implicit `recovery-reselect`. The suffix is pressure-prefilled and indexed.
- Streaming request accounting records `stream_completed`, `terminal_status`,
  and `terminal_error`; a late SSE failure is not classified as success merely
  because HTTP 200 headers were already committed.
- Claude artifacts are group-writable for the host verifier. Dashboard state
  is reconciled against supervisor liveness and completed artifacts.
- An independent service watchdog reclaims the owned QW3 process group after
  runner death. The durable outer supervisor restarts unexpected inner-runner
  exits and reuses completed arms, but does not retry an explicit
  `paused_for_investigation` state.
- The Claude runner passes `--disallowedTools Agent`; analysis also rejects an
  observed `Agent` call as a protocol violation.
- Stable harness/KVMem prefix mode retains historical assistant reasoning
  framing across a new user query, avoiding a prefix rewrite at phase switch.

## Acceptance results

### Build and deterministic tests

- `qw3`, CUDA scorer, request-plan, Anthropic adapter, harness semantics, and
  KVMem store targets built successfully.
- Request-plan, prefix-reuse, refresh-policy, Anthropic adapter, harness
  semantics, KV block store, and KVMem store tests passed.
- Claude/DeepSWE Python harness suite: **29/29 passed**.
- CUDA softmax/adaptive scorer parity suite: **PASS**, including 8,193-block
  paths and FP16/FP8 prototype variants.

### Real-model GPU canaries

1. Threshold crossing, MTP=4: 4,088-token request crossed A=4,096 during
   decode; the next turn reused 4,032 tokens and prefetched only 105 new
   tokens. No cold rebuild occurred; warm and cold outputs were identical;
   MTP remained enabled with no fallback.
2. Threshold crossing, plain prefill-only: reused 4,032 tokens and prefetched
   112 of 4,144 prompt tokens. Warm and cold outputs were identical.
3. Guided MTP reselection: one mid-decode refresh logged
   `mtp_rebase=bridge`; the next request hit the new M checkpoint, M source
   index and MTP payload admission passed, and a third continuation hit again.
   There was no page-table error or permanent MTP fallback.
4. Oversized tool/request-plan test: both capacity-fitted tool results returned
   HTTP 200. A deliberately rewritten old prefix finalized as
   `pressure-prefill`, not stale-query recovery selection.
5. A single tool callback jumping beyond A+B generated a private retrieval
   query, pressure-refreshed successfully, and did not exhaust the GPU pool.

Temporary validation servers were stopped after the run; the pre-existing user
QW3 service was not modified or terminated.

## Deliberately not implemented

- Multiple concurrent/interleaved trajectory KVMem states.
- Native multimodal Qwen3.8 inference (an existing documented model capability
  limitation, not part of this serving-bug repair).
