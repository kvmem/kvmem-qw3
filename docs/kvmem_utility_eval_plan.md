# Plan: KVMem utility evaluation (external Python over qw3's OpenAI API)

## Context

`docs/motivation_experiment_summary_en.md` establishes baselines on a 102-sample LongMemEval-S
subset (17 per type × 6 types) with Qwen3.6-27B, `enable_thinking=true`, `temp=0`, DeepSeek-V4-Pro
answer-equivalence judge:

| setting | accuracy |
| --- | ---: |
| Full Context (upper bound) | 83/102 = 81.37% |
| Compact-only | 22/102 = 21.57% |
| RAG k=22 (~20K retrieved) | 75/102 = 73.53% |
| RAG k=44 (~41K retrieved) | 74/102 = 72.55% |

RAG recovers accuracy by **splicing raw history text back into the answer prompt**, so larger k =
more cache-miss/local-compute tokens (k=66 → 1.75× answer latency). **KVMem is the alternative:
retrieval happens directly on the KV cache (no text splicing).** The full history is prefilled once
into a tiered KV store; at answer time a fixed **32K-token window** is selected (mean-k / `content_mean`)
from already-resident KV. The question being benchmarked: *does KV-native retrieval at a 32K budget
match or beat RAG in its k=22–k=44 band — and approach Full Context — without paying RAG's text-splice
answer cost?*

**Hard requirement from the user:** the evaluation must be **external Python**, hitting qw3 **only**
through its OpenAI-compatible HTTP API. qw3 is the isolated service; no eval logic in qw3 C++.

## UPDATE 2026-06-29 — the "no code change" finding below was WRONG; selection was recency-only

The "Key finding" section below claimed step-mode reselect already used "the
question-conditioned decode query … the KV analog of RAG's embed-query→rank." **That was
false.** The batched-prefill path never captured a retrieval query, so at the
prefill→decode boundary `g_query_ready_=false`, `kvmem_retrieval_score()` early-returned,
and `pick_topk_blocks` fell back to **sink + recency** (highest block id). The eval was
silently measuring "last ~32K tokens," not relevance.

**Fix landed (`--kvmem-query-conditioned`, default-OFF, byte-identical when unset):** the
server marks the final user message's token span; the executor de-RoPEs the in-span query
rows during prefill into a multi-token content-frame buffer and, at the boundary,
scores every block by the mean over the M question tokens of the existing per-block dot
(rewards blocks broadly relevant to the whole question) instead of falling back to recency. Verified on a needle
far outside the recency window: flag ON retrieves the needle and answers correctly; flag
OFF hallucinates. **The harness now sends the question as the final user message** (system +
user(history) + user(question)) so the server can bracket it. **Re-run the eval with
`--kvmem-query-conditioned` added to the serve flags** (Part A); the partial run saved
before this fix was recency-only and is stale. See memory
[[kvmem-query-conditioned-selection]]. The original (now-corrected) analysis is retained
below for history.

## Key finding — no qw3 code change is needed

My earlier memory ("serve/CB hard-errors on kvmem") is **stale**. Current code:

- `process_request` (`src/qwen_native_backend.cpp:1347-1376`): kvmem now routes through the
  continuous-batching (CB) path via the **window-aware ragged verify** route. The *only* remaining
  hard error is the opt-in layered verifier (`QW3_CONTINUOUS_MTP_LAYERED_VERIFY`) — which we leave
  unset.
- `continuous_batch_request_supported` (`:3397`) only requires `dump==nullptr && max_tokens>=0` —
  kvmem is no longer a blocker.
- **Per-request isolation is built in:** `initialize_continuous_active` (`:5502`) gives every request
  its own `QwenExecutor` + `reset_state()` + `configure_executor_kvmem()` (`:5507-5520`), borrowing
  the *shared* pinned CPU-tier buffer (`set_host_tier_pool`, `:5514`) so each request does **not**
  re-alloc the 64 GB tier.
- **Step-mode answer-time reselect is exactly the requested behavior:** `kvmem_on_prefill_complete`
  (`:5457`, invoked on every prefill path incl. MTP at `:4626/:5658/:6033`) registers the whole
  prompt then calls `kvmem_reselect()` at the prefill→decode boundary; in `step` mode
  `kvmem_on_decode_step` (`:5491`) and `kvmem_mtp_advance_to` (`:5472`) **skip** decode-time reselect.
  The reselect query is the question-conditioned decode query (`kvmem_retrieval_score`,
  `qwen_executor.cpp:4033`), the KV analog of RAG's embed-query→rank.
- `qw3 serve` already parses **all** `--kvmem*` flags into the global engine config
  (`src/qw3_cli.cpp:349-407`) and honors `--enable-thinking` (`:511`).

⇒ The user's architecture is achievable **today, unchanged**: launch `qw3 serve --kvmem …`, point
external Python at the URL.

## Memory / sizing facts

- KV ≈ **64 KiB/token** (1 MiB per 16-token block, from the 2M session log). A 109K-token history ≈
  **~6.8 GiB KV** — fits entirely in the **64 GB CPU tier**; NVMe is effectively untouched per sample
  (NVMe only engaged in the 2M growth test). NVMe still configured (under `/data`) as required safety
  headroom.
- `--kvmem-gpu-memory-ratio 0.5` produced peak process GPU ~46.8 GB in the session test → meets the
  user's "≤48 GB" target.
- History tokens: mean 109,375, **max 113,270** (doc §2.2). Set `--ctx 163840` (160K) to hold
  max-history + question + thinking output with margin.

## Approach

### Part A — qw3 as an isolated OpenAI service (launch command, no code change)

```
QW3_FATTN_NSPLIT=1 QW3_PREFILL_FA2_NSPLIT=1 QW3_KVMEM_TIMING=1 \
build/qw3 serve \
  --model models/Qwen3.6-27B-Q8_0.gguf \
  --ctx 163840 \
  --kv-dtype fp16 \
  --kvmem --kvmem-block-tokens 16 \
  --kvmem-budget 32768 \
  --kvmem-method retrieval --kvmem-retrieval-method content_mean \
  --kvmem-update-mode step \
  --kvmem-query-conditioned \
  --kvmem-gpu-memory-ratio 0.5 \
  --kvmem-cpu-gb 64.0 \
  --kvmem-nvme-gb 256.0 --kvmem-nvme-dir /data/qw3_kvmem_eval_nvme \
  --native-mtp-speculate --mtp-chain 4 \
  --prefill-chunk 2048 \
  --enable-thinking \
  --temp 0.6 \
  --port 8080
```

Notes: `content_mean` = "mean k". `update-mode step` = reselect only at the prefill→decode boundary
(answer-time, question-conditioned). The Qwen3 thinking recipe (temp 0.6) is used because temp=0
greedy makes Qwen3-thinking loop; per-request sampling params come from the eval client body.
MTP is distribution-lossless under temp>0 as of 2026-06-29 (point-mass speculative-sampling accept
test), so `--native-mtp-speculate` is safe with temp 0.6 and affects throughput, not the sampled
distribution — output is equal-in-distribution to non-MTP, though not bit-identical (fp-atomic
split-K attention). NVMe dir under `/data` (root is full).

### Part B — external Python evaluation harness (`scripts/kvmem_eval/`)

All new, pure Python; the only qw3 touchpoint is the HTTP API.

1. **`dataset.py` — reproduce the 102-subset.** Download public LongMemEval-S; group by
   `question_type`; deterministically take **17 per type** (sort by `question_id` ascending, take
   first 17 — fully reproducible, recipe documented). Caveat surfaced below.
2. **`prompt.py` — render the QA prompt**, mirroring doc §3.1 Full Context layout:
   `System instruction` + `Question date` + `Full history sessions` (all `haystack_sessions` with
   `haystack_dates`, flattened in order) + `Question`. The question is **last** so the answer-time
   reselect query is question-conditioned. Sent as OpenAI `messages` (system + one user turn);
   `enable_thinking` honored server-side.
3. **`client.py` — OpenAI client.** `POST {base_url}/v1/chat/completions`, **concurrency = 1**
   (sequential, matches the doc and avoids tier/GPU contention), `temperature=0`,
   `max_tokens` generous (e.g. 2048 for answer body after thinking), long read timeout (full-history
   prefill is minutes). Capture wall-clock TTFT + total + raw completion.
4. **`judge.py` — DeepSeek-V4-Pro answer-equivalence.** Official LongMemEval per-type judge prompt,
   `enable_thinking=true`; **API key strictly from env `DEEPSEEK_API_KEY`** (never hardcoded /
   committed); base URL configurable. Returns correct/incorrect per sample.
5. **`run_eval.py` — orchestrator.** Iterate the 102 samples → render → call qw3 → judge → collect.
   Writes `results/kvmem_eval_<ts>.jsonl` (per-sample: id, type, model answer, gold, verdict, TTFT,
   latency) + a summary mirroring **doc §4.1**: overall `correct/accuracy` and the **6-type table**,
   placed alongside Full / Compact / RAG-k columns for direct comparison. Optionally parses the qw3
   server log for per-sample prefill/decode tok/s, kvmem step-1..4 reselect ms, and tier residency
   (the KV-native "no text-splice" cost story vs RAG's local-compute tokens).

## Critical files

- **New only:** `scripts/kvmem_eval/{dataset,prompt,client,judge,run_eval}.py` + `results/`.
- **No qw3 source changes.** (Reference, read-only: `src/qwen_native_backend.cpp:1347/3397/5457/5502`,
  `src/qw3_cli.cpp:349-407,511`, `src/qwen_executor.cpp:4033` confirm the API path already does what
  the eval needs.)
- Mirror conventions from `scripts/kvmem_session_profile.py` (model path, `/data` nvme dir, timeouts,
  JSON+human summary) and the existing OpenAI client `scripts/oai_client.py`.

## Verification (before the full 102 run)

1. **Service smoke:** start the serve command; one tiny chat request returns coherent text; server log
   shows `kvmem` enabled, `update_mode=step`, budget 32768.
2. **End-to-end on 3 samples** (one easy `single-session-assistant`, one `multi-session`, one
   `knowledge-update`): render → qw3 → DeepSeek judge all succeed; answers are on-topic (not "info not
   available" across the board).
3. **Per-request isolation + memory budget:** run the 3 samples **sequentially**; confirm via server
   log that GPU peak stays **≤48 GB**, CPU-tier usage returns to baseline between samples (no leak),
   and a 109K-token sample keeps NVMe at ~0. Confirm sample N+1's answer does not reference sample N's
   history.
4. **Full run:** 102 samples sequential; produce the overall + per-type accuracy tables next to the
   doc's Full/Compact/RAG columns. Report.

## Caveats (surface in the results writeup)

- **Quant differs:** doc baselines used `qwen3.6-27b-fp8` (vLLM); kvmem requires the local
  `Qwen3.6-27B-Q8_0` MTP gguf. Same model family; compare against the doc's Full=81.37% upper bound on
  that family, not as a bit-identical re-run.
- **Subset is reconstructed, not ID-identical** to the doc's 102 (the doc doesn't publish its sample
  IDs). Same construction recipe (17×6 balanced, deterministic). If the user can supply their exact
  sample-ID list or baseline harness prompts, swap them in for apples-to-apples parity.
- **Cost framing differs from RAG:** via the stateless chat API each request re-prefills the full
  history (cold), so report the one-time prefill cost and the (millisecond) answer-time
  reselect + decode cost **separately** — the latter is the warm-cache analog comparable to the doc's
  TTFT, and carries **no retrieval-text local-compute tokens** (KVMem's structural advantage over RAG).

## Constraints / non-goals

- **No commits without explicit user confirmation** (harness, results, or any prior uncommitted edits).
- DeepSeek key **only** via `DEEPSEEK_API_KEY` env; never written to disk or committed. NVMe dir under
  `/data`.
- `--kvmem` stays default-OFF and byte-identical to plain when unset; the service is only the existing
  `qw3 serve` with kvmem flags — no new serve mode, no executor changes.
- On approval, also copy this plan to `docs/kvmem_utility_eval_plan.md` as the durable doc (plan mode
  currently restricts edits to this plan file).
- Out of scope (deferred, user said "later"): 8-bit kvmem, async NVMe stage-out, prefill-throughput
  work.
