# OSWorld 256K prefix cache and timeout repair

The interrupted run is `official256k-prefix-no-proxy100-20260904T145233Z`.
Its logs are preserved with `repair_stop_snapshot.json`. Tasks 001 and 002
have evaluator results; 003 failed in the transport and requires rerunning;
004 was stopped by the user at step 197 and is incomplete. Neither incomplete
task should be converted to a model score of zero.

## Diagnosis

The existing ordinary prefix cache already shares main-model and MTP KV pages.
`initialize_continuous_active` performs lookup/restoration before prefill
allocates suffix pages. Cached entries do not contribute to the queued/active
request reservation counter. The physical page count is already independently
configurable through `QW3_CONTINUOUS_BATCHING_KV_POOL_PAGES` and
`QW3_CONTINUOUS_BATCHING_MTP_KV_POOL_PAGES`; `--ctx` remains the per-sequence
limit. Increasing `--ctx` is unnecessary and changes the experiment.

The old log's `kv_seq_len=0` on a cold MTP request was stale telemetry: that
lane only refreshed the state snapshot on a prefix hit. It did not mean the
executor finished without KV. Completion logging now reports the final KV
length and separate reused/prefilled token counts.

The uninterrupted miss sequence starts when the 100-turn history begins
sliding (step 102), not when the prompt first reaches 128K. The agent computes
`start_step = max(1, total_steps - history_n)` and rewrites the first historical
user message. Every following step changes early context. Screenshot folding
also replaces earlier images every ten steps. Comparing the preserved message
JSON shows approximately 7.3K common characters at steps 102–103, and 10.2K at
184–185, out of 182K–311K characters. Character counts are diagnostic evidence,
not token counts. The long recurrent-state checkpoint is no longer valid.
An exact prefix cache cannot reuse a later suffix after this early divergence.
These semantic changes explain cold prefill even when enough pages exist;
LRU eviction during cold allocation is a consequence, not proof of failed
prefix admission. The new miss log reports `longest_cached_lcp` in tokens.

## Changes

- Lookup hashes prompt token bytes once, reducing page-boundary probing from
  quadratic to linear work. It still verifies exact tokens, image identities,
  M-RoPE positions and complete MTP state.
- Page budgets count unique physical pages instead of summing references from
  overlapping checkpoints. Pool cleanup rejects duplicate free-list returns.
- Active requests hold cache-entry leases through executor destruction,
  including exception paths. Final completion releases pages and reservation
  before notifying the HTTP caller. A failed dynamically admitted MTP request
  is completed individually instead of being lost outside the initial batch.
- The default request reservation budget follows physical pool capacity;
  logical context remains independently bounded. The existing explicit
  `QW3_CONTINUOUS_BATCHING_MAX_TOTAL_TOKENS` override remains available.
- A request cancellation predicate reaches the continuous-batching worker.
  Pending requests are removed without waiting for another long decode;
  active requests check cancellation between prefill chunks and decode output.
  HTTP handlers detect a closed client socket for nonstream requests too.
- Clients can label requests with `x-qw3-request-id` and call
  `POST /v1/requests/{id}/cancel`. HTTP 202 means cancellation is pending;
  HTTP 200 with `completed=true` means the entire request lifetime has ended;
  unknown IDs return 404 and are not an acknowledgement.
- The OSWorld adapter disables hidden SDK retries, labels attempts, waits for
  cancellation acknowledgement before retrying, closes the SDK client on all
  exits, and does not retry bad requests. Capacity rejections use backoff.
  The run script sets the nonstream timeout to 600 seconds by default;
  `OSWORLD_OPENAI_TIMEOUT` remains configurable.

Cancellation acknowledgement is enabled by `QW3_OSWORLD_REQUEST_CANCEL=1` in
the QW3 evaluation script. Other OpenAI-compatible servers retain their normal
protocol unless that extension is explicitly enabled. The chunk-level prefill
cancellation guarantee here applies to the ordinary continuous-batching route
used by this experiment; persistent KVMem sessions have separate lifecycle
semantics.

## Validation

`qw3-prefix-cache-lifetime` exercises shared physical pages above half of a
256K pool, duplicate cleanup, eviction that initially frees zero pages, and
cancellation acknowledgements held until the final request lease is released.
`test_request_retry.py` verifies no hidden SDK retries and no overlapping retry
without an acknowledgement. Existing trace instrumentation tests are included.

`scripts/prefix_cache_long_multimodal_canary.py` owns an isolated server on
port 18081. It uses actual consecutive OSWorld screenshots, starts above 100K,
appends through 160K, keeps both pools at 262,144 slots and MTP=4, and forces
checkpoint eviction with an entry cap of four. It checks every appended turn
for reuse, small prefill and TTFT, and compares selected outputs against cold
servers. Cancellation checks cover cold prefill and decode, including capacity
rejection while an earlier request is active. Reports live in
`benchmark/osworld_v2/prefix_repair_20260905/`.

The strict-append canary and the unchanged `history_n=100` benchmark answer
different questions. Full prefill after history truncation remains expected;
eliminating that cost would require changing the input-history policy or a
separately validated approximate/state-reconstruction method.

Observed strict-append result: 31 requests, 102,086 → 163,937 prompt tokens,
30/30 warm hits, 27 checkpoint evictions, 2,078–2,093 prefilled tokens per
warm turn, TTFT 1.079–1.524 seconds (mean 1.304 seconds). At the last turn
the pool uses 10,247 of 16,384 pages. All three sampled cold outputs match.
Cancellation drained in 0.407 seconds during prefill and 0.103 seconds during
decode after client timeout. A separate socket test verifies automatic
disconnect detection without sending a cancellation command.

The separate 48-token ordinary MTP canary also passes: warm/cold output is
byte-identical and speculative drafting remains active after restoration.
Task 003 rerun: `official256k-prefix-repair003-20260904T234454Z` (500-step cap,
original sampling/history settings; independent results directory).
