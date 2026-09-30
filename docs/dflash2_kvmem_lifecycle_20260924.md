# Q8_0 / FP8 KVMem lifecycle qualification

The persistent DFlash 2 goal remains active. This stage exercises the Q8_0
target with E4M3 FP8 KV; it adds no BF16 target-weight or BF16-KV support.
The official BF16 draft and existing target residual storage are unchanged.

## Corrections

The initial 4K production comparison diverged at zero-based callback 12.
A target-only diagnostic, using the same prefix and teacher tokens, localized
the first difference to standard-attention layer 3: verifier widths 1–5 were
exact, while widths 6–8 differed. Its working window had five KV tiles.

FlashInfer `MergeStates` chooses a different reduction kernel when the row
count exceeds the number of tiles. The previous multirow optimization
therefore changed serial-decode arithmetic in this regime. The adapter now
launches the serial decoder's large-index merge kernel across all uniform
rows. This retains one merge launch and leaves the single-token baseline
unchanged. Nonuniform tile counts still use the existing per-row merge.

Snapshot cloning now copies the complete KVMem metadata and immutable draft
snapshot. CPU prefix offload packs draft raw K/V together with target state,
accounts for those bytes, and allocates independent draft buffers on restore.
This supplies the storage primitive; it does not make DFlash continuous
batching available.

Target-only requests for private retrieval and live cache sessions now use the
same transaction controller with speculation/tracing disabled. Their prefix
admission checks no longer require nonexistent MTP payloads. Ordinary plain
generation retains its original route.

The first guided run had equal final text but different retrieval decisions:
plain refreshed at token 13; DFlash skipped that boundary and refreshed at 15.
Speculative cycles are now capped at the refresh threshold, then bridge one
token at a time until a safe text boundary. The runner compares generated
queries, refresh positions/modes and recent-window sizes as well as output.

## Completed evidence

- All 32 host CTests pass after rebuilding the current host sources.
- Prefix storage passes byte-exact CPU round trips, recycled-buffer overwrite,
  reader lifetime and draft metadata checks at logical origins 0 and 2^40.
- Attention CTests pass the independent CPU oracle and 168 serial/batch
  comparisons: FP8/FP16, head dimensions 128/256, widths 2/4/5/6/7/8, seven
  prefix lengths around KV-tile boundaries.
- The real Q8 target passes all widths 1–8 at four positions after 4K history
  and KVMem pressure selection: all 64 layer residuals and vocabulary logits
  compare exactly.
- Both the 4K and 32K production suites pass eight requests on each route, with strict
  greedy output comparison: ingest, root/warm query, history growth, tail
  rewrite, cancellation after five callbacks, recovery and old-prefix fork.
- The 32K suite uses a fixed 8K repository, retains at most 8K history tokens,
  and exercises reclamation and divergence before the retained tail.
- Nine guided-suite requests per route pass output and policy equality.
  Direct/private queries contain 21/27 tokens; both routes refresh at token 13
  for the capped 16-token fallback and naturally completed 57-token query.
- Retained-history semantic-chunk replay passes both repeated requests on
  both routes. This is not bounded-repository semantic support.
- Named-cache save, frozen fork/repeat, append, version-2 query and eviction
  pass on both routes. This also exercises the target-only session controller
  and complete snapshot cloning; it is not bounded named-cache support.
- Twenty real-model transaction executions pass: clean reference versus
  discarded-private/published-reasoning rollback at boundaries 63/511/1535/4095,
  plus four query-retention modes. Target state, main KV, draft raw K/V,
  logits and query bytes are exact. This is a state oracle, not a measurement
  of generated summaries' quality.
- Full 168-case attention memcheck reports zero errors.
- The bounded four-case racecheck reports zero hazards. Nine selected
  regression CTests and 44 ordinary-QW3 production checks pass.

The latest matched-output 256-token warm KVMem diagnostic has median
end-to-end rates **44.235 plain / 87.668 DFlash tok/s (1.982×)** and decode
rates **44.92 / 90.64 tok/s**. End-to-end ranges are 44.202–44.547 and
87.071–88.462; median TTFT is 76.56 / 80.58 ms. These are three warm repeats
within one process per route, plain first, not the goal's independent,
alternating-order none/MTP/DFlash or public comparisons. Raw samples and phases
are in `benchmark/dflash2/q8_lifecycle_timing_summary_01.json`.

Evidence is retained in `benchmark/dflash2/q8_lifecycle_validation_merge02.json`,
`q8_lifecycle_validation_merge03.json`, `q8_lifecycle_validation_merge04.json`,
`q8_lifecycle_validation_compact01.json`, `q8_lifecycle_validation_control02.json`
and `q8_lifecycle_validation_remaining01.json`, plus their per-run logs and
binary hashes. Named-cache evidence is `q8_lifecycle_validation_cache03.json`.
Failed attempts are retained:

- Initial 4K production and layer diagnostics expose the merge discrepancy.
  The repeat with `QW3_MTP_VERIFY=sequential` is not a separate DFlash verifier:
  that override applies only to built-in MTP, so DFlash still used batching.
- The layer harness initially attempted to commit a discarded repository
  append. It now replays and publishes actual tokens before the next block.
- The initial 32K harness incorrectly sized physical context from logical
  history; bounded KVMem correctly rejected it. The physical repository is
  now fixed at 256 blocks × 32 tokens, independent of logical history.
- A first compile failed on mixed integer types in `std::max`; the corrected
  build is `q8_lifecycle_merge_build_02.log`.
- The original plain route rejected private retrieval/cache sessions. Later
  equal-text guided runs still had unequal policy; only the policy-gated run
  after refresh/admission fixes qualifies.
- Cache erase leaves an `evicted` tombstone by API contract. The initial probe
  incorrectly required the ID to disappear; the corrected assertion checks
  that it is no longer ready.
- Full attention race instrumentation was stopped after 107 seconds. It is
  retained as incomplete, not a passing run; a bounded four-case variant
  covers FP8/FP16 with eight rows and two/five KV tiles.

## Remaining stage gates

All recorded stage cases now pass. Named-cache and semantic suites explicitly use the
retained-history store because the bounded repository does not yet admit
those modes. These suites cannot establish bounded support for those APIs.

Archive reload, advanced sessions, resource-exhaustion/long-soak checks, GPU
sampling, continuous batching, formal quality and repeated performance/public
comparisons remain open. Stage timings are diagnostics, not formal benchmark
claims. See [the goal contract](dflash2_q8_fp8_goal_20260924.md).

## Reproduction

```bash
cmake --build build-dflash-flashinfer --target qw3-dflash-kvmem-lifecycle-probe qw3-target-batch-parity-probe qw3-thinking-compaction-probe qw3-flashinfer-decode-gqa -j 8
python3 benchmark/dflash2/run_q8_lifecycle_validation.py --run-id UNIQUE_RUN_ID
```

The runner refuses to overwrite manifests/logs, records source/binary/log
hashes and executes GPU cases sequentially. It stops on the first failure.
