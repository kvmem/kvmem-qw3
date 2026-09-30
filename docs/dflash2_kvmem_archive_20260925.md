# Q8/FP8 DFlash 2 KVMem archive stage — 2026-09-25

The persistent goal remains **active and incomplete**. This stage keeps Q8_0
target weights and E4M3 FP8 target KV. The optional DFlash cache uses the draft's
existing FP32 representation; it does not add BF16 target or KV support.

## Implementation

Ladder state v2 adds optional draft metadata plus per-layer raw K and V. The
metadata records uint64 logical position, bounded row count, independent RoPE
position, tensor shape and a SHA-256 fingerprint of the draft configuration,
safetensors index (when present), and loaded weight shards. File digest caching
uses the target archive's existing stat-validated SHA-256 cache.

The target archive manifest/layout remains unchanged: the same target KV archive
can be read with or without DFlash. Target-only ladders still use state v1. Old
readers reject v2 explicitly. New readers validate table/file lengths, duplicate
and unknown entries, draft completeness, shape and positions before GPU restore.
The executor also checks that required target tensors are present and correctly
sized before replacing target state.

A matching draft restores raw K/V and rebuilds rotated K from its saved RoPE
frame. A legacy ladder or different draft fingerprint clears the draft cache at
the restored target position and logs the cold start. Attach no longer resets a
successfully restored draft. The official draft's full 2048-row cache is exactly
83,886,080 payload bytes (80 MiB), independent of total archive history; JSON
metadata and entry-table overhead are additional. Each ladder has its own cache
copy, so total archive disk usage still grows with ladder count.

Archive commands preserve explicit `--native-mtp-chain 0`. Their build, residual
replay, query and frozen-server request paths use the selected speculation mode.
Archive defaults no longer set a process-global MTP speculation environment
variable, which could contaminate a subsequent target-only control in the same
process. The historical implicit MTP default is retained when no mode is chosen.

When the requested prefix precedes the first ladder, residual replay starts a
fresh session at token zero. Previously it attempted to preserve an empty source
index and failed with `cannot preserve the existing mean-K content index`.

## Validation

- CPU suite: 33 tests pass, including a new archive test for v1/v2 round trips,
  exact arbitrary payload bits, 2^40 logical positions, empty caches, and corrupt
  versions/tables/positions/tensor layouts. The model test verifies fingerprints
  change with configuration and same-size weight edits, and ignore unrelated
  files.
- Official-topology FP32 and BF16-weight draft GPU fixtures each pass seven
  numerical cases. Each also exercises 16 exact archive continuation restores
  while crossing three RoPE rebases beyond a 2^40 logical origin. Different-draft
  and legacy restores clear stale cache. This validates the draft only.
- Real-model matrix: 12 independent CLI processes pass 54 assertions. Four
  builds cover target-only/DFlash 4K archives and resumed/uninterrupted 8K
  archives. Eight attach processes produce 16 deterministic responses; the
  4K direct and 3K residual-replay outputs match the no-speculation control.
  Target raw K, V, tokens and target ladder state match across build modes.
  Resumed 8K archives match uninterrupted raw K, V, token and complete new
  ladder files byte for byte, including the draft cache. Evidence:
  `benchmark/dflash2/q8_archive_lifecycle03/validation.json`.
- After the final payload-size overflow guard, all 33 CPU tests and four
  selected CUDA-build CTests pass. Both draft fixtures were rerun, and the
  BF16-weight fixture's memcheck reports zero errors.
- Six edge-matrix processes pass 12 response checks: 1K replay before the first
  ladder, 4K guided-query fallback at 32 thinking / 16 compact-query tokens, and 4K attention
  probing. Each checks repeated-query and target/DFlash equality of answer,
  decoded count, private-query state and retrieval indices, not only text.
  Evidence: `benchmark/dflash2/q8_archive_edges_lifecycle02/validation.json`.
  The initial `edges_lifecycle01` zero-index failure is retained.
- Natural completion of the compact query with a 128-token limit (32 private
  thinking tokens) passes in two additional processes and four responses. Both
  routes produce the same 27-token compact query, retrieval policy and final
  answer. Evidence: `benchmark/dflash2/q8_archive_edges_lifecycle04/validation.json`.
  `edges_lifecycle03` mistakenly increased the thinking budget while keeping the
  compact-query limit at 16; its required-completion assertion failed and the
  run is retained, not counted as a completed-query pass.

In total, the successful production matrices cover 20 independent CLI processes
(four builds and 16 attach processes) and 32 responses. These are correctness
and lifecycle tests; the short, repeated engineering-log prompt is not a quality
benchmark. Timings in JSONL remain diagnostics and exclude model loading/attach.

Reproduce with:

```sh
cmake --build build-dflash-host -j 12
ctest --test-dir build-dflash-host --output-on-failure
cmake --build build-dflash-flashinfer -j 12 --target qw3 qw3-dflash-forward qw3-dflash-archive
build-dflash-flashinfer/qw3-dflash-forward benchmark/dflash2/official_fp32_fixture
build-dflash-flashinfer/qw3-dflash-forward benchmark/dflash2/official_bf16_fixture
python3 benchmark/dflash2/run_q8_archive_validation.py --run-id UNIQUE
python3 benchmark/dflash2/run_q8_archive_edge_validation.py --base benchmark/dflash2/q8_archive_UNIQUE --run-id UNIQUE
```

The process runner refuses existing directories, records commands and source,
binary, input and log hashes, and runs GPU jobs sequentially. It builds from a
fixed engineering-log corpus, compares target-only and DFlash archive bytes,
tests repeated full-prefix and residual-replay queries, legacy cold attach,
and resumes a cloned unsealed checkpoint before comparing with an uninterrupted
build. Checkpoint resumption is a controlled restart simulation, not a power-loss
or filesystem crash-consistency test.

## Limits

These tests use a retained-history archive with fixed 12K context capacity,
1024-token working set, 512-token generation reserve, 32-token blocks and 4 GiB
CPU tier. They do not qualify arbitrary-size bounded-repository archive APIs,
archive HTTP service, stochastic quality, production summary quality or a long
soak. GPU target sampling, actual DFlash continuous batching, formal independent
performance/quality comparisons and final integration remain open. No new public
performance result is claimed by this archive stage.
