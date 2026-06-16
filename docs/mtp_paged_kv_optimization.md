# MTP + Paged KV Optimization Notes

Date: 2026-06-16

This note records the current MTP results under paged KV, the optimizations already tried, and the next implementation plan. Unless otherwise stated, these tests use:

- Model: `models/Qwen3.6-27B-Q8_0.gguf`
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- Context: `--ctx 262144`
- KV dtype: `--kv-dtype fp16`
- Paged KV: enabled with `--paged-kv --kv-pool-pages 49152`
- Continuous batching: disabled, benchmark variant `plain`
- Output length: 1024 tokens, with `--ignore-eos`
- Prefill chunk: `--prefill-chunk 2048`
- MTP timing: `QW3_MTP_PHASE_SYNC=1`

## Current CLI Changes

The MTP depth policy is now exposed as explicit CLI parameters:

```bash
--mtp-policy fixed|adaptive
--mtp-adaptive-min-chain N
--mtp-adaptive-max-chain N
```

Default behavior is unchanged:

- `--mtp-policy fixed`
- no MTP unless `--mtp-chain N` or `--native-mtp-speculate` is provided
- no continuous batching unless `--continuous-batching` is provided

For single-shot CLI benchmarking, MTP speculation still needs `--native-mtp-speculate`. In `serve` mode, `--mtp-chain N` maps to MTP speculation automatically.

## Baseline and MTP Results

### Paged KV, MTP Off

| Input | Prefill tok/s | Decode tok/s | Wall output tok/s | Decode time |
|---:|---:|---:|---:|---:|
| 32K | 3745.26 | 43.25 | 31.51 | 23.678s |
| 64K | 3434.21 | 40.64 | 23.07 | 25.196s |
| 128K | 2920.00 | 36.54 | 14.01 | 28.023s |

JSON outputs:

- `/tmp/qw3_pagedkv_nomtp_32k64k_1k.json`
- `/tmp/qw3_pagedkv_nomtp_128k_1k.json`

### Paged KV, Fixed MTP Chain 4

| Input | Prefill tok/s | Decode tok/s | Wall output tok/s | Decode time | vs MTP off decode |
|---:|---:|---:|---:|---:|---:|
| 32K | 3818.13 | 70.65 | 44.26 | 14.493s | +63.4% |
| 64K | 3513.72 | 44.80 | 24.61 | 22.858s | +10.2% |
| 128K | 3004.26 | 27.40 | 12.61 | 37.378s | -25.0% |

Important observation:

- 32K benefits strongly from MTP.
- 64K benefits only slightly.
- 128K is slower than plain paged KV, so long-context MTP is currently not robust.

JSON outputs:

- `/tmp/qw3_pagedkv_mtp4_32k64k_1k.json`
- `/tmp/qw3_pagedkv_mtp4_fixed_64k128k_1k.json`

## MTP Phase Breakdown

### 32K, Chain 4

```text
batches=236 drafted=943 accepted=788 rejected=73 rollbacks=73
acceptance=0.8356
batched_verify_tokens=1179
draft_s=1.388s
verify_s=12.553s
prefix_s=0.517s
restore_s=0.030s
replay_s=0.000s
```

Verifier dominates decode time, but the reduction in decode-loop count is large enough that MTP is a net win.

### 64K, Chain 4

```text
batches=246 drafted=982 accepted=778 rejected=84 rollbacks=84
acceptance=0.7923
batched_verify_tokens=1228
draft_s=1.532s
verify_s=20.370s
prefix_s=0.914s
restore_s=0.036s
replay_s=0.000s
```

Verifier cost grows with context and consumes most of the MTP benefit.

### 128K, Chain 4

```text
batches=239 drafted=956 accepted=784 rejected=70 rollbacks=70
acceptance=0.8201
batched_verify_tokens=1195
draft_s=1.660s
verify_s=33.981s
prefix_s=1.700s
restore_s=0.031s
replay_s=0.000s
```

At 128K, verifier time alone is greater than the full plain decode time:

- Plain decode: 28.023s
- MTP verify: 33.981s
- MTP full decode: 37.378s

This is the core long-context MTP problem.

## Improvements Already Tried

### 1. Exposed Adaptive MTP Policy

Added explicit CLI support for:

```bash
--mtp-policy adaptive
--mtp-adaptive-min-chain 2
--mtp-adaptive-max-chain 4
```

Benchmark command shape:

```bash
QW3_MTP_PHASE_SYNC=1 python3 scripts/continuous_batching_benchmark.py \
  --qw3 ./build/qw3 \
  --model models/Qwen3.6-27B-Q8_0.gguf \
  --ctx 262144 \
  --input-token-targets '65536 131072' \
  --concurrency-levels '1' \
  --max-tokens 1024 \
  --prefill-chunk 2048 \
  --variants 'plain' \
  --ignore-eos \
  --extra-arg=--paged-kv \
  --extra-arg=--kv-dtype --extra-arg=fp16 \
  --extra-arg=--native-mtp-speculate \
  --extra-arg=--mtp-chain --extra-arg=4 \
  --extra-arg=--mtp-policy --extra-arg=adaptive \
  --extra-arg=--mtp-adaptive-min-chain --extra-arg=2 \
  --extra-arg=--mtp-adaptive-max-chain --extra-arg=4 \
  --extra-arg=--mtp-paged-prefix \
  --extra-arg=--max-total-tokens --extra-arg=0 \
  --extra-arg=--kv-pool-pages --extra-arg=49152 \
  --extra-arg=--mtp-kv-pool-pages --extra-arg=49152
```

Results:

| Input | Policy | Decode tok/s | Decode time | Result |
|---:|---|---:|---:|---|
| 64K | fixed chain 4 | 44.80 | 22.855s | best |
| 64K | adaptive 2..4 | 42.33 | 24.190s | slower |
| 128K | fixed chain 4 | 27.40 | 37.378s | best MTP so far |
| 128K | adaptive 2..4 | 26.11 | 39.225s | slower |

Adaptive started at depth 3, then promoted to 4. This increased the number of verifier batches and made total verifier time worse. Lower initial depth is not a good default for these long-context outputs.

JSON output:

- `/tmp/qw3_pagedkv_mtp4_adaptive_64k128k_1k.json`

### 2. FlashInfer Batch Decode Partition Toggle

Tested 128K fixed chain 4 with:

```bash
QW3_EXPERIMENTAL_FLASHINFER_BATCH_DECODE_PARTITION=0
```

Result:

| Mode | Decode tok/s | Decode time | Notes |
|---|---:|---:|---|
| default | 27.40 | 37.378s | partition enabled |
| partition disabled | 21.27 | 48.150s | much slower |

Conclusion: long-context verifier needs KV partitioning.

JSON output:

- `/tmp/qw3_pagedkv_mtp4_128k_partition0_1k.json`

### 3. FlashInfer Rowwise Merge Toggle

Tested 128K fixed chain 4 with:

```bash
QW3_EXPERIMENTAL_FLASHINFER_BATCH_DECODE_ROWWISE_MERGE=0
```

Result:

| Mode | Decode tok/s | Decode time | Notes |
|---|---:|---:|---|
| default | 27.40 | 37.378s | rowwise merge enabled |
| rowwise merge disabled | 27.58 | 37.127s | roughly equal, tiny improvement |

Conclusion: rowwise merge is not the main bottleneck. It can be revisited, but disabling it is not enough to make 128K MTP profitable.

JSON output:

- `/tmp/qw3_pagedkv_mtp4_128k_rowmerge0_1k.json`

### 4. Fixed MTP Chain Depth Sweep at 128K

| Chain | Decode tok/s | Decode time | Acceptance | Verify time | Notes |
|---:|---:|---:|---:|---:|---|
| off | 36.54 | 28.023s | - | - | plain paged KV baseline |
| 2 | 18.67 | 54.838s | 0.9169 | 50.938s | too many verifier batches |
| 3 | 23.01 | 44.493s | 0.8437 | 40.932s | still too many verifier batches |
| 4 | 27.40 | 37.378s | 0.8201 | 33.981s | best MTP so far, still below baseline |

Conclusion: simply reducing chain depth does not solve 128K. Deeper chains reduce verifier batch count enough to beat chain 2/3, but chain 4 still cannot beat plain decode because each verifier batch is too expensive.

JSON outputs:

- `/tmp/qw3_pagedkv_mtp_chain2_128k_1k.json`
- `/tmp/qw3_pagedkv_mtp_chain3_128k_1k.json`
- `/tmp/qw3_pagedkv_mtp4_fixed_64k128k_1k.json`

## Current Diagnosis

The long-context MTP bottleneck is verifier attention.

For 128K, fixed chain 4:

- MTP reduces the number of outer decode rounds to 239 batches for 1024 output tokens.
- Acceptance is reasonable at 0.8201.
- Snapshot, restore, and replay are not the issue:
  - `snapshot_s=0.000s`
  - `restore_s=0.031s`
  - `replay_s=0.000s`
- Draft is also not the issue:
  - `draft_s=1.660s`
- The bottleneck is:
  - `verify_s=33.981s`

This means MTP will not be profitable at 128K until verifier attention becomes faster or verifies fewer/cheaper tokens.

## Next Modification Plan

### Step 1: Finish FlashInfer Verifier Parameter Sweep

Run 128K fixed chain 4 with different FlashInfer batch decode page sizes:

- `QW3_EXPERIMENTAL_FLASHINFER_BATCH_DECODE_PAGE_SIZE=128`
- `QW3_EXPERIMENTAL_FLASHINFER_BATCH_DECODE_PAGE_SIZE=256` current default
- `QW3_EXPERIMENTAL_FLASHINFER_BATCH_DECODE_PAGE_SIZE=512`
- `QW3_EXPERIMENTAL_FLASHINFER_BATCH_DECODE_PAGE_SIZE=1024`

Goal: find whether the current 256-page metadata/tile plan is suboptimal for 128K verifier.

Success criterion:

- 128K fixed chain 4 decode improves materially toward or above 36.54 tok/s.

### Step 2: Add Verifier Attention Timing

Current `verify_s` includes the full verifier forward batch, not just attention. Add finer timing for MTP verifier:

- qkv projection
- FlashInfer attention
- FFN/recurrent body
- logits/argmax
- metadata preparation/copy

Goal: identify whether the 128K verifier cost is dominated by attention kernel time, metadata overhead, or non-attention body work.

Success criterion:

- `native mtp_spec_summary` or a new `native mtp_verify_timing` log can explain at least 90% of `verify_s`.

### Step 3: Reuse Verifier Metadata Across MTP Batches

For a single request with fixed context growth, verifier page tables are highly regular. Today the batch decode path constructs and copies metadata every verifier batch.

Plan:

- cache page index/indptr layout for the MTP verifier when page table and batch shape are unchanged except for sequence length
- update only `last_page_len`, `seq_len`, and tile metadata as needed
- avoid per-batch heap allocation for `std::vector<int32_t>` metadata in the hot verifier loop

Success criterion:

- measurable reduction in `verify_s`, especially at 64K/128K.

### Step 4: Add Long-Context MTP Profitability Guard

Until verifier is optimized, MTP should avoid negative-return regions. Add a deterministic guard or adaptive fallback:

- If prompt is above a threshold and estimated verifier cost exceeds plain decode benefit, either:
  - disable MTP for that request, or
  - use a safer depth only when it improves measured throughput.

The guard must be explicit and observable in logs, for example:

```text
native mtp_policy_summary: fallback=plain reason="estimated verifier cost exceeds benefit"
```

Success criterion:

- 128K with `--mtp-chain 4 --mtp-policy adaptive` should not be slower than plain paged KV.

### Step 5: Optimize Continuous Batching + MTP Separately

Do not use continuous batching results to judge single-request MTP until the single-request verifier is profitable. After single-request 128K is fixed:

- batch verifier rows across requests
- avoid per-request recurrent snapshot allocation spikes
- merge MTP verifier timing with continuous batching timing logs
- retest C=1/2/4 under paged KV + continuous batching + MTP

Success criterion:

- C=2/C=4 MTP continuous batching improves total decode throughput over paged KV + continuous batching without MTP.

## Recommended Current Serving Defaults

For stable high-performance serving today:

```bash
./build/qw3 serve \
  --model models/Qwen3.6-27B-Q8_0.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx 262144 \
  -n 0 \
  --paged-kv \
  --continuous-batching \
  --max-active 4 \
  --body-batch \
  --kv-dtype fp16 \
  --prefill-chunk 2048 \
  --max-total-tokens 0 \
  --kv-pool-pages 49152
```

Do not enable MTP by default for long-context serving yet. For single-request experiments at 32K/64K:

```bash
./build/qw3 serve \
  --model models/Qwen3.6-27B-Q8_0.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx 262144 \
  -n 0 \
  --paged-kv \
  --mtp-chain 4 \
  --mtp-policy fixed \
  --mtp-paged-prefix \
  --kv-dtype fp16 \
  --prefill-chunk 2048 \
  --max-total-tokens 0 \
  --kv-pool-pages 49152 \
  --mtp-kv-pool-pages 49152
```

## Open Questions

- Does FlashInfer batch decode page size materially affect 128K verifier cost?
- How much of `verify_s` is attention versus non-attention body work?
- Can verifier metadata be cached enough to make 128K fixed chain 4 profitable?
- Should MTP policy be cost-based using measured verifier/acceptance instead of only acceptance-depth heuristics?
- Once single-request MTP is profitable at 128K, how should batched verifier be integrated with continuous batching?
