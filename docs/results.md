# KVMem Performance Results

## Tiered-MTP context-growth benchmark (128K → 10M)

Date: 2026-07-02. Build: tiered-MTP (Design B — the MTP head's KV cache rides the
same GPU→CPU→NVMe block store as the 16 standard-attention layers instead of a dense
full-context cache).

### Configuration

| Parameter | Value |
|---|---|
| Model | Qwen3.6-27B-Q8_0.gguf (27.7 GiB weights) |
| GPU | RTX Pro 6000 (97 GiB) |
| Ladder | 128K, 256K, 512K, 1M, 2M, 4M, 8M, 10M |
| Decode tokens / turn | 256 |
| MTP | native speculate, chain=4 |
| KV dtype | fp16 |
| Prefill chunk | 2048 |
| kvmem budget (window) | 32768 tok |
| kvmem block tokens | 256 |
| kvmem method | retrieval |
| GPU memory ratio | 0.50 |
| CPU tier | 64 GiB |
| NVMe tier | 700 GiB (`/data/qw3_kvmem_eval_nvme`) |
| Determinism | `QW3_FATTN_NSPLIT=1`, `QW3_PREFILL_FA2_NSPLIT=1` |
| Wall time | 67.1 min (rc=0, 8/8 turns) |

### GPU residency: tiered MTP vs dense baseline

| Metric | Dense MTP (baseline) | Tiered MTP | Δ |
|---|---|---|---|
| peak_gpu_proc | 80302 MiB | **39382 MiB** | **−40920 MiB (−40.0 GiB, −51%)** |
| GPUmib @ turn 0 (128K) | 79851 (flat) | **38931 (flat)** | −40920 |
| GPUmib @ turn 7 (10M) | 79851 (flat) | **38931 (flat)** | −40920 |

GPU residency is constant across the entire 128K→10M ladder. The dense MTP cache
was an eager full-context allocation (~40 GiB) that never shrank; tiered MTP folds
the MTP layer into the bounded 2.14 GiB working set.

### Table 1 — End-to-end throughput & latency

| turn | ctx (tok) | new prefill tok | prefill (s) | prefill tok/s | prefill ms/tok | decode 256tok (s) | decode tok/s | decode ms/tok | accept |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 131,328 | 131,072 | 38.51 | 3403.7 | 0.294 | 4.15 | 61.65 | 16.22 | 0.509 |
| 1 | 262,400 | 130,816 | 40.20 | 3254.3 | 0.307 | 3.23 | 79.33 | 12.61 | 0.619 |
| 2 | 524,543 | 261,888 | 80.74 | 3243.5 | 0.308 | 3.64 | 70.40 | 14.21 | 0.623 |
| 3 | 1,048,832 | 524,033 | 162.52 | 3224.4 | 0.310 | 2.88 | 88.74 | 11.27 | 0.769 |
| 4 | 2,097,408 | 1,048,320 | 386.67 | 2711.1 | 0.369 | 3.99 | 64.21 | 15.57 | 0.487 |
| 5 | 4,194,560 | 2,096,896 | 798.24 | 2626.9 | 0.381 | 3.46 | 73.99 | 13.52 | 0.694 |
| 6 | 8,388,864 | 4,194,048 | 1595.80 | 2628.2 | 0.381 | 6.57 | 38.94 | 25.68 | 0.491 |
| 7 | 10,486,016 | 2,096,896 | 799.13 | 2624.0 | 0.381 | 3.59 | 71.31 | 14.02 | 0.669 |

Aggregate: 10.48M prefill tokens in 3901.8 s = **2687 tok/s mean prefill**. Prefill
throughput degrades only ~24% from 128K→10M despite the KV store growing 80×.

### Table 2 — KVMem tiering latency & overhead

| turn | selection (ms) | stage-in (ms) | stage-out@turn (ms) | assemble (ms) | in-prefill offload (s) | offload % of prefill | offload GiB/s | spill tier |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.075 | 0.013 | 164.6 (48 blk) | 1.83 | 3.09 (336 blk) | 8.0% | 1.80 | CPU |
| 1 | 0.072 | 0.013 | 136.7 (63 blk) | 1.34 | 3.86 (449 blk) | 9.6% | 1.93 | CPU |
| 2 | 0.114 | 0.010 | 49.0 (15 blk) | 1.38 | 8.43 (1009 blk) | 10.4% | 1.99 | CPU |
| 3 | 0.301 | 0.011 | 535.2 (31 blk) | 1.35 | 17.84 (2017 blk) | 11.0% | 1.88 | CPU→NVMe |
| 4 | 0.319 | 0.015 | 1188.2 (63 blk) | 1.37 | 97.53 (4033 blk) | 25.2% | 0.69 | NVMe |
| 5 | 0.655 | 0.014 | 282.2 (15 blk) | 1.37 | 220.19 (8177 blk) | 27.6% | 0.62 | NVMe |
| 6 | 1.394 | 0.016 | 753.1 (31 blk) | 1.36 | 439.32 (16353 blk) | 27.5% | 0.62 | NVMe |
| 7 | 1.831 | 0.015 | 324.2 (15 blk) | 1.36 | 220.66 (8177 blk) | 27.6% | 0.62 | NVMe |

Assemble sub-breakdown is stable per turn (~0.03 ms pages + ~1.23 ms re-RoPE +
~0.10 ms k-bar), flat regardless of context. Stage-in is effectively free (working
window already resident at turn start). Block size = 17 MiB (17 layers: 16 std + MTP).

### Table 3 — MTP speculative-decode efficiency (decode phase)

| turn | batches | drafted | accepted | acceptance | draft (s) | verify (s) | restore (s) | committed tok/batch |
|---|---|---|---|---|---|---|---|---|
| 0 | 85 | 336 | 171 | 0.509 | 0.498 | 2.265 | 0.034 | 3.01 |
| 1 | 74 | 294 | 182 | 0.619 | 0.429 | 1.939 | 0.014 | 3.46 |
| 2 | 73 | 292 | 182 | 0.623 | 0.431 | 1.943 | 0.019 | 3.51 |
| 3 | 63 | 251 | 193 | 0.769 | 0.366 | 1.655 | 0.008 | 4.06 |
| 4 | 68 | 271 | 188 | 0.487 | 0.400 | 1.826 | 0.015 | 3.76 |
| 5 | 87 | 344 | 169 | 0.694 | 0.510 | 2.365 | 0.036 | 2.94 |
| 6 | 87 | 344 | 169 | 0.491 | 0.510 | 2.365 | 0.036 | 2.94 |
| 7 | 70 | 278 | 186 | 0.669 | 0.411 | 1.906 | 0.020 | 3.66 |

Verify-attention (`verify_s`) is 55–65% of decode wall time — the MTP cost center is
the verify batch scanning the assembled 32K window, not drafting or restore.

### Table 4 — Tier residency growth (logical KV vs physical placement)

| turn | ctx (tok) | total blocks | KV logical (GiB) | GPU-KV (GiB) | CPU (GiB) | NVMe (GiB) |
|---|---|---|---|---|---|---|
| 0 | 131,328 | 513 | 8.52 | 2.14 | 6.38 | 0.00 |
| 1 | 262,400 | 1025 | 17.02 | 2.14 | 14.88 | 0.00 |
| 2 | 524,543 | 2049 | 34.02 | 2.14 | 31.88 | 0.00 |
| 3 | 1,048,832 | 4097 | 68.02 | 2.14 | 64.00 | 1.88 |
| 4 | 2,097,408 | 8193 | 136.02 | 2.14 | 64.00 | 69.88 |
| 5 | 4,194,560 | 16385 | 272.02 | 2.14 | 64.00 | 205.88 |
| 6 | 8,388,864 | 32769 | 544.02 | 2.14 | 64.00 | 477.88 |
| 7 | 10,486,016 | 40961 | 680.02 | 2.14 | 64.00 | 613.88 |

Total logical KV reaches 680 GiB at 10M; only 2.14 GiB stays GPU-resident. CPU fills
its 64 GiB cap by turn 3, then NVMe absorbs the rest (613.88 GiB < 700 GiB budget).

### Key efficiency findings

- **GPU residency is flat and bounded**: 38931 MiB every turn, 128K→10M. GPU-KV
  working set pinned at 2.14 GiB; process peak 39382 MiB (was 80302 dense).
- **The spill-cost cliff is the CPU→NVMe transition, not context length.** Turns 0–3
  evict D2H into the pinned CPU tier at ~1.9 GiB/s (offload ≈10% of prefill). Once CPU
  fills its 64 GiB cap (turn 3), all eviction is NVMe-write-bound at **~0.62 GiB/s**,
  and offload jumps to a steady **~27% of prefill**. That 0.62 GiB/s is the real
  long-context throughput limiter — the first lever for a bigger/faster NVMe tier or
  write batching.
- **Prefill compute holds up**: excluding offload, effective forward throughput is
  ~2.6–3.4K tok/s across the whole ladder; the visible tok/s drop from turn 4 is
  almost entirely the NVMe spill tax, not attention scaling.
- **Assemble + selection overhead is trivial** (<2 ms/turn combined through 8M) — the
  tiering machinery itself is not a bottleneck.
- **MTP stays healthy under tiering**: acceptance 0.49–0.77, committed 2.9–4.1
  tok/batch across all turns.

### Reproduction

```
QW3_KVMEM_MTP_TIER=1 python3 scripts/kvmem_session_profile.py \
  --model models/Qwen3.6-27B-Q8_0.gguf \
  --ladder 128K,256K,512K,1M,2M,4M,8M,10M \
  --decode-tokens 256 --chain 4 --window 32768 --block-tokens 256 \
  --gpu-ratio 0.5 --cpu-gb 64 --nvme-gb 700 --temp 0.6 --kv-dtype fp16 \
  --nvme-dir /data/qw3_kvmem_eval_nvme \
  --out-json /data/chaidi/kvmem_eval/results/kvmem_growth_128k_10m_mtp_tiered.json
```
