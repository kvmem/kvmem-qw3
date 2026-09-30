# DFlash 2 / KVMem long-history cold concurrency baseline

Two seeds, 12 fixed real DeepSWE prefixes per configuration; Q8_0 + FP8 KV, K128K/G64K.

| Seed | C | Output tokens | Group wall s | Cold tok/s | Callback-window tok/s | Post-all-first tail tok/s | TTFT p50 s | Max gap s | GPU peak GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 1 | 10798 | 1020.06 | 10.59 | 80.41 | 80.41 | 68.07 | 0.08 | 53.60 |
| 1000 | 2 | 12931 | 1039.90 | 12.43 | 21.64 | 85.71 | 99.59 | 87.37 | 65.53 |
| 1000 | 4 | 14567 | 1056.82 | 13.78 | 17.38 | 75.25 | 174.77 | 221.82 | 89.33 |
| 1001 | 1 | 11490 | 1019.57 | 11.27 | 85.87 | 85.87 | 68.05 | 0.11 | 53.60 |
| 1001 | 2 | 13197 | 1039.12 | 12.70 | 22.12 | 85.58 | 99.62 | 85.58 | 65.53 |
| 1001 | 4 | 13490 | 1045.07 | 12.91 | 16.38 | 77.96 | 184.71 | 186.06 | 89.33 |

Native cold fixed groups; no prefix reuse, HTTP routing, tool execution or SWE success evaluation.

Callback-window and native per-request decode times include peer prefill. Post-last-first-callback tails have changing live batch widths; they are not fixed-C throughput. Outputs differ across concurrency, so token-rate ratios are not equal-work acceleration ratios.

All six formal stages passed process/resource/identity checks and exercised their requested joint verification width. No middecode refresh occurred; this does not qualify refresh isolation or full HTTP/bounded lifecycle.

Profiling and candidate A/B are separate runs. Memory is a sampled peak, not an allocator high-water mark.
