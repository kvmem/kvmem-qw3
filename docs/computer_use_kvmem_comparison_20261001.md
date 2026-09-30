# KVMem on/off: short desktop-history replay

Date: 2026-10-01. This is fixed-input inference replay, not a live-task success-rate comparison.

All six profiles completed all nine requests (three frozen cases × three repetitions): **54/54 completed**. Output checks found the expected initial Notepad action and expected filenames in final summaries. No replayed tool was dispatched.

## Matched configuration

- Same local Qwen3.8-27B Q8_0 weights, vision tower, binary and RTX PRO 6000 Blackwell Server Edition (97,887 MiB).
- SHA-256 identities for the target, vision and draft assets are recorded in `computer_use_model_assets_20261001.json`.
- Binary SHA-256: `e14b429817f42cedee5659c786087f0265f6335a58073d884a2108782ca9a2ad`.
- Logical context 1,048,576, FP8 target KV, temperature 0, maximum generation 512 tokens.
- KVMem on: K131072/G65536, prefill budget 131072, block size 128, 48GiB CPU tier capacity. Off: dense KV, same logical context and exact request inputs.
- Target-only, native MTP chain 4, and DFlash 2 with seven maximum draft tokens run separately. Native MTP and DFlash are never combined.
- Prefix reuse disabled for every profile. Vision cache starts fresh per profile and is retained across the same ordered sequence of requests. The first pass is reported separately from the two subsequent passes.
- Request lengths: initial desktop 3,532 tokens; Notepad history 21,126; browser history 33,079. The last includes 21 screenshot images / 21,000 visual tokens.
- Images and histories are identical across profiles and hashed in the frozen corpus manifest. All six profiles reported the same prompt counts for each case.

## Main comparison at 33,079 tokens

Warm values below are medians of repetitions 1 and 2. TTFT is measured at the first visible streamed text/tool output. Decode throughput comes from the engine's separated decode timing. GPU peak is the maximum total-device sample over the profile, sampled approximately once per second, including startup.

| Decoder | GPU peak on / off (GiB) | Warm TTFT on / off (s) | Warm decode on / off (token/s) | Warm request on / off (s) |
|---|---:|---:|---:|---:|
| Target only | 37.37 / 62.30 | 8.76 / 8.49 | 43.75 / 44.84 | 12.92 / 12.55 |
| Native MTP | 38.67 / 65.37 | 8.90 / 8.54 | 100.62 / 103.85 | 10.71 / 10.30 |
| DFlash 2 | 42.45 / 67.54 | 8.82 / 8.55 | 99.24 / 99.50 | 10.66 / 10.38 |

For this short-history profile, KVMem reduces the sampled DFlash GPU peak by about 37%, while decode throughput at 33K is nearly unchanged and TTFT is slightly higher. The large allocation difference also reflects the configured 1M logical context; it is not a measurement of how much memory a tightly capped 33K dense server would need. DFlash is faster than native MTP on the shorter cases here, but does not beat it on every case.

## DFlash across the three inputs

| Input | Cold first-pass TTFT on / off (s) | Warm TTFT on / off (s) | Warm decode on / off (token/s) |
|---|---:|---:|---:|
| initial_desktop | 1.583 / 1.306 | 0.876 / 0.851 | 174.920 / 180.390 |
| notepad_history | 7.013 / 6.347 | 5.454 / 5.305 | 124.810 / 129.785 |
| browser_history | 10.571 / 9.851 | 8.824 / 8.549 | 99.240 / 99.495 |

## Scope and retained evidence

All inputs remain below K128K, with visual history pinned. No sparse retrieval benefit, above-budget quality result, long-history capacity claim, or 48GB physical-card qualification follows from these measurements. KVMem on/off can produce differently worded summaries; completed responses and the limited output checks do not establish full numerical or behavioral equivalence. Output token counts are retained, so total request latency should not be mistaken for equal-length decode speed.

The three live Windows smoke tasks passed with KVMem + DFlash, as reported in `computer_use_codex_m2_20261001.md`. The off profiles above replay inference inputs only. Live on/off success-rate trials from a common VM snapshot remain open.

Runtime evidence: `apps/computer_use/.runtime/comparison-short-02/{summary.json,aggregate.json,requests.csv,output_checks.json}`; each profile includes its command/binary manifest, engine log, raw response chunks and GPU/CPU resource samples. Per-request CPU seconds, RSS, token counts and timing are also checked in as `computer_use_short_comparison_20261001.csv`. The corpus is `apps/computer_use/.runtime/replay-corpus-01/`.

The exploratory `comparison-short-01` remains intact. It exposed insufficient 32GiB CPU backing for native MTP at 1M context and a transient busy GPU lock at a profile transition. The final `comparison-short-02` uses the common 48GiB CPU tier and bounded lock waiting; all six profiles completed.

Reproduce: run `compare_replay.py` against the hashed corpus, then `summarize_replay.py` on the result directory. See the application README for commands.
