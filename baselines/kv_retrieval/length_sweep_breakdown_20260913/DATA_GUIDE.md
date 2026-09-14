# Data definitions and comparison boundaries

This directory contains a fresh Qwen3.8-27B Q8_0 / FP8 KV microbenchmark on one exclusive RTX PRO 6000 Blackwell Server Edition. It is not a new agent-accuracy evaluation. No production qw3 or upstream implementation files were changed by this experiment.

The historical tokens come from a captured DeepSWE `superjson-error-stack-serialization` trajectory, via `../decode_comparison_20260912/inputs.json`. Each length uses the same chronological prefix followed by the same chat suffix. The lengths are 8192, 16384, 32768, 65536, and 131072 tokens. 245760 is supplemental and must not be silently substituted for 128K. These are history lengths, not four equal active-attention budgets.

| System | Execution and memory configuration |
| --- | --- |
| qw3 | Frozen candidate_v4 native engine; continuous batching with one active request; prefix cache; context 256K; no summary compaction in this fixed-token test |
| KVMem | Same frozen native executable and weight file; single-request path; select 128K, generation 128K, block 32; history/index on CPU, context capacity 2M; no NVMe or GC |
| InfLLM | Author context manager through isolated Qwen hybrid-model adapter; initial 128, local 4096, retrieval 16 × 128, GPU cache 32 blocks; 2 CPU threads; async global stream disabled for validated speculative rollback |
| InfiniGen | Isolated Qwen adapter of author selection/prefetch algorithm; alpha 4, partial ratio 0.2, selection cap 2048 per query head; exact GPU integer sorting/dedup and query-only forecast projection; 8 CPU threads |

All use batch 1, greedy decoding, 128 generated outputs, MTP=0 and MTP=4, three repetitions. The external adapters use the exact Q8 weight mapping with their isolated Q8 kernels and BF16 HF/FLA activations; the native engine uses different fusion, scheduling, residual precision, and cache representation. External MTP uses its full-history draft cache. Therefore, the total gap is a system comparison; it cannot all be attributed to retrieval policy. Internal exact-output checks validate external optimizations/MTP against serial generation from the same restored prepared state. They do not establish equal outputs or task accuracy across different methods.

Native MTP output is not byte-identical to native serial output at every point: KVMem 32K and qw3 128K differ in this sweep. The CSV explicitly records `native_output_text_match_mtp0`; acceptance and verification counts are also retained. Do not describe all four systems' MTP as empirically byte-exact in this test. Use MTP-off results when attributing the non-speculative retrieval/engine gap, and avoid treating MTP-on rate differences as pure cache-management overhead.

## Metrics

- `steady_decode`: 127 useful outputs after the first token divided by that decode interval. Initial prefill is excluded.
- `new_query`: Prepare N history tokens, append the same question (27 text tokens, 40 with chat wrappers), and generate 128 outputs. `effective_tokens_per_s = 128 / request_s`. This includes actual query prefill, working-set refresh, KV movement, replay, and generation. It excludes preparing the pre-existing history.
- The steady-decode and new-question scenarios produce different answers. Their throughput difference alone is not a measurement of query-maintenance cost. For that purpose use TTFT/phase timers and compare effective throughput with the decode rate inside the same new-query request (`new_query_decode_mtp*.csv`).
- Native new-query timing is server request time, with HTTP client wall time retained separately. External timing starts from a ready query tensor inside the process. Client serialization, tokenization, and Claude Code history management are not measured as one identical end-to-end interface.
- Native plain/MTP=0 logs do not expose all separate semantic/replay timers. These unavailable fields stay blank, rather than being reported as zero. Actual replay events are parsed from both native log formats. Full request time/TTFT still includes that work in either path.
- Native query repetitions start from fresh servers, with an untimed 1024-token/128-output warmup. External repetitions restore the same prepared recurrent state, KV history boundary, retrieval state, and MTP boundary. The new question cannot be reused from a previous repetition.
- Native raw `environment` is the inherited arm configuration; effective MTP=0 overrides set `QW3_KVMEM_GUIDED_MTP=0` and disable the CLI automatic guided-refresh option, which otherwise rejects startup with `mtp-chain=0`. This does not disable explicit-user-query selection or replay. Newer records include `effective_environment` directly. The zero-request startup rejection is archived in `failed_attempts/mtp0_guided_cli_startup/`.
- A new query does not necessarily trigger semantic selection below the 128K budget. Actual native phase counters and raw logs determine whether it ran. At 128K the appended request has 131112 tokens; 240K provides another above-budget check. The completions route uses the supplied question directly; CLI guided-refresh defaults do not automatically insert the chat harness's internal query-generation request. Full query replay remains enabled.
- `guided_query_kvmem_mtp4/` separately measures an explicit request for internal guided-query generation (thinking=0, max query=256) at 128K and 240K, one repetition each. Include its generation, selection and full replay in effective throughput. Do not call the direct-query main table a measurement of this more expensive path, or treat the one-run supplement as three-repetition evidence.
- Internal guided-query tokens are auxiliary work and do not increase the 128 useful-output-token numerator. `kvmem_query_phases.csv` includes the native phase accounting where available; phases are wall-clock intervals, not device-exclusive compute times.
- The zero-thinking guided supplement explicitly sets `kvmem_query_guided_direct=true`; this still generates an internal query. The initial attempt missing this required flag was rejected with HTTP 400 before the measured request ran and is retained in `failed_attempts/guided_zero_thinking_missing_direct_flag/`. Actual internal query token counts are exported separately from the 256-token maximum.
- Guided outcomes are retained without filtering: 128K generates 20 query tokens; 240K immediately ends and logs `fallback-original reason=empty-output`, then uses the original 27-token question. `guided_query_generated_tokens=0` in the latter case; `guided_query_used_tokens=27` is not auxiliary generation. Its total request is valid for the attempted-guided policy, but must not be called a successful guided-generation latency sample.
- InfLLM uses the author's whole-chunk append for supplied user tokens. Earlier per-token append results in `external_infllm.json` are retained under `query_serial_append_diagnostic`; headline query rows come from `infllm_chunk_query.json`. Generated speculative candidates still undergo sequential retrieval verification. InfiniGen keeps causal per-position selection with dense compute batched up to five positions.

## Files

- `measurements.csv`: individual formal, diagnostic and appended-query measurements, repetition IDs, output hashes and source JSON paths.
- `summary.csv`: median/min/max for every scenario, method, history length and MTP setting.
- `steady_decode_mtp0.csv`, `steady_decode_mtp4.csv`: compact four-system decode tables.
- `new_query_mtp0.csv`, `new_query_mtp4.csv`: compact effective-throughput tables including new-query processing.
- `supplement_240k.csv`: separately labelled above-budget supplemental results. The compact tables above contain only the five requested lengths.
- `host_traffic_profiles.csv`: separately instrumented host spans and logical retrieval/cache-traffic counts, including rejected MTP positions.
- `trace_activity.csv`, `trace_copy_groups.csv`, `trace_kernels.csv`, `trace_runtime.csv`: actual Nsight activity for a separate 128K new-query request, not the unprofiled throughput samples.
- Native tracing repeats the same smaller-length request prelude as the formal ladder, then captures only the 128K query. This preserves the formal checkpoint/cache retention state; earlier requests are outside the capture range.
- `analysis_status.json`: completion/identity/equivalence checks. Partial files during execution are not final results.
- `pause.json`, `resume_status.json`: OSWorld interruption/archive and verified restoration records. Interrupted arm data are not scored as failures.

## Interpreting the breakdown

GPU-source diagnostics retain identical selected KV but allocate extra GPU history. Oracle-ID diagnostics replay previously recorded identical selections; they cannot serve unknown future requests. They estimate removable costs inside the same implementation and are not deployable baselines. InfLLM's existing author GPU cache/LRU remains enabled in formal measurements. InfiniGen's formal measurements retain selected-row CPU gather and prefetch.

Do not add host enqueue, worker wait, GPU stream durations, or copy durations as disjoint wall-time percentages: these overlap. The Nsight export instead computes exclusive kernel/memset-only, copy-only, compute–copy overlap, and no-recorded-GPU-activity intervals. The last category is not a direct measurement of CPU execution or SM occupancy. GPU kernels include model and retrieval computation. Copy bytes describe transfers across devices; they do not measure all HBM traffic during attention.

The external raw `extra_resident_history` string inherited by the first InfLLM run is stale boilerplate. Full-history GPU shadows are allocated by `resident()` only for GPU-source diagnostics; use the experiment code and explicit scenario, not that inherited string, to interpret formal memory placement.

Do not infer an accuracy win from throughput, an equal active window from an equal input length, or an intrinsic algorithm-speed ratio from a comparison that also changes engine kernels and MTP implementation.
