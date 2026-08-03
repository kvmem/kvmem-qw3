# MemoryAgentBench plain-baseline parity

Date: 2026-08-02

## Compatibility finding

`/home/chaidi/kvmem-efficiency-bench` and the baseline implementations under
`/home/chaidi/AgentLongBench-Long` are method references, but their runners
cannot be used directly for the current MemoryAgentBench experiment.

| Requirement | Existing efficiency runner | Required here |
|---|---|---|
| Evaluation unit | AgentLongBench/BEAM sample with one question | One parquet context row with 1–100 independent questions |
| Full run | One answer per selected sample | 146 contexts / 3,671 questions |
| Context reuse | Not defined across questions | Transform/build once, then frozen independent branches |
| Output budget | Fixed 32 tokens | Official source-specific 10–2,000 tokens |
| Final sampling | temperature 0, top-p 1 | temperature 0.6, top-p 0.95, top-k 20 |
| Thinking | Off | Off (already matched) |
| RAG preparation | Re-embeds history in every `retrieve()` call | Embed/index full context once per row |
| Scoring | Efficiency only; no official judge | Same deterministic and API judge scripts as KVMem |

The new runner is
`scripts/kvmem_eval/run_memoryagentbench_baselines.py`. It preserves the old
repositories and only imports their pinned method semantics:

- original question-independent Codex checkpoint-compaction prompt;
- pinned Jina v2 small English model and Qwen-token 1024/128 chunks;
- final-only global RAG, selected by relevance and presented chronologically;
- a complete-prompt 32,768-token recent sliding window.

## Frozen-branch semantics

The official row is rendered by the same
`prepare_memoryagentbench_archive_row.py` used by the KVMem experiment. Each
method then follows this sequence:

1. Build the shared representation once:
   - Compact: one question-independent summary/cursor chain;
   - RAG: one set of full-context embeddings, followed by independent query
     scoring;
   - Sliding: one fixed recent suffix that fits every question in the row.
2. Prime an ordinary dense QW3 lossless prefix-cache entry with that shared
   representation.
3. Send every question as a fresh two-message request (official system + one
   user message). A previous question or answer is never appended.
4. For Compact+RAG, append only the question-specific retrieved blocks and
   question after the reused compact prefix.

The common prefix is padded only with newlines until its open-user Qwen token
length is exactly divisible by the 16-token physical KV page size. Warmup and
scored requests use the same trailing separator. In addition, this benchmark
sets `QW3_PREFIX_CACHE_COMMIT_GUARD_PAGES=1`, so the published entry ends one
complete KV page before the natural warmup boundary. The guard is necessary
because a byte-identical BPE prefix can still have a different final token when
the following question changes; without it, some rows repeatedly cold-prefilled
about 159K tokens per question. The server default remains zero, so unrelated
prefix-cache users are unchanged. The full launcher enables
`QW3_PREFIX_CACHE_TRACE=1` and rejects its two-question gate unless both
questions produce dense `prefix_cache hit` records.

The long-lived baseline service also sets
`QW3_PREFIX_CACHE_MAX_PAGES=8192`. The server default remains unlimited, but an
explicit internal limit is now preserved instead of being overwritten at
startup. This matters for hybrid models because every distinct cached prefix
owns both pinned normal-attention pages and a recurrent-state checkpoint. A
32K frozen-branch workload otherwise creates one checkpoint per question while
placing too little pressure on the much larger 256K KV pool to trigger its
emergency eviction path. The bounded LRU keeps the actively reused row prefix,
evicts completed question-specific entries, and is lossless: it changes only
whether an old prefix is recomputed, never the prompt or attention semantics.
The real Window32K recovery run held the configured bound, produced cache hits
without misses, and stabilized at roughly 38.6 GiB instead of exhausting the
device after about 300 questions.

This cache is not KVMem: `--kvmem` is absent, no sparse block scoring or
selection occurs, and reused pages retain ordinary dense-attention KV. Plain
baseline servers disable MTP because QW3 prefix-cache v1 explicitly routes MTP
requests through cold prefill. MTP is a speculative decode optimization rather
than a context-processing method; disabling it is required here so the frozen
row prefix is genuinely reused instead of recomputed for every independent
question. The two-question gate verifies two explicit `prefix_cache hit`
records before a full method run is accepted.

## Frozen parameters

| Parameter | Value |
|---|---:|
| Model | Qwen3.6-27B-Q8_0 |
| Dense KV | FP8 |
| Context window | 262,144 |
| Generation reserve for compact methods | 32,768 |
| Prefill chunk | 2,048 |
| MTP | disabled for plain baselines |
| Parallelism | 1 |
| Dense prefix-cache commit guard | 1 page (16 tokens) |
| Dense prefix-cache page budget | 8,192 pages (internal bounded LRU) |
| Final sampling | temp 0.6, top-p 0.95, top-k 20, no thinking |
| Compact input | 232,000 |
| Compact summary output | 25,000 |
| Shared compact open-prefix cap | 192,000 |
| RAG | Jina, 1024-token blocks, overlap 128, top-30 |
| Sliding window | 32,768 complete prompt tokens |

Window32K's method-level definition is pinned to both
`AgentLongBench-Long/script/slidingWindow/run_sliding_window.py` and the prior
`kvmem-efficiency-bench/configs/runs/qw3_sliding_window.json`; their paths and
SHA-256 hashes are stored in every phase-specific run config. The reused
semantics are the 262,144-token server context, a 32,768-token *complete final
prompt* budget, and selection of the largest recent suffix from the original
history. Model/KV precision, task-specific output limits, sampling, and MTP are
comparison controls rather than properties of the window algorithm, so they
match the current MemoryAgentBench/KVMem experiment as listed above. In
particular, MTP is disabled because the current QW3 dense-prefix-cache route
would otherwise cold-prefill every independent question.

The 192K compact-prefix cap is intentionally question independent. The largest
official formatted question is 3,739 Qwen tokens. It leaves enough room under
the 229,376-token final-input cap for a top-30 RAG payload (at most 30,720 raw
chunk tokens plus labels) without changing the compact checkpoint between
Compact-only and Compact+RAG. For histories with fewer than 30 chunks, RAG
selects every available chunk and records the smaller effective top-k; a strict
T30 requirement would make the 6K/32K MemoryAgentBench sources undefined.

## Execution and artifacts

The orchestration entry point waits for the current KVMem generation to reach
146 completed contexts before using the GPU:

```bash
tmux new-session -d -s mab_plain_baselines_20260802 \
  'cd /home/chaidi/qw3 && \
   bash scripts/kvmem_eval/run_memoryagentbench_baselines_full.sh'
```

It performs, in order:

1. exact official prompt preparation;
2. offline GPU RAG preparation while QW3 is stopped;
3. a two-question prefix-cache gate and full Compact-only run;
4. server restart, gate, and full Compact+RAG run;
5. server restart, gate, and full Sliding Window run;
6. official deterministic scoring for each method.

The length phases are finalized in benchmark order. After the >262,144-token
generation writes `status_over256k.json`, run
`finalize_memoryagentbench_over256k.sh`: it hard-checks 30 contexts / 1,316
questions per method, completes the DeepSeek special metrics, verifies the
cached KVMem judge, and writes `comparison_over256k.{json,md}`. The <=262,144
phase starts only after that long-context comparison exists.

If a completed method's post-run scorer fails because of an environment issue,
the long-context launcher is resumable at method boundaries via
`START_METHOD=compact-rag` or `START_METHOD=sliding-window`. Completed row
artifacts are never deleted or regenerated.

Default workspace:

```text
/data/chaidi/kvmem_eval/results/memoryagentbench_plain_baselines_20260802
```

Each method writes the same scorer-facing schema as KVMem under
`methods/<method>/rows/*/results.jsonl`. Compact and RAG checkpoints live under
`shared/`; per-question writes are fsynced and resumable. API-based LongMemEval
and InfBench judging is intentionally not launched with an embedded credential;
after generation it uses the existing `judge_memoryagentbench_special.py` with
`DEEPSEEK_API_KEY` supplied only through the process environment.

The generator/RAG runner and official scorer intentionally use separate Python
environments. Generation uses `kvmem-efficiency-bench/.venv-rag`; deterministic
official scoring uses `KVMem_Motivation/.venv`, which contains the complete
MemoryAgentBench dependency set including `editdistance`, `rouge-score`, and
scikit-learn. Launchers expose this as `SCORER_PY` so a missing optional scorer
dependency cannot discard or force regeneration of completed model outputs.

The finalizer also runs `audit_memoryagentbench_comparison.py`. It refuses a
completed result unless all four methods contain the same 146 contexts / 3,671
question identities, all 400 API judgments and deterministic official metrics
exist, Compact and Compact+RAG use byte-identical shared compactions, RAG stays
within its 30,720-token source-block ceiling and varies by question, Window32K
uses a fixed row suffix, and the >256K / <=256K / full comparison tables contain
30/116/146 contexts respectively. The machine-readable verdict is written to
`completion_audit.json`.

## Final >256K comparison

The 2026-08-03 long-context phase contains exactly 30 contexts and 1,316
questions whose canonical Qwen archive length is greater than 262,144 tokens.
All methods use the same question identities. The 1,000 deterministic questions
use the pinned local MemoryAgentBench metrics; the remaining 300 LongMemEval and
16 InfBench questions are judged with `deepseek-v4-pro`. KVMem's full run has
400/400 special judgments available, of which the same 316 long-context keys
are selected here. Each plain baseline has exactly 316/316 judgments.

| Aggregation | KVMem Adaptive Mean-K | Compact-only | Compact+RAG | Window32K |
|---|---:|---:|---:|---:|
| Context-macro mean | 50.03 | 33.59 | 41.72 | 19.94 |
| Question-weighted mean | 61.32 | 42.83 | 55.38 | 29.50 |
| Macro-category mean | 40.99 | 27.54 | 34.80 | 17.95 |

The context-macro row weights each of the 30 source contexts equally. The
question-weighted row pools all 1,316 question-level headline metrics. The
macro-category row first forms source-level scores and then weights the four
MemoryAgentBench categories equally; it is the primary ledger value.

| Category | KVMem | Compact-only | Compact+RAG | Window32K |
|---|---:|---:|---:|---:|
| Accurate Retrieval | 70.31 | 41.16 | 61.62 | 30.16 |
| Conflict Resolution | 39.00 | 27.50 | 35.00 | 12.50 |
| Long Range Understanding | 38.25 | 25.19 | 27.91 | 13.04 |
| Test Time Learning | 16.42 | 16.33 | 14.67 | 16.08 |

| Source / official headline metric | Questions | KVMem | Compact-only | Compact+RAG | Window32K |
|---|---:|---:|---:|---:|---:|
| EventQA / substring exact match | 500 | 93.60 | 71.80 | 86.20 | 54.80 |
| FactCon-MH / substring exact match | 100 | 5.00 | 5.00 | 5.00 | 1.00 |
| FactCon-SH / substring exact match | 100 | 73.00 | 50.00 | 65.00 | 24.00 |
| InfBench / summary F1 | 16 | 38.25 | 25.19 | 27.91 | 13.04 |
| LongMemEval-S / LLM judge accuracy | 300 | 52.33 | 30.67 | 47.67 | 9.67 |
| ReDial / recall@5 | 200 | 16.42 | 16.33 | 14.67 | 16.08 |
| RULER QA / substring exact match | 100 | 65.00 | 21.00 | 51.00 | 26.00 |

The strict phase audit passes. It verifies one frozen compaction per context,
byte-identical Compact/Compact+RAG common prefixes, per-question RAG retrievals
bounded to 29,928--30,720 source tokens, one fixed Window32K suffix per context,
identical question IDs, all required metrics, and zero KVMem scorer fallbacks.
KVMem uses `key-direction-adaptive` Mean-K with dynamic 1/2/4 prototypes; all
3,671 scorer events in its full run use the adaptive CPU streaming path without
fallback.

The phase audit also exposed and repaired one harness issue before publication:
the first Window32K smoke request had originally selected its suffix using only
the two smoke questions. Those two answers were regenerated using the fixed
suffix constrained by all 100 questions in that context, and the runner now
keeps the complete question list when `--question-limit` restricts generation.
After repair, every one of the 30 Window rows has exactly one shared-context
hash. The regenerated two deterministic answers remained incorrect, so this
changed experiment validity but not the displayed aggregate scores.

Authoritative artifacts:

- `/data/chaidi/kvmem_eval/results/memoryagentbench_plain_baselines_20260802/comparison_over256k.json`
- `/data/chaidi/kvmem_eval/results/memoryagentbench_plain_baselines_20260802/comparison_over256k.md`
- `/data/chaidi/kvmem_eval/results/memoryagentbench_plain_baselines_20260802/completion_audit_over256k.json`
- `/data/chaidi/kvmem_eval/results/memoryagentbench_kvmem_archive_20260802_full/run_config.json`

The <=256K generation phase was explicitly deferred after this cohort finished.
The runtime sentinel `run/skip_under256k` is present; no <=256K baseline model
requests were launched. Earlier `prepare_under256k_early.log` files contain only
prompt/manifest preparation and are not model outputs or scored results.
