# 9B versus historical 27B LongMemEval parameter audit

Read-only experiment audit on 2026-09-15. Reader code, frozen inputs and running
campaign parameters were not changed. Model size is the intended variable;
the differences below are additional variables. The current experiment is not
a controlled model-size-only reproduction of the historical 27B experiments.

## Evidence and limits

- Historical KVMem: `/data/chaidi/kvmem_eval/results/s500_32k_bt32_meank_unified_manifest.json`,
  including actual server/evaluation argv and git revision
  `23fb2ac6ddf511fdf0e074030e6b17287e30930e`.
  Historical prompt, client and generation defaults were inspected at that revision.
- Historical RAG: `/home/chaidi/kvmem_eval/KVMem_Motivation/configs/exp2_addttft_retrieval_k22.yaml`,
  matching runner/prompt/compaction source, and 102 actual rows in
  `outputs/exp2/addTTFT/retrieval_k22/retrieval_outputs.jsonl`.
  Its archived manifest points to the original Windows experiment paths.
  The local `outputs/exp1/compact_summaries.jsonl` has only 12 rows, all overlapping
  with the current subset. The top-22 retrieval details file is only a Git LFS
  pointer. Therefore configuration-derived facts about the old RAG protocol are
  distinguished from values independently observed across all 102 outputs.
- Current 9B: `benchmark/longmemeval_qwen35_9b_matched102_20260915/plan.json`,
  per-arm `server_manifest.json`, request JSON, results and
  `scripts/kvmem_eval/run_qwen35_longmemeval.py` / `qwen35_lme_rag_worker.py`.
- Primary current judge is the separate DeepSeek V4.1 Flash worker, recorded in
  `external_grading.json`; the original frozen plan still names the initial
  local judge. Do not use the original plan alone to identify current grading.

## Differences affecting quality or protocol

| Item | Historical 27B | Current 9B |
|---|---|---|
| Final output cap | KVMem explicitly 32768. Archived RAG client omits max_tokens (`null`); actual service default is unverified. | Both reader arms explicitly 8192, including thinking. |
| KV precision | KVMem explicitly FP16. Old RAG service precision is not established by the displayed model name. | Both FP8 KV. |
| Final-answer sampling | KVMem temperature 0.6, top-p 0.95, inherited top-k 20; archived RAG configuration temperature 0.0. | Both temperature 0.6, top-p 0.95, top-k 20. |
| Random seed | Historical KVMem argv/client do not set a seed; that revision's generation default field is 0. RAG client does not explicitly set it. | Both server and request explicitly seed 1000. |
| Prompt and history rendering | KVMem uses the archived long-memory instruction, dated `=== Conversation ... ===` blocks and a wrapped final question. RAG has a separate LongMemEval instruction and single assembled prompt. | New shared reader instruction, changed history/date wrappers and question rendering. |
| Query scope | Old KVMem uses the complete final user-message span, including question framing/date. | Uses only the literal question's token span. |
| KVMem ingest path | One full-history-plus-question streaming chat request. | Question-blind raw-token history session, followed by forced selection/full replay for the explicit question. |
| RAG embedder | Archived configuration: `all-MiniLM-L6-v2`. Full 102-row retrieval-detail verification remains incomplete. | Pinned `jina-embeddings-v2-small-en`; reuses the AgentLongBench retrieval implementation. |
| RAG selection | Top 22 chunks; actual complete prompts across 102 rows range 25880–36475 tokens, mean 32414.1; 42 exceed 32768. | Largest ranked chunk set fitting the entire 32768-token prompt, then chronological ordering. |
| Summary policy | Archived config: context window 100000, handoff-compaction prompt, summary plus uncompressed tail. The 12 available summary rows each show one compaction and 2–8 remaining sessions. | Groups of at most 24576 new-history tokens, summary input at most 32768, output at most 2048, non-thinking, new factual-memory prompt, no separate uncompressed tail. Completed runs typically have five rounds. |
| Summary output limit | Archived LLM summary requests omit max_tokens. `summary_max_chars=4000` limits only the mock summarizer and is not an enforced LLM output cap. | Explicit max_tokens=2048. |
| Judge | Historical DeepSeek V4 Pro with official per-type prompts; archived RAG judge usage contains reasoning tokens. | DeepSeek V4.1 Flash, official per-type prompts, explicitly non-thinking, temperature 0, max_tokens=16. Local 27B scores are preserved separately. |

The currently identical reader settings within the 9B comparison do not imply
that either method reproduces its own historical 27B protocol. In particular,
the new compaction/RAG configuration is a baseline change, not just a model swap.
The 8192 cap is an accidental alignment omission. The current score/latency gap
cannot be attributed to any single listed change without a controlled rerun.

## Storage and execution differences

Historical KVMem sets GPU memory ratio 0.5, CPU backing 64 GiB and NVMe budget
256 GiB. Current KVMem uses ratio 1.0, CPU backing 16 GiB and no NVMe. These are
configured capacities, not evidence that all tiers were exercised in every run.

The historical engine revision differs from the current frozen executable;
the latter includes subsequent MTP, checkpoint, prefix and KV management work.
Historical environment flags explicitly force attention split counts to one;
the current runner clears QW3-prefixed variables and uses current defaults.
Historical KVMem keeps a server across samples and uses streaming chat; current
9B starts a fresh server per arm and uses non-streaming raw completions. Server
loading is excluded from current phase costs. These timing definitions and
execution choices require alignment before cross-run efficiency comparisons.

Historical RAG's reported model is `qwen3.6-27b-fp8`, accessed through port 8000.
This does not establish its actual weight format, KV format, engine, MTP setting,
seed or default output cap. Those fields remain unverified pending the original
server launch record; do not fill them in from the model name or another run.

## Confirmed aligned items

- Same 102 IDs and order, 17 of each of six types, no abstention; current
  question/history/reference content is verified semantically against the archive.
- KVMem selection budget 32768, generation reserve 32768, logical capacity
  262144, block size 32, sink 256 tokens (old eight blocks), recent 0, mean-K
  retrieval, step updates and query conditioning.
- KVMem main weights Q8, MTP chain 4, thinking budget 4096, prefill chunk 2048,
  temperature 0.6 / top-p 0.95 / top-k 20. Same MTP chain does not imply the
  same MTP implementation across engine revisions.
- RAG configured chunk size 1024 and overlap 128. Chunk/date formatting and
  embedder differ, so these numbers alone do not establish identical retrieval.

## Proposed alignment order

1. Specify the reference protocol explicitly. Preserve the existing 8K-output
   campaign as its own revision; do not silently overwrite its predictions.
2. Align output cap, prompts/query span, ingest protocol, sampling/seed and KV
   precision. A strict model-size comparison needs one shared protocol for both
   model sizes; some differences cannot be repaired by changing only max_tokens.
3. Fix the RAG embedder, summary prompt/trigger/output cap/tail handling and
   selection rule. A hard 32K total cap and an old uncapped top-22 policy cannot
   both be reproduced literally; choose one shared protocol and label it.
4. Regrade both models' saved answers with one frozen judge configuration.
   This can align judgment without rerunning reader inference.
5. For efficiency claims, additionally align storage, backend, warm/cold server
   lifecycle and timing boundaries; record unknown historical fields explicitly.
