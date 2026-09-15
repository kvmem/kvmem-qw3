# Qwen3.5-9B memory-QA pilot

This experiment compares KVMem and rolling compaction plus RAG on the official
cleaned LongMemEval-S histories. It tests memory-answer quality; its compactor is
not Claude Code's native auto-compaction implementation.

Artifacts: `benchmark/longmemeval_qwen35_9b_20260915/`. The HF mirror download
clears HTTP/HTTPS/ALL_PROXY in both cases, pins dataset revision
`98d7416c24c778c2fee6e6f3006e7a073259d48f`, and verifies the file against its LFS
SHA-256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.
It is a separate data revision from the old 27B experiments.

The fixed pilot uses four answerable questions from each of six question types
and six abstention questions. Within each pool, sort SHA256 of
`20260915:<question_id>`; the same key orders the resulting thirty questions.
No model results enter selection. This balanced sample does not reproduce the
full 500-question distribution. Gold answers and evidence annotations are
separated from inference inputs; only the grader reads the reference answers.

Both readers use the same frozen qw3 binary, Qwen3.5-9B Q8 main/MTP weights,
FP8 KV, MTP=4, temperature 0.6, top-p 0.95, top-k 20, seed 1000, thinking cap
4096, and maximum total output 8192 tokens. The final answer prompt has a 32768
token budget. Each arm gets a fresh server; policy order alternates by question.
Full prompts, intermediate summaries, model outputs, timings, and process/model
identities are retained. An infrastructure error stops the runner for review;
an incorrect or truncated model answer remains in the results.

KVMem uses 32-token blocks, a 32768-token selection budget, a 32768-token
generation reserve, mean-K retrieval, a GPU index and CPU backing with no NVMe
or GC. Recorded history is prefetched without the final question through the
raw-token session API. The prefix ends on a block boundary; at most 31 history
tokens are delivered with the final question. Final-query selection is forced
using the question's exact token span, and the normal query replay executes.
This QA task supplies an explicit query; it does not require synthesizing an
additional guided query. The native tokenizer is checked against the local
tokenizer on the complete prompt.

Compaction uses the same 9B reader to recursively summarize all chronological
sessions. Each new-history group is at most 24576 tokens, the entire summary
prompt at most 32768, and the summary output at most 2048. Sessions remain
intact. Summary requests never see the final question or its answer. There is
no separate uncompressed tail: the final prompt combines the resulting memory
summary with retrieved excerpts from the full raw-history archive.

RAG reuses the pinned Jina v2 small implementation and weight checks from the
existing retrieval baseline. It indexes per-session 1024-Qwen-token chunks with
128-token overlap and includes dates. Jina input lengths are checked against
8192 to prevent silent truncation. CPU indexing occurs before the query;
query embedding and ranking occur after history compaction. The largest prefix
of ranked chunks that fits the **complete** 32768-token answer prompt is then
ordered chronologically. Index construction, summarization and final-query work
are recorded separately. Equal final prompt limits do not imply equal peak
memory use or identical ingestion work.

Initial grading uses a separate local Qwen3.8-27B Q8 model at temperature zero,
with the official LongMemEval per-type/abstention prompt function. It first
checks positive and negative calibration examples. Method names are absent
from judge prompts and judgment order is deterministically shuffled. This is a
local-judge pilot, not an official leaderboard score. Every raw hypothesis is
retained so a stronger external judge can later re-grade without inference reruns.

Run with the repository `.venv/bin/python`:

```bash
python scripts/kvmem_eval/run_qwen35_longmemeval.py --limit 1 --prepare-only
python scripts/kvmem_eval/run_qwen35_longmemeval.py --limit 1
python scripts/kvmem_eval/score_qwen35_longmemeval.py --limit 1
python scripts/kvmem_eval/run_qwen35_longmemeval.py
python scripts/kvmem_eval/score_qwen35_longmemeval.py
python scripts/kvmem_eval/report_qwen35_longmemeval.py
```

The runner records and checks source/model/data identities when resuming. Do
not overwrite an incomplete arm or silently combine results across revisions.
The old DeepSWE/OSWorld artifacts remain separate and preserved.

Preflight checked all thirty histories: full prompts span 106463--110415 Qwen
tokens, and the largest individual session is 6298 tokens. The information-flow
and complete-prompt budget regression checks both pass.

The first judge calibration attempt is retained in
`judge_calibration_attempt1/`. It exposed an invalid positive fixture: the
preference reference is a rubric, which cannot be passed verbatim as a correct
assistant answer. Before grading any candidate predictions, this calibration
pair was replaced with an independent synthetic vegetarian/spicy-food fixture
and actual positive/negative replies. The second calibration passed all fourteen
checks. This is a basic sanity check, not a validation of judge accuracy on the
complete benchmark. Candidate outputs and the inference protocol were unchanged.

`report/paired_results.csv` exposes individual answers, local judgments and phase
costs. `report/summary.json` includes per-stratum scores and paired discordances.
The report includes only completed, scored pairs and marks whether all thirty
pairs are present. Measured service totals sum ingestion/index/summarization and
final-query phases; model/server loading and uninstrumented client work are
excluded. Raw-completions TTFT may be a thinking token, not a visible final answer.

## NFC parity-check repair and resume

After 22 completed pairs, sample `352ab8bd` stopped before inference because a
history excerpt contained a decomposed combining macron (`where\u0304`). The HF
tokenizer declares NFC normalization; its encoding composes this to `wherē`,
while the native GGUF tokenizer consumes the original bytes. Only the diagnostic
`/tokenize` request now applies the HF tokenizer's declared normalizer. Actual
model input IDs, query offsets, summaries, retrieval and generation are unchanged.

The normalized diagnostic matches HF input IDs for all 30 full prompts; only
this one prompt changes under normalization. All 44 completed result hashes,
the old source/identity, the failed pre-inference directory, and the explicit
identity migration are retained in `resume_nfc_audit/`. Three protocol tests
pass, including the combining-mark regression. No frozen engine/model was
modified. Resume continues the remaining eight pairs without replacing answers.
