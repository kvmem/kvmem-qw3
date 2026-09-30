# 9B rerun of the archived 27B Compaction+RAG pipeline

The user requested reusing the old summary, retrieval, and prompting pipeline,
replacing its reader model with Qwen3.5-9B. This run regenerates 102 summaries
and answers as required by that pipeline. It does not rerun 27B or KVMem.

Artifacts: `benchmark/longmemeval_9b_legacy27b_compactrag_20260915/`.

## Source and protocol

The original repository is `/home/chaidi/kvmem_eval/KVMem_Motivation`, revision
`84f86d8f1f8551c01e978ec11b8c0144a34e4038`. The `legacy_source/` directory is a
byte-for-byte git export, with SHA256 hashes in `legacy_source_manifest.json`.
The runner calls the original `make_compact_summaries` and `run_retrieval`
functions. Instrumentation records their requests and responses without
replacing the prompts or modifying generated text.

The working tree of that repository has subsequent edits. In particular its
current 99%-of-capacity compaction default is not the archived experiment's
behavior. The exported original code triggers when the assembled trial prompt
exceeds **100,000 tokens**, then summarizes previously accumulated sessions and
keeps the newly arriving session and subsequent sessions as uncompressed tail.
Twelve archived summary fixtures check the exact overflow boundary, compacted
session IDs, source token count, and retained tail. These fixture summaries are
only used for validation, never as the 9B experiment's generated memory.

| Item | Restored behavior |
|---|---|
| Input | Same 102 archived question IDs and histories |
| Summary prompt | Original context checkpoint / handoff prompt |
| Summary and final-answer temperature | 0 |
| Summary and answer thinking | Enabled; no added thinking-token cap |
| Output cap | Omitted from requests; no added 2K, 8K, or 32K cap |
| Memory | Generated summary plus uncompressed tail |
| Embeddings | all-MiniLM-L6-v2, local cached weights, CPU |
| Retrieval chunks | 1024 tokenizer tokens, overlap 128 |
| Selection | Fixed top-22, descending similarity, original duplicate handling |
| Final prompt | Original summary + tail + retrieved chunks + dated question |
| Final input cap | No additional hard 32K trimming |

The original MiniLM encoder defaults to a 256-token encoding limit even though
retrieval chunks are 1024 reader tokens. This behavior is deliberately kept.
Changing encoder truncation, deduplicating the tail against retrieved chunks,
or sorting retrieved chunks chronologically would change the original method.

The cached MiniLM snapshot is
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; its files and package versions are
recorded. The historical archive did not pin the encoder revision or hardware.
No Jina indices, summaries, or answers from the newer 9B baseline are reused.

## Serving and comparison limits

The reader is the existing Qwen3.5-9B Q8 GGUF, served by the same frozen qw3
binary as the preceding 9B run, with FP8 KV, MTP=4, and logical context 262144.
The native server has no explicit output cap: omitted `max_tokens` resolves to
remaining context. Sampling defaults are temperature 0, top-p .95, top-k 20,
seed 0, presence penalty 0, repetition penalty 1. Both summary and answer use
the old client request fields. A persistent server handles the campaign.

The original 27B service used vLLM. Its exact runtime defaults were not fully
archived. This run restores the **application-level pipeline**, not a claim
that model size is the only change across the entire historical software stack.
The new and old tokenizer vocabularies are identical; the single-user-message
chat template used by this pipeline is also identical. Each request records
the expected HF prompt token count and actual native usage.

`kvmem_reference_reuse.json` identifies the existing 102 9B KVMem answers by
hash. Symlinks and cached matching grades expose them as reference results;
they are not newly generated or relabeled as following the legacy RAG prompt
and sampling protocol. The preceding fresh 9B Jina/32K baseline remains intact.

The existing DeepSeek V4.1 Flash judge is used with official LongMemEval
per-type prompts. Reader errors and output-limit terminations remain visible.
The old runner's end-to-end retrieval field includes history embedding; it
must not be compared to the preceding run's pre-indexed query-only timing
without separating those phases. Its estimated recompute time is an estimate,
not measured prefill time.

## Run

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS=8 \
  /home/chaidi/kvmem_eval/KVMem_Motivation/.venv/bin/python \
  scripts/kvmem_eval/run_longmemeval_9b_legacy_compactrag.py
```

The Python retriever has CUDA hidden; the child native reader receives a clean
GPU-enabled environment. Proxy variables are removed from the launcher and
native server; all model assets are already cached locally. The reader is
launched in a detached process group, with status, stdout, and server logs
under the artifact root.

## Failure-handling revision v2 (2026-09-15)

Sample `118b2229` generated 162620 reasoning tokens with no final summary and
`finish_reason=length`. The old client returned reasoning when final content
was empty, and the original runner propagated that as memory. Two downstream
requests followed; their artifacts remain available but do not belong to a
valid continuation of the repaired pipeline. The third was manually interrupted.

The user authorized fixing the issue and continuing. Valid requests, prompts,
retrieval, sampling, thinking and output limits are unchanged, verified by
`revisions/failure_handling_v2/validation.json`. The original source export and
v1 identities are preserved. The revised wrapper rejects truncated or empty
summaries before another request can consume them. Empty final answers are
also rejected; nonempty length-terminated answers remain available for grading.
Incomplete SSE/network failures are infrastructure errors, not model failures.

The original first failed request is recorded as a terminal sample failure,
without regenerating that sample or altering parameters. Such failures count
as unsolved in the full cohort. Counters distinguish `processed_samples`,
`completed_samples` (valid generated answers) and `failed_samples`.

`score_longmemeval_legacy_compactrag.py` verifies terminal-failure evidence by
response hash and final-content checks, then records an unsolved outcome without
calling the LLM judge. Valid answers use the unchanged DeepSeek judge/rubric.
Its separate `judge_deepseek_v41_flash_v2/` report includes failures in the main
denominator and additionally shows answer coverage and conditional accuracy.
The v1 judge reports remain intact; cached grades for identical valid answers
are reused. This repairs propagation/accounting, not the model's tendency to
loop under the unchanged generation parameters.
