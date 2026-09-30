# Fresh 9B KVMem versus compaction+RAG comparison

The user narrowed the rerun to **Qwen3.5-9B only**. No 27B inference was
started. The abandoned two-model preparation is retained separately under
`benchmark/longmemeval_aligned_9b27b_20260915/` with a scope-cancellation record.

The active output root is
`benchmark/longmemeval_aligned_9b_20260915/`; reader and judge artifacts are in
its `9b/` directory. There are 102 paired questions (204 fresh reader runs),
using exactly the archived subset and order, with 17 questions per type.

| Shared reader setting | Value |
|---|---|
| Model | Qwen3.5-9B Q8 main and MTP weights |
| Engine | Same frozen qw3 binary for both policies |
| KV precision | FP8 |
| Active history/input budget | 32768 tokens |
| KVMem generation reserve | 32768 tokens |
| Final request output cap | 32768 tokens, thinking plus answer |
| Thinking budget | 4096 tokens |
| Answer sampling | temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 1.5, repetition penalty 1.0 |
| Seed / MTP / prefill chunk | 1000 / 4 / 2048 |
| Logical server capacity | 262144, including on the paged compaction+RAG server |
| Final QA instruction, question/date format, tokenizer | Shared implementation |
| Lifecycle / timing | Fresh server per arm; server loading excluded from service phase costs |
| Judge | DeepSeek V4.1 Flash, non-thinking, temperature 0, official LongMemEval per-type prompts |

The answer sampling recipe follows the model card's general-thinking setting:
<https://huggingface.co/Qwen/Qwen3.5-9B>.
Summarization remains deterministic and question-blind, with temperature 0,
thinking disabled and an explicit 2048-token summary-output cap. This is a
policy-specific maintenance request, not the final answer cap.

KVMem uses mean-K, 32-token blocks, sink 256, recent 0, exact literal-question
token spans and full query replay. It ingests history before receiving the
question. Storage is GPU ratio 1.0, CPU budget 16 GiB, NVMe disabled, GC disabled.

Compaction+RAG uses the pinned Jina v2 small embedder, 1024-token chunks with
128 overlap, dated per-session indexing, rolling factual summaries over groups
of at most 24576 new-history tokens, and no separate uncompressed tail. The
complete final prompt is capped at 32768 by ranked selection followed by
chronological ordering. Actual retained information is determined by each
policy; matching budgets does not equate the policies' selected histories.

Both policies now use one explicitly frozen protocol. This is a fresh 9B
comparison, not a claim that the incomplete historical 27B RAG run had identical
settings. Old reader answers and old grades are not reused. Only identical,
model-independent CPU Jina indices are reused, verified against the original
retriever manifest and 204 file hashes. Their original measured preprocessing
times and provenance remain available; do not count index reuse as fresh
runtime work when reporting actual campaign wall time.

The previous 8K-output run was stopped after a complete pair. All 178 completed
arm result files were hash-verified and preserved. Nothing was deleted based on
correctness. See `previous_run_preserved.json` and `index_reuse_audit.json`.

Parameter checks exercise the actual request-building function for both arms,
including the 32768 cap, sampling and thinking settings; an accidental 8192
answer cap raises an assertion. Common server arguments are compared directly.
The shared HF tokenizer's IDs and merge rules are verified against the 9B GGUF.
`source_snapshot/`, `identity.json`, per-arm server manifests and request JSON
retain the exact code, model identities and effective settings.

Run with:

```bash
.venv/bin/python scripts/kvmem_eval/run_longmemeval_aligned_models.py
python3 scripts/kvmem_eval/score_qwen35_longmemeval_deepseek.py \
  --output-root /home/chaidi/qw3/benchmark/longmemeval_aligned_9b_20260915/9b \
  --watch --prompt-key
```

The cloud judge receives only question, reference/rubric and final answer.
Its key is held in process memory, not saved in commands, source or artifacts.
It grades only completed pairs and stores immutable snapshots separately from
reader results. Truncated answers remain labeled as such and are not silently
removed or resampled.
