# Qwen3.5-9B on the historical 27B LongMemEval-S subset

This campaign uses the **actual 102 archived questions** from
`/home/chaidi/qw3/selected_12_samples.jsonl`. Despite its filename, this file
contains 102 records: 17 per question type, with no abstention questions.
Its SHA-256 is
`7bdc816736e4505d16d0fa196415ee32cb63314e9f64b7341a50ba76f4ec888c`.

Both IDs and order match the historical 27B full-context and compaction+RAG
top-22 outputs. The historical 500-question KVMem run also includes all 102
IDs. Do not reconstruct this subset with `dataset.py`'s sorted-ID recipe: that
recipe overlaps with only 27 of these actual archived questions.

The complete question/history/reference records of all 102 questions match
the downloaded official cleaned LongMemEval-S data semantically. The preceding
30-question 9B pilot is complete. Seven of its question payloads match this
subset exactly, so both completed arms are copied with hashes and provenance;
the other 95 question pairs require new inference. No answers are chosen or
replaced based on correctness. The original plan used a local 27B judge.
On September 15 the user requested DeepSeek V4.1 Flash for grading; the
external judge now grades both methods under the same protocol, while the
local judge artifacts remain available for comparison (see below).

## Active-context alignment

| Setting | Historical 27B KVMem main run | This 9B KVMem run |
|---|---:|---:|
| Selection budget | 32768 | 32768 |
| Generation reserve | 32768 | 32768 |
| KV block size | 32 | 32 |
| Logical history capacity | 262144 | 262144 |
| Retrieval | mean-K | mean-K |
| Sink / recent tokens | 256 / 0 | 256 / 0 |
| Main MTP chain | 4 | 4 |

These are **initial selected-context budgets**, not a claim that generation
keeps exactly 32768 tokens active throughout. Answer tokens are appended while
generating. The generation reserve is capacity for that growth, not additional
history selected at the start.

The new 9B compaction+RAG arm caps the **entire answer prompt** at 32768 tokens,
including system framing, the summary, retrieved chunks, dates and question.
It uses the same rolling-summary/Jina protocol as the preceding 9B pilot.
Summary requests also have a complete-input cap of 32768. Whole chunks are
retained, so actual answer prompts may be slightly below the cap.

Historical 27B compaction+RAG top-22 was not a strict 32K configuration:
its 102 final inputs range from 25880 to 36475 tokens, with mean 32414.1;
42 exceed 32768. Top-6 and top-12 use smaller inputs, while top-44 and top-66
use larger inputs. Thus old RAG scores must not be described as a controlled
comparison at the same 32K limit. Historical summarization, retrieval, reader
precision, prompt, output limits and judge versions also differ from this
campaign's grading. This campaign aligns the subset and active-budget
target, but does not establish a pure model-size effect from the old scores.

## Artifacts and execution

Output directory:
`benchmark/longmemeval_qwen35_9b_matched102_20260915/`.

- `sample_ids.txt`: exact IDs in archived order.
- `plan.json`: frozen configuration and data identity.
- `reuse_manifest.json`: the 14 copied arms and their original result hashes.
- `historical27b_manifest.json`: old result sources and hashes.
- `*_historical27b.json`: old results selected into the same 102-ID order.
- `preflight.json`: token-transport, text round-trip and capacity checks for all 102.
- `report/paired_results.csv`: per-question answers, judgments and phase timings.

Use `QWEN35_LME_OUTPUT_ROOT` to select this output directory. The default
remains the original 30-question pilot directory. The frozen engine, model
weights, inference algorithm and 9B decoding settings remain unchanged.

The expanded preflight also caught native/HF text-segmentation differences for
Unicode fractions such as `½` and `¾`: the native `digit` classification does
not cover the same Unicode numeric categories as HF. Both encodings recover
the same text, but token IDs can differ. The experiment already sends integer
IDs from the official HF tokenizer directly, so the native text encoder is
not used for these inference inputs. Before permitting this transport, the
runner now checks **every HF token's ID against the model's GGUF vocabulary**
and verifies all merge rules. GGUF's 243 extra `[PAD...]` rows are outside the
HF input vocabulary. If native re-encoding differs, both text round trips and
the complete vocabulary mapping are required and the difference is recorded.
No prompt text, actual inference token IDs, or query offsets are changed.
The original failed preflight and the fraction diagnostic are retained.

```bash
export QWEN35_LME_OUTPUT_ROOT=/home/chaidi/qw3/benchmark/longmemeval_qwen35_9b_matched102_20260915
.venv/bin/python scripts/kvmem_eval/run_qwen35_longmemeval.py
.venv/bin/python scripts/kvmem_eval/score_qwen35_longmemeval.py
.venv/bin/python scripts/kvmem_eval/report_qwen35_longmemeval.py
```

## DeepSeek V4.1 Flash grading, September 15

`score_qwen35_longmemeval_deepseek.py` uses the official endpoint
`https://api.deepseek.com/chat/completions`, model `deepseek-flash`, non-thinking
mode, temperature 0, and at most 16 output tokens. The model alias maps to
V4.1 Flash according to the [official release note](https://api-docs.deepseek.com/news/news260910/).
Actual response model names, request IDs, usage and fingerprints are retained.
The official LongMemEval per-type prompt function is loaded unchanged from the
same source used for local grading. API requests contain the question,
reference/rubric and final answer, with neither policy names nor thinking
traces. A yes/no verdict is accepted only for a normally finished response;
errors, truncation and non-binary outputs do not count as incorrect answers.

The process reads the API key through a hidden terminal prompt or environment
variable. It never writes credentials to source, reports, requests or logs;
proxy use and HTTP redirects are disabled. Four concurrent cloud requests
require no GPU, so the 9B campaign continues normally. `--watch` freezes each
set of newly completed **pairs**, reuses existing hash-verified grades, and
waits for subsequent pairs until all 102 are scored. It does not rerun reader
inference or overwrite the local judge.

```bash
python3 scripts/kvmem_eval/score_qwen35_longmemeval_deepseek.py --prompt-key --watch
```

External artifacts live under `judge_deepseek_v41_flash/`: `status.json`,
`identity.json`, the 12-case `calibration.json`, individual `scores/`, and
immutable `runs/*_N_pairs/` snapshots with paired CSV, selection hashes and
local-judge comparisons. `report/` contains the latest external snapshot.
The experiment root's original `report/` continues to denote local-judge
results. Neither judge's score is an official leaderboard score.

The first external snapshot scored 40 pairs: KVMem 36/40 and compaction+RAG
36/40. On the **same earlier 21 pairs**, external scores are 17/21 and 18/21,
versus local scores 13/21 and 18/21. The four changed verdicts are all KVMem:
three preference questions and one temporal-reasoning question. All prediction
hashes match, so this change is attributable to the judge, not new reader
answers. These interim type proportions are unbalanced; keep the 40-pair
snapshot separate from the 21-pair judge comparison and the eventual full run.
