## 1. Experimental Objective

This set of experiments focuses on two issues in long-context memory scenarios:

1. **Utility**: After historical conversations exceed the normal context window, how much information is lost if we keep only a compact summary?
2. **Cost**: If text retrieval/RAG is used to splice historical fragments back into the prompt, how much accuracy can be recovered, and how many additional input tokens, TTFT, and monetary costs are required?

The final comparison covers three types of methods:

| Method | Input format | Role |
| --- | --- | --- |
| Full Context | Complete history + question | Utility upper bound; also the ideal baseline for cache continuation |
| Compact-only | Compact summary + recent sessions + question | Standard context-compression baseline |
| Compact + RAG | Compact summary + recent sessions + retrieved chunks + question | Traditional “summary + text retrieval” recovery approach |

Core conclusion: **Compact-only significantly loses long-term historical details; RAG can recover part of the information, but it does so by splicing raw text back into the prompt. Therefore, the larger top-k becomes, the higher the cache-miss/prefill cost of the answer prompt.**

## 2. Dataset and Model Setup

The experiments use LongMemEval-S-style tasks. Each sample contains a multi-session history, a query, and a gold answer. The model needs to answer based on the history, and a judge then determines whether the model’s answer is equivalent to the gold answer.

### 2.1 Sample Selection

| Item | Setting |
| --- | ---: |
| Total samples | 102 |
| Number of question types | 6 |
| Samples per type | 17 |

| question_type | Number of samples | Data description |
| --- | --- | --- |
| single-session-user | 17 | The answer mainly comes from information mentioned by the user in one historical session. This tests whether the model can locate user-side factual details in a long history. |
| single-session-assistant | 17 | The answer mainly comes from a response previously given by the assistant in one historical session. This tests whether the model can remember or retrieve content it previously generated. |
| single-session-preference | 17 | The answer comes from user preferences reflected in a single session, such as likes, habits, and selection tendencies. This tests the model’s ability to extract preference information. |
| multi-session | 17 | The question requires combining information from multiple historical sessions. This tests cross-session integration and long-term memory ability. |
| temporal-reasoning | 17 | The question requires reasoning over historical information at different time points, such as order, the most recent state, or time-related changes. This tests temporal reasoning ability. |
| knowledge-update | 17 | The history contains information updates. The model must use newer information to override older information. This tests whether the model can handle knowledge changes and avoid interference from outdated information. |
| Total | 102 |  |

### 2.2 Length Statistics

| Metric | Mean | Median | Max |
| --- | ---: | ---: | ---: |
| query tokens | 18.89 | 16 | 52 |
| sessions per sample | 47.98 | 48 | 62 |
| full history tokens | 109,375.03 | 109,551.5 | 113,270 |
| tokens per session | 2,279.58 | 2,250.5 | 19,135 |

### 2.3 Model and Evaluation

| Item | Setting |
| --- | --- |
| generation model | `qwen3.6-27b-fp8`, local vLLM service |
| temperature | 0.0 |
| enable_thinking | true |
| tokenizer | `motivation/models/tokenizers/Qwen3.6-27B-FP8` |
| judge | DeepSeek V4 Pro judges whether the model answer is equivalent to the gold answer |
| metrics | correct / accuracy, average input, TTFT, first content, end-to-end latency, cost |

Note: When `enable_thinking=true`, `TTFT` is the first streaming delta, which is usually a reasoning token. The first user-visible token is closer to `first_content_sec`.

## 3. Method Details

### 3.1 Full Context

Full Context directly feeds the complete history into the model:

```text
System instruction
Question date
Full history sessions
Question
Answer
```

It is used to measure the accuracy upper bound when the complete history is visible. In the cost analysis, we assume that the complete history is already in the KV/cache. When answering the next turn, the old history is charged as a cache hit, and the new query is charged as a cache miss.

### 3.2 Compact-only

Compact simulates context-checkpoint compression. In the experiment, we simulate a 100K context window:

1. Find the session boundary in the history that is closest to 100K tokens.
2. Feed this earlier part of the history into the model and generate a compact summary.
3. Append the recent/uncompressed sessions after the 100K boundary.
4. Finally answer the question using the summary + recent sessions.

Compact prompt:

```text
You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task.

Include:
- Current progress and key decisions made
- Important context, constraints, or user preferences
- What remains to be done (clear next steps)
- Any critical data, examples, or references needed to continue

Be concise, structured, and focused on helping the next LLM seamlessly continue the work.
```

The statistics for compact summary generation are shown below. These use 102 measured MTP records. `summary body tokens` refers to the compact summary that is ultimately written into the subsequent prompt; `thinking / reasoning tokens` refers to the reasoning output generated before the summary when `enable_thinking=true`; `summary total completion tokens` is the sum of the two.

| Metric | Mean | Median | Max |
| --- | ---: | ---: | ---: |
| summary body tokens | 895.46 | 904 | 1,518 |
| thinking / reasoning tokens | 3,061.46 | 2,835.5 | 6,765 |
| summary total completion tokens | 3,956.92 | 3,840.5 | 7,690 |
| summary generation input tokens | 98,503.37 | 98,578.5 | 100,071 |

The statistics for the recent/uncompressed sessions retained after compaction are shown below:

| Metric | Mean | Median | Max |
| --- | ---: | ---: | ---: |
| retained recent sessions | 4.76 | 5 | 8 |
| retained recent session tokens | 11,079.84 | 11,045.5 | 14,513 |

Compact summary generation latency is counted separately as the “compaction trigger overhead” and can be analyzed together with subsequent TTFT/end-to-end latency. Across the 102 measured MTP samples, the average TTFT of the summary generation request is **2.83s/sample**, the average time to first body token is **150.30s/sample**, the average decode time is **184.99s/sample**, and the average full summary generation time is **187.82s/sample**.

The total time for complete summary generation is **19157.4s = 5.32h**, and the average time is 19157.4 / 102 = **187.82s/sample**.

### 3.3 Compact + RAG

RAG does not replace the summary. Instead, it supplements Compact-only by adding raw historical fragments back in:

```text
compact_summary_time = 19157.4 / 102 = 187.82s / sample
```

Final Compact-only prompt:

```text
System instruction
Question date
Compact summary
Recent / uncompressed sessions
Question
Answer
```

RAG implementation parameters:

| Parameter | Setting |
| --- | --- |
| splitting method | Split by session first, then split chunks within each session |
| chunk_size | 1024 tokens |
| chunk_overlap | 128 tokens |
| stride | 896 tokens |
| embedding model | `all-MiniLM-L6-v2` |
| similarity | Dot product / cosine after normalization |
| query | Current question |
| top_k | 6 / 12 / 22 / 44 / 66 |

The rationale for the k values:

```text
k=6  : Original main setting; theoretically covers 1024 + 5 × 896 = 5504 tokens
k=22 : Main expansion point needed to cover the longest session of about 19135 tokens
k=44/66 : Upper-bound probes with larger retrieval budgets
```

Note: The theoretical coverage length only indicates the “injectable context size.” The actual top-k chunks are selected by similarity, so they are not guaranteed to come from the same session or to be contiguous.

### 3.4 Hardware Used in the Experiments

Hardware data is divided into two parts:

**Remote vLLM server**

| Item | Data |
| --- | --- |
| Hostname | cs-8def1-95c25-server |
| CPU | Intel Xeon Gold 6334 @ 3.60GHz |
| CPU cores | 16 physical / 32 logical |
| CPU frequency | base 3.60GHz, max 3.70GHz |
| Memory | 503 GiB |
| GPU | NVIDIA A40 |
| GPU Memory | 49,140 MiB |
| Driver | 580.65.06 |

**Local RAG retrieval machine**

| Item | Data |
| --- | --- |
| OS | Windows 11 |
| CPU | AMD Ryzen 9 7945HX with Radeon Graphics |
| CPU cores | 16 physical / 32 logical |
| CPU frequency | MaxClockSpeed 2501 MHz |
| Memory | 16,844,877,824 bytes, about 15.69 GiB |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| GPU Memory | 4,293,918,720 bytes, about 4.00 GiB |
| Driver | 32.0.15.9636 |

## 4. Experimental Results

### 4.1 Accuracy Changes

| setting | correct | accuracy | Gap from Full |
| --- | ---: | ---: | ---: |
| Full Context | 83/102 | 81.37% | 0 |
| Compact-only | 22/102 | 21.57% | -61 |
| RAG k=6 | 64/102 | 62.75% | -19 |
| RAG k=12 | 71/102 | 69.61% | -12 |
| RAG k=22 | 75/102 | 73.53% | -8 |
| RAG k=44 | 74/102 | 72.55% | -9 |
| RAG k=66 | 75/102 | 73.53% | -8 |

The accuracy trend can be understood in three steps. First, Compact-only drops from 83/102 to 22/102, showing that ordinary summaries severely lose long-term historical details. Second, RAG k=6 improves to 64/102, accounting for the main gain, which shows that information lost by the summary can indeed be recovered through raw historical fragments. Third, after k=22 improves to 75/102, performance essentially saturates; k=44 and k=66 do not produce stable additional improvements.

By question type, the gains from RAG are uneven:

| question_type | Full | Compact-only | k=6 | k=12 | k=22 | k=44 | k=66 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| single-session-assistant | 17/17 | 3/17 | 17/17 | 17/17 | 17/17 | 17/17 | 17/17 |
| multi-session | 13/17 | 0/17 | 10/17 | 10/17 | 13/17 | 13/17 | 13/17 |
| temporal-reasoning | 14/17 | 2/17 | 9/17 | 10/17 | 12/17 | 14/17 | 13/17 |
| knowledge-update | 15/17 | 9/17 | 12/17 | 13/17 | 15/17 | 13/17 | 14/17 |
| single-session-user | 16/17 | 4/17 | 11/17 | 12/17 | 12/17 | 12/17 | 12/17 |
| single-session-preference | 8/17 | 4/17 | 5/17 | 9/17 | 6/17 | 5/17 | 6/17 |
| Total | 83/102 | 22/102 | 64/102 | 71/102 | 75/102 | 74/102 | 75/102 |

The most systematic observations are:

1. **single-session-assistant has no problem at all**
   All k values achieve 17/17, indicating that RAG handles this type of question very easily.
2. **The gains from k=6 to k=22 mainly come from multi-session and temporal-reasoning**
   multi-session improves from 10/17 to 13/17, and temporal-reasoning improves from 9/17 to 12/17. This indicates that expanding the context can indeed recover part of the evidence needed for cross-session and temporal reasoning.
3. **single-session-user remains a stubborn gap**
   Full-context reaches 16/17, but k22/k44/k66 all stop at 12/17. In many of these error cases, the answer session has already been retrieved, such as 118b2229, 58ef2f1c, and 5d3d2817, yet the model still answers “information not available.” This shows that the problem is not only top-k coverage, but also the position of evidence within retrieved fragments, noise, and the model’s ability to use the evidence.
4. **preference questions are the hardest overall**
   Full-context itself only reaches 8/17, and RAG reaches only 5–6/17. These questions require inferring user preferences from context rather than extracting facts, so increasing top-k helps only marginally.
5. **knowledge-update is sensitive to more context**
   It reaches 15/17 at k22, but regresses at k44/k66. A typical pattern is that both old and new information are included, and the model selects the old answer, such as in examples involving Rachel’s relocation, mortgage amount, and recent family trip.

single-session-assistant reaches 100% under RAG, likely because this category is naturally more suitable for retrieval: the questions contain clear topic words, entity names, or keywords from previous assistant responses, so the answer session is hit in the top 6 and always ranked first. In contrast, single-session-user questions are more often generalized personal-memory queries. The answer is often hidden in short user self-statements, and the surface semantic overlap between the query and evidence text is weaker, resulting in lower top-6 retrieval hit rate and final accuracy.

### 4.2 Time Cost

This section recomputes time cost under the **warm-cache / cache-continuation** perspective. LongMemEval-S historical sessions simulate long conversations that have already occurred, so before the latest query arrives, the historical content should theoretically already be in the model’s KV/cache. To better approximate this real scenario, this supplementary experiment enables vLLM prefix cache and serially executes a warm-up request and an official request within each sample. We check `cache_hit_tokens` to verify whether the historical prefix hits the cache as expected.

In this section, TTFT is uniformly defined as the wall-clock time between sending the official answer request and the model outputting the first thinking/reasoning token. Since `enable_thinking=true` in the experiments, the first token is usually a reasoning token rather than a user-visible body token.

#### Time Measurement Procedure

**Full Context** first sends `system prompt + question date + complete historical sessions` as a warm-up request, so that the complete history enters the prefix cache. In the official test, it sends `system prompt + question date + complete historical sessions + final question`, and measures the time from sending the official request to outputting the first thinking token. This time represents the waiting time to answer the final question when the complete history is already cached.

**Compact-only** has two stages. The first stage is summary generation: first warm up `system prompt + question date + first 100K historical sessions`, then send `system prompt + question date + first 100K historical sessions + compact instruction`, and measure the time from sending the compact request to the complete summary output. The second stage is final answering: first warm up `system prompt + question date + compact summary + recent sessions`, then send `system prompt + question date + compact summary + recent sessions + final question`, and measure the time from sending the official answer request to outputting the first thinking token. The total input-side latency of Compact-only consists of summary generation time and final-answer waiting time.

**Compact + RAG** adds a retrieval stage on top of Compact-only. After summary generation and compacted-context warm-up are complete, the final question is used to retrieve top-k historical fragments, and the latency of query embedding, vector search, top-k chunk reading, and prompt construction is measured. It then sends `system prompt + question date + compact summary + recent sessions + retrieved chunks + final question`, and measures the time from sending the official answer request to outputting the first thinking token. The total input-side latency of Compact + RAG consists of summary generation time, retrieval time, and final-answer waiting time.

#### Conclusion 1: Measured MTP Results

This supplementary experiment runs on the remote vLLM MTP configuration. It finally uses the fastest stable measured setting, `num_speculative_tokens=4`, while enabling prefix cache, setting `max_model_len=131072`, and using a CUDA graph capture size of 5. The experiment reruns only Compact-only and Compact + RAG, not Full Context, because Full Context in this section only involves the final TTFT of cached continuation and does not involve the summary-output stage.

A total of 102 samples are measured in this round, with 17 samples for each question_type. Compact summaries are generated completely. The results show that for long-context compression tasks with about 99K input tokens and complete summary output, summary output decoding is the main bottleneck: each summary outputs 3,957 tokens on average, and summary output decoding takes 184.990s on average, or about 21.39 tok/s. This speed is lower than the 50+ tok/s observed in short-context probes, indicating that prefix cache mainly reduces cached prefill/TTFT, while long-context decoding still needs to continue generation over KV close to 100K tokens.

RAG retrieval itself averages about 2.14–2.19s and changes little as k increases. The main additional cost comes from the increase in final local compute tokens after retrieved chunks are spliced back into the final answer prompt, which raises final cached-input TTFT. At k=66, total latency increases to 1.300x on average and 1.294x at the median relative to Compact-only.

##### Token Hit Statistics

| Method | Summary input tok | Summary cache hit tok | Summary output tok | RAG retrieved tok | Final input tok | Final cache hit tok | Final local compute tok | Final cache hit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full Context | - | - | - | - | 109,137 | 108,788 | 350 | 99.68% |
| Compact-only | 98,503 | 97,208 | 3,957 | 0 | 12,078 | 10,848 | 1,230 | 89.82% |
| RAG k=6 | 98,503 | 97,208 | 3,957 | 5,681 | 17,768 | 10,848 | 6,920 | 61.05% |
| RAG k=12 | 98,503 | 97,208 | 3,957 | 11,256 | 23,344 | 10,848 | 12,496 | 46.47% |
| RAG k=22 | 98,503 | 97,208 | 3,957 | 20,481 | 32,569 | 10,848 | 21,721 | 33.31% |
| RAG k=44 | 98,503 | 97,208 | 3,957 | 40,860 | 52,948 | 10,848 | 42,100 | 20.49% |
| RAG k=66 | 98,503 | 97,208 | 3,957 | 61,031 | 73,119 | 10,848 | 62,271 | 14.84% |

##### Time Statistics: Mean

| Method | Summary cached input TTFT | Summary output decode | Retrieval | Final cached input TTFT | Total | Relative to Compact-only |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Full Context | - | - | - | 1.849 | 1.849 | - |
| Compact-only | 2.826 | 184.990 | 0.000 | 1.517 | 189.333 | 1.00x / 100.0% |
| RAG k=6 | 2.826 | 184.990 | 2.193 | 5.825 | 195.833 | 1.03x / 103.4% |
| RAG k=12 | 2.826 | 184.990 | 2.137 | 10.086 | 200.038 | 1.06x / 105.7% |
| RAG k=22 | 2.826 | 184.990 | 2.152 | 17.617 | 207.584 | 1.10x / 109.6% |
| RAG k=44 | 2.826 | 184.990 | 2.137 | 35.820 | 225.773 | 1.19x / 119.2% |
| RAG k=66 | 2.826 | 184.990 | 2.155 | 56.104 | 246.074 | 1.30x / 130.0% |

##### Time Statistics: Median

| Method | Summary cached input TTFT | Summary output decode | Retrieval | Final cached input TTFT | Total | Relative to Compact-only |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Full Context | - | - | - | 1.675 | 1.675 | - |
| Compact-only | 2.775 | 183.852 | 0.000 | 1.495 | 188.200 | 1.00x / 100.0% |
| RAG k=6 | 2.775 | 183.852 | 2.162 | 5.808 | 194.967 | 1.04x / 103.6% |
| RAG k=12 | 2.775 | 183.852 | 2.098 | 10.090 | 199.054 | 1.06x / 105.8% |
| RAG k=22 | 2.775 | 183.852 | 2.101 | 17.653 | 206.182 | 1.10x / 109.6% |
| RAG k=44 | 2.775 | 183.852 | 2.098 | 35.834 | 224.365 | 1.19x / 119.2% |
| RAG k=66 | 2.775 | 183.852 | 2.107 | 56.383 | 243.477 | 1.29x / 129.4% |

#### Conclusion 2: Idealized Estimate at 55.3 tok/s

Based on the 102 measured MTP records above, Compact-only and RAG share the same compact summary generation. The average compact-summary request input per sample is 98,503 tokens, of which 97,208 tokens hit the prefix cache; the summary cached-input TTFT is 2.826s. The summary output is 3,957 tokens, and the current measured summary output decode time is 184.990s.

If only the summary-output stage is replaced with the currently best observed post-TTFT decode speed of 55.3 tok/s, then:

```text
summary output decode = 3,956.92 / 55.3 = 71.55s
compact summary generation = 2.826 + 71.55 = 74.38s / sample
```

That is, under the measurement setup of these 102 MTP records, the ideal compact-output decode time is about 71.6s/sample. If the compact request’s own TTFT is also included, compact summary generation is about 74.4s/sample. For all 102 samples, total summary-output decode time is about 7,299s, or 2.03h; including TTFT, compact summary generation totals about 7,587s, or 2.11h.

##### Token Hit Statistics

| Method | Summary input tok | Summary cache hit tok | Summary output tok | RAG retrieved tok | Final input tok | Final cache hit tok | Final local compute tok | Final cache hit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full Context | - | - | - | - | 109,137 | 108,788 | 350 | 99.68% |
| Compact-only | 98,503 | 97,208 | 3,957 | 0 | 12,078 | 10,848 | 1,230 | 89.82% |
| RAG k=6 | 98,503 | 97,208 | 3,957 | 5,681 | 17,768 | 10,848 | 6,920 | 61.05% |
| RAG k=12 | 98,503 | 97,208 | 3,957 | 11,256 | 23,344 | 10,848 | 12,496 | 46.47% |
| RAG k=22 | 98,503 | 97,208 | 3,957 | 20,481 | 32,569 | 10,848 | 21,721 | 33.31% |
| RAG k=44 | 98,503 | 97,208 | 3,957 | 40,860 | 52,948 | 10,848 | 42,100 | 20.49% |
| RAG k=66 | 98,503 | 97,208 | 3,957 | 61,031 | 73,119 | 10,848 | 62,271 | 14.84% |

##### Time Statistics: Mean

| Method | Summary cached input TTFT | Summary output decode | Retrieval | Final cached input TTFT | Total | Relative to Compact-only |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Full Context | - | - | - | 1.849 | 1.849 | - |
| Compact-only | 2.826 | 71.554 | 0.000 | 1.517 | 75.897 | 1.00x / 100.0% |
| RAG k=6 | 2.826 | 71.554 | 2.193 | 5.825 | 82.397 | 1.09x / 108.6% |
| RAG k=12 | 2.826 | 71.554 | 2.137 | 10.086 | 86.602 | 1.14x / 114.1% |
| RAG k=22 | 2.826 | 71.554 | 2.152 | 17.617 | 94.148 | 1.24x / 124.0% |
| RAG k=44 | 2.826 | 71.554 | 2.137 | 35.820 | 112.337 | 1.48x / 148.0% |
| RAG k=66 | 2.826 | 71.554 | 2.155 | 56.104 | 132.639 | 1.75x / 174.8% |

In the mean table, only summary output decode is replaced; all other retrieval and final cached-input TTFT values are kept from the 102 measured MTP records. After replacement, the Compact-only total latency drops from 189.333s to 75.897s. Since the shared cost of summary generation is reduced, the relative share of RAG’s extra prefill becomes more obvious: k=66 changes from 1.30x in measured results to 1.75x in the idealized estimate.

##### Time Statistics: Median

The median is estimated using the median summary-output tokens from the 102 measured MTP records: 3,840.5 tokens / 55.3 = 69.448s. The remaining retrieval and final cached-input TTFT values use the measured medians from the 102 MTP records.

| Method | Summary cached input TTFT | Summary output decode | Retrieval | Final cached input TTFT | Total | Relative to Compact-only |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Full Context | - | - | - | 1.675 | 1.675 | - |
| Compact-only | 2.775 | 69.448 | 0.000 | 1.495 | 73.718 | 1.00x / 100.0% |
| RAG k=6 | 2.775 | 69.448 | 2.162 | 5.808 | 80.194 | 1.09x / 108.8% |
| RAG k=12 | 2.775 | 69.448 | 2.098 | 10.090 | 84.412 | 1.15x / 114.5% |
| RAG k=22 | 2.775 | 69.448 | 2.101 | 17.653 | 91.977 | 1.25x / 124.8% |
| RAG k=44 | 2.775 | 69.448 | 2.098 | 35.834 | 110.156 | 1.49x / 149.4% |
| RAG k=66 | 2.775 | 69.448 | 2.107 | 56.383 | 130.713 | 1.77x / 177.3% |

Compared with the measured summary output decode time of 184.990s/sample in the 102 MTP records, the ideal output speed of 55.3 tok/s accelerates the compact-output stage by about 2.59x and saves about 113.4s per sample. Since Compact-only and all RAG k settings share this summary generation, this optimization reduces the total latency for all groups, but it does not change the trend that final cached-input TTFT increases as RAG k increases.

### 4.3 Price Cost

Price is counted only on the input side and follows the same warm-cache/cache-continuation measurement setup as Section 4.2: input tokens are charged separately as `cache hit tokens` and `local compute tokens`. In other words, Full Context, Compact summary generation, and Compact/RAG final answering are not charged as if the entire prompt were cache misses. Instead, the reusable prefix is charged as cache hit, and newly added or non-reusable parts are charged as cache miss.

| Input component | Billing method | Reason |
| --- | --- | --- |
| Historical prefix in Full Context cached continuation | cache hit | After warm-up, the complete historical prefix has entered the prefix cache |
| Final question and a small number of newly added formatting tokens in Full Context | cache miss / local compute | Newly added relative to the warm-up request |
| Historical prefix in Compact summary generation | cache hit | The early history before compaction enters the prefix cache after warm-up |
| Compact instruction and other newly added parts in Compact summary generation | cache miss / local compute | Newly added relative to the warm-up request |
| Summary + recent prefix in Compact/RAG final answer | cache hit | Reusable after final pre-warm |
| Final question, top-k labels, retrieved chunks, and other newly added parts in Compact/RAG final answer | cache miss / local compute | Newly added relative to final pre-warm and increases with k |

The cost that truly grows with RAG k mainly comes from the retrieved chunks newly added in the final answer stage. They enter the local compute/cache-miss portion.

According to Qwen and DeepSeek official documentation queried on June 17, 2026, the API prices are as follows:

| Pricing setup | cache miss | cache hit |
| --- | ---: | ---: |
| Qwen experimental setup | ¥3 / 1M | ¥0.25 / 1M |
| DeepSeek V4 Pro setup | ¥3 / 1M | ¥0.025 / 1M |

The cumulative token counts across 102 samples used in the calculation are as follows:

| setting | cache hit tokens | cache miss / local compute tokens |
| --- | ---: | ---: |
| Full cached | 11,096,376 | 35,700 |
| Compact-only | 11,021,712 | 257,584 |
| RAG k=6 | 11,021,712 | 837,946 |
| RAG k=12 | 11,021,712 | 1,406,732 |
| RAG k=22 | 11,021,712 | 2,347,716 |
| RAG k=44 | 11,021,712 | 4,426,366 |
| RAG k=66 | 11,021,712 | 6,483,808 |

The Qwen calculation is:

```text
input cost = cache hit tokens × 0.25 / 1M + cache miss tokens × 3 / 1M
```

Qwen setup:

| setting | input cost | vs Full |
| --- | --- | --- |
| Full cached | ¥2.8812 | 1.00x |
| Compact-only | ¥3.5282 | 1.22x |
| RAG k=6 | ¥5.2693 | 1.83x |
| RAG k=12 | ¥6.9756 | 2.42x |
| RAG k=22 | ¥9.7986 | 3.40x |
| RAG k=44 | ¥16.0345 | 5.57x |
| RAG k=66 | ¥22.2069 | 7.71x |

The DeepSeek V4 Pro calculation is:

```text
input cost = cache hit tokens × 0.025 / 1M + cache miss tokens × 3 / 1M
```

DeepSeek V4 Pro setup:

| setting | input cost | vs Full |
| --- | --- | --- |
| Full cached | ¥0.3845 | 1.00x |
| Compact-only | ¥1.0483 | 2.73x |
| RAG k=6 | ¥2.7894 | 7.25x |
| RAG k=12 | ¥4.4957 | 11.69x |
| RAG k=22 | ¥7.3187 | 19.03x |
| RAG k=44 | ¥13.5546 | 35.25x |
| RAG k=66 | ¥19.7270 | 51.30x |

The key point about price cost is that as top-k increases, retrieved chunks continuously increase local compute/cache-miss tokens, so RAG k=66 is significantly more expensive than Full cached continuation. Under the DeepSeek V4 Pro setup, cache hits are cheaper, so the absolute cost of Full cached is lower, and the relative impact of the additional miss tokens introduced by RAG is larger.
