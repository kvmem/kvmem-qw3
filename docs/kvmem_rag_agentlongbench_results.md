# KVMem 与 RAG：AgentLongBench 52 条诊断样本实验记录

更新日期：2026-07-19

本文记录 KVMem 与 RAG 在 AgentLongBench 52 条冻结诊断样本上的结果、
retrieval 覆盖分析，以及“相同 KVMem selected window 重新 dense prefill”对照实验。
目标是为后续论文中的实验结果解读保留可复查的数字、原始产物和结论边界。

## 1. 数据范围与解释边界

这 52 条来自 AgentLongBench 后 250 条长上下文样本，但**不是随机抽样**。
筛选条件是：KVMem 32K 与 64K 对同一题的官方评分相同，但与旧版
Compact+RAG（1024-token block、overlap 128、top-30）的官方评分不同。
因此它们是用于研究 KVMem/RAG 差异原因的诊断子集。

- 样本数：52。
- target length：128K 为 21 条，256K 为 31 条。
- actual length：128K 为 36 条，256K 为 16 条。
- 下文的 `40/52` 与 `29/52` 可以用于该诊断子集内的方法比较和机制分析，
  但不能直接表述为 AgentLongBench 全集上的无偏准确率提升。

另一个数据限制是：32-token RAG delivery 保存了逐样本召回块和按任务汇总的
正确数，但没有逐样本答案与评分。因此可以准确报告 RAG 总体/任务准确数，
也可以逐题比较 retrieval 覆盖；但无法从该 delivery 精确恢复
“KVMem 正确而最佳 RAG 错误”的具体样本 ID。根据两者按任务的正确数，
这类样本的严格数量范围是 13--21 条。

## 2. 实验配置

### 2.1 KVMem

| 参数 | 设置 |
| --- | --- |
| 模型 | Qwen3.6-27B-Q8_0 |
| KVMem budget | 32,768 tokens |
| generation budget | 32,768 tokens |
| block size | 32 tokens |
| retrieval | mean-k |
| sink blocks | 8 |
| recent blocks | 0 |
| sub-block | 关闭 |
| update mode | step |
| query conditioned | 开启，显式 final-query span |
| thinking budget | 4,096 tokens |
| sampling | temperature 0.6，top-p 0.95 |

KVMem 的 32K 是最终 active window 的预算，包含 chat framing、任务说明、
selected history、最终问题和 assistant generation prefix；query 不是在 32K
之外再次附加的。

### 2.2 RAG block=32 消融

| 参数 | 设置 |
| --- | --- |
| tokenizer | Qwen tokenizer |
| embedding | `jina-embeddings-v2-small-en` |
| block size | 32 tokens |
| top-k | 960 blocks，即 30,720 chunk tokens |
| overlap | 0、4、8 |
| block step | 32、28、24 |
| 输入顺序 | retrieval 后按原始历史位置升序重排 |

RAG 的 30,720 tokens 是 history chunk token 总量，任务说明和最终问题另行加入；
KVMem 的 32,768 tokens 是整个 active window。两者规模接近，但预算口径并非
逐 token 完全相同，论文中不应把它描述为严格相等的 context token 数。

## 3. 最终回答准确率

### 3.1 总体结果

| 方法 | 正确数 | Exact 准确率 | 相对最佳 RAG 的变化 |
| --- | ---: | ---: | ---: |
| RAG，overlap=0 | 24/52 | 46.15% | -16 条 / -30.77 pp |
| RAG，overlap=4 | 28/52 | 53.85% | -12 条 / -23.08 pp |
| RAG，overlap=8 | 29/52 | 55.77% | -11 条 / -21.15 pp |
| **KVMem 32K** | **40/52** | **76.92%** | 基准 |

在该诊断子集上，最佳 RAG 是 overlap=8，KVMem 比它多答对 11 条，提升
21.15 个百分点。最佳 RAG 错 23 条，KVMem 错 12 条，对应错误数减少
`(23 - 12) / 23 = 47.83%`。

RAG overlap 从 0 增加到 4 时增加 4 条正确答案（+7.70 pp），从 4 增加到 8
只再增加 1 条（+1.92 pp）；从 0 到 8 总计增加 5 条（+9.62 pp）。

### 3.2 按任务类型

下表使用效果最好的 RAG overlap=8。

| 任务类型 | 样本数 | KVMem 正确 | RAG 正确 | KVMem - RAG |
| --- | ---: | ---: | ---: | ---: |
| Count Correctness (Env) | 5 | 5 | 4 | +1 |
| Count Frequency (Env) | 10 | 10 | 2 | +8 |
| Count Frequency (Tool) | 10 | 6 | 4 | +2 |
| Find Duplicates (Tool) | 10 | 6 | 7 | -1 |
| Find Round with Largest Value (Env) | 9 | 9 | 9 | 0 |
| Find Target Offsets (Tool) | 2 | 0 | 1 | -1 |
| Intersection | 3 | 1 | 1 | 0 |
| Weighted Summation (Env) | 3 | 3 | 1 | +2 |
| **合计** | **52** | **40** | **29** | **+11** |

最大的差异来自 `Count Frequency (Env)`：KVMem 为 10/10，RAG 为 2/10，
单一任务贡献了总体 11 条净优势中的 8 条。KVMem 并非在所有任务上都更好：
RAG 在 `Find Duplicates (Tool)` 和 `Find Target Offsets (Tool)` 各多答对 1 条。

## 4. Retrieval 集合差异

### 4.1 全部 52 条的 source-text overlap

KVMem internal block ID 相对于完整 chat prompt 定义，RAG 的
`mflat_cXXXXX` 相对于 flattened history 定义，二者不能直接比较 ID。
分析先把 KVMem selected token 映射到统一 logical message-text 坐标，再与
RAG 的精确 token span 比较。

| RAG overlap | logical-char Jaccard | RAG 被 KVMem 覆盖 | KVMem 被 RAG 覆盖 | projected same-ID Jaccard（覆盖≥50%） | Jaccard <25% | Jaccard ≥60% |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 22.08% | 40.56% | 31.82% | 21.19% | 35/52 | 0/52 |
| 4 | 22.99% | 42.50% | 32.40% | 21.13% | 34/52 | 0/52 |
| 8 | 23.76% | 44.57% | 32.63% | 20.75% | 31/52 | 0/52 |

即使使用效果最好的 overlap=8，两种方法选中 source text 的平均 Jaccard 也只有
23.76%；31/52 低于 25%，没有一条达到 60%。这直接说明 KVMem 与 RAG 大多
没有选择相同的证据集合，最终答案差异不能只归因于“相同文本使用 KV cache
还是重新 prefill”。

KVMem 每条样本平均覆盖约 82,243 个去重 logical characters；RAG overlap=0
约为 64,972，overlap=8 约为 60,701。overlap=8 相比 overlap=0 的去重覆盖
减少 4,271 characters（6.57%），但准确率增加 9.62 pp。这说明在该设置下，
相邻覆盖和局部连续性比最大化文本覆盖广度更重要；重叠 chunk 本身不应被当作
实现错误。

### 4.2 旧版逐样本 RAG outcome 分组（辅助证据）

32-token delivery 没有逐样本评分。作为辅助分析，可以使用旧版
1024-token Compact+RAG 的逐样本 outcome：在 27 条“KVMem 正确、旧版 RAG
错误”的样本中，overlap=8 的 retrieval Jaccard 为 21.44%，其中 20/27 低于
25%；两者都正确的 13 条中，平均 Jaccard 为 29.50%。这个趋势支持
retrieval-set divergence 是重要原因，但这里的 outcome 来自旧版 RAG，不能
冒充 32-token RAG 的逐样本结果。

## 5. 基于参考答案的局部事实覆盖

在 KVMem Exact 正确的 40 条样本中，依据问题结构和参考答案定位所需局部事实，
再检查 KVMem 与最佳 RAG（block=32、overlap=8）的覆盖，得到：

| 归因类别 | 样本数 | 含义 |
| --- | ---: | --- |
| 直接答案事实缺失候选 | 9 | KVMem 覆盖答案所需正向事实，RAG 漏掉至少一个 |
| 完整证明/负证据不充分 | 4 | RAG 有部分局部证据，但不足以形成完整判断 |
| 两者覆盖相同局部证据 | 25 | 不能用局部 retrieval 缺失解释潜在答案差异 |
| 非局部或无法可靠定位 | 2 | 当前规则不能作可信局部归因 |

9 条直接事实缺失候选分为三种模式，每种 3 条：

1. exhaustive count coverage：全局计数集合少召回一个 occurrence；
2. needle in homogeneous tool list：在高度同质化的超长工具列表中漏掉目标；
3. round-conditioned fact fragment：漏掉指定轮次中的 category/feedback 片段。

这些事实缺口提供了最直接的 retrieval 失败证据。例如：

- `38_a_4254`：参考计数 27；KVMem 覆盖 27/27，RAG 覆盖 26/27；
- `32_a_6729`：参考计数 5；KVMem 覆盖 5/5，RAG 覆盖 4/5；
- `44_r19_a_0565`：KVMem 召回 round 19 中唯一的 `0565`，RAG 未召回；
- `40_i8_j13_a_0953`：KVMem 覆盖 round 8 和 round 13，RAG 漏掉 round 13；
- `1_r102_b`：KVMem 覆盖 round 102 的 6 个 category lines，RAG 完整覆盖 2/6。

9 条直接缺口涉及的 KVMem evidence block 中，最差的 mean-k 非 sink 排名为
第 846 名，全部都在 KVMem 的 1024-block budget 内，也没有一个排到第 960 名
以后。因此 KVMem 的优势不能简单解释成“1024 比 RAG top-960 多了 64 个块”；
更可能来自不同 scoring representation、query/block 对齐方式和 chunk 边界。

重要限制：由于最佳 RAG 没有逐样本评分，这 9 条应称为“直接 retrieval gap
候选”，不能直接写成 9 条已证实的 `KVMem correct / RAG wrong` 样本。

## 6. 相同 selected window 的 dense replay（模式 B）

为了区分 retrieval 内容与 contextualized KV 表示，将 KVMem dump 中最终选中的
全部 token 按原 prompt 位置排序，解码为 raw text，然后关闭 KVMem，在普通
dense attention 中重新 prefill。replay 文本包含任务说明、chat framing、selected
history、最终问题和 assistant prefix；query 已在 selected window 内，只保留一次。

当前默认 KVMem 会把选中块的 K re-RoPE 到连续 sparse-window 位置；它不是保留
original position 的实验模式。因此 KVMem 与 dense replay 都使用压缩后的连续
窗口。主要变量是：

- KVMem 复用在原完整左前缀中计算出的 contextualized K/V；
- dense replay 只在压缩后的 selected text 上重新计算 hidden states 和 K/V。

### 6.1 全部 52 条结果

| 方法 | 正确数 | Exact 准确率 |
| --- | ---: | ---: |
| KVMem 32K | 40/52 | 76.92% |
| selected-text dense replay | 38/52 | 73.08% |

| 逐题关系 | 样本数 |
| --- | ---: |
| 两者都正确 | 35 |
| 两者都错误 | 9 |
| 仅 KVMem 正确 | 5 |
| 仅 dense replay 正确 | 3 |

- 逐题一致：44/52（84.62%）。
- KVMem 正确样本被 replay 保持：35/40（87.50%）。
- KVMem 错误样本被 replay 修复：3/12（25.00%）。
- KVMem 净多 2 条，即 +3.85 pp。
- discordant pair 为 5 对 3，McNemar exact two-sided `p=0.7266`；当前样本量
  不能证明 contextualized KV 带来统计显著的总体准确率提升。

按任务看，所有 8 条翻转都集中在超长工具列表任务：

| 任务 | 样本数 | KVMem | dense replay |
| --- | ---: | ---: | ---: |
| Count Correctness (Env) | 5 | 5 | 5 |
| Count Frequency (Env) | 10 | 10 | 10 |
| Count Frequency (Tool) | 10 | 6 | 4 |
| Find Duplicates (Tool) | 10 | 6 | 6 |
| Find Round with Largest Value (Env) | 9 | 9 | 9 |
| Find Target Offsets (Tool) | 2 | 0 | 0 |
| Intersection | 3 | 1 | 1 |
| Weighted Summation (Env) | 3 | 3 | 3 |

5 条仅 KVMem 正确的样本全部要求找到正向工具列表事实：
`Shiftry`（2 vs replay 0）、`Iron Leaves`（4 vs 3）、`Clauncher`
（true vs false）、`1010`（4 vs 3）、`0953`（true vs false）。这与
contextualized KV 帮助保留碎片所属轮次/工具调用关系的假设一致。

作为进一步的相关性观察，仅 KVMem 正确的 5 条平均被切成 243.8 个 selected
runs，而两者都正确的 35 条平均为 144.8 个 runs。更严重的碎片化更容易使纯文本
replay 丢失轮次与列表归属；但 task-type 是混杂变量，这个数字不是独立因果证明。

### 6.2 9 条直接 retrieval-gap 候选

前述 9 条直接答案事实缺口候选单独 replay 了两次，均为 9/9 正确。这个结果说明：
对于已经定位到 KVMem 召回而 RAG 漏掉的正向事实，只要把 KVMem selected text
提供给模型，普通 dense prefill 也足以得到正确答案；这些样本不需要依赖原
contextualized KV 才能回答。

### 6.3 Replay 输入审计与限制

- 每条原始 selected token 数：32,737--32,768。
- 解码再分词后的 dense input：32,726--32,763 tokens。
- 平均 token delta：-7.54；最差为 -29，占 32K 的 0.09% 以下。
- selected run 数：56--334，平均 161.33。
- 52/52 prompt 均通过原 prompt token parity、selected block parity 和
  本地/服务端 replay tokenizer parity；52/52 生成与评分完整，无 replay
  `finish_reason=length`。

解码非连续 token runs 后，新的文本边界会产生少量 BPE merge，因此模式 B 是
“完全相同 selected token stream 解码得到的文本”对照，而不是 dense server
最终 input token ID 逐个完全相等的对照。

此外，KVMem 基线与 replay 均使用 temperature 0.6，且没有固定 seed；8 条逐题
翻转中可能包含 sampling variance。3 条 replay-only-correct 中还有 1 条对应
KVMem 基线的异常 `finish_reason=length`。若论文要对 contextualized KV 作强因果
结论，应使用相同代码版本，固定 seed 或 greedy decoding，并对两种模式重复运行。

## 7. 可用于论文的结论

### 7.1 当前数据直接支持

1. **在该差异诊断子集上，KVMem 明显优于测试过的 block=32 RAG。**
   KVMem 为 40/52，最佳 RAG 为 29/52，差 11 条和 21.15 pp。
2. **两种方法的 retrieval 集合高度不同。** 最佳 RAG 与 KVMem 的平均 source-text
   Jaccard 只有 23.76%，且没有样本达到 60%。
3. **retrieval 差异能够解释一批具体错误。** 至少有 9 条 KVMem-correct 样本存在
   可定位的 RAG 正向事实缺口候选，分为 exhaustive count、homogeneous-list
   needle 和 round-conditioned fragment 三类。
4. **KVMem 的主要收益首先来自选择了更有效的内容。** 9 条直接 retrieval-gap
   候选使用相同 selected text 重新 dense prefill 仍为 9/9。
5. **contextualized KV 可能对高度碎片化的工具列表有额外帮助，但总体增益较小。**
   全部 52 条中 KVMem 比相同文本 replay 多 2 条，差异只出现在工具列表任务。

### 7.2 当前数据不能直接支持

1. 不能把 76.92% vs 55.77% 写成 AgentLongBench 全集上的无偏总体提升；52 条是
   按方法差异筛选的诊断子集。
2. 不能声称 KVMem 多答对的 11 条全部由 retrieval 导致；最佳 RAG 缺少逐样本评分，
   且生成与推理也可能造成差异。
3. 不能声称 contextualized KV 已被统计显著地证明优于相同文本 replay；当前
   `p=0.7266`，同时存在 sampling 和少量 retokenization 混杂。
4. 不能把 KVMem 的优势归因于保留 original position；当前默认实现会把 selected
   K re-RoPE 到连续窗口位置。

### 7.3 建议的论文表述

可以使用如下较稳妥的表述：

> On a 52-example diagnostic subset selected for disagreement between KVMem and
> prior RAG baselines, KVMem with a 32K active-memory budget achieved 40/52 exact
> matches, compared with 29/52 for the best 32-token RAG variant. The retrieved
> source sets had only 23.76% mean character-level Jaccard overlap, and targeted
> evidence analysis identified nine KVMem-correct examples where RAG omitted a
> directly answer-bearing fact. Re-prefilling the text decoded from KVMem's exact
> selected window preserved 38/52 answers and solved all nine direct-gap cases,
> indicating that retrieval choice accounts for most observed gains, while
> contextualized KV representations may provide an additional benefit on highly
> fragmented tool-list tasks.

在正文或脚注中应同时说明：该子集是 disagreement-based diagnostic set，
selected-text replay 与 KVMem 的 5-vs-3 discordance 未达到统计显著。

## 8. 原始数据与复现入口

### RAG delivery

- `/home/chaidi/kvmem_eval/rag_blocks_52_delivery/README.md`
- `/home/chaidi/kvmem_eval/rag_blocks_52_delivery/overlap_0_blocks.jsonl`
- `/home/chaidi/kvmem_eval/rag_blocks_52_delivery/overlap_4_blocks.jsonl`
- `/home/chaidi/kvmem_eval/rag_blocks_52_delivery/overlap_8_blocks.jsonl`

### KVMem retrieval capture 与分析

- `/data/chaidi/kvmem_eval/results/agentlongbench_rag52_kvmem32k_overlap_20260718/kvmem_retrieval_dump.jsonl`
- `/data/chaidi/kvmem_eval/results/agentlongbench_rag52_kvmem32k_overlap_20260718/analysis/per_sample_overlap.jsonl`
- `/data/chaidi/kvmem_eval/results/agentlongbench_rag52_kvmem32k_overlap_20260718/analysis/overlap_summary.json`
- `/data/chaidi/kvmem_eval/results/agentlongbench_rag52_kvmem32k_overlap_20260718/decisive_gap_analysis/per_sample_decisive_gaps.jsonl`
- `/data/chaidi/kvmem_eval/results/agentlongbench_rag52_kvmem32k_overlap_20260718/decisive_gap_analysis/decisive_gap_summary.json`
- `/data/chaidi/kvmem_eval/results/agentlongbench_rag52_kvmem32k_overlap_20260718/decisive_gap_analysis/decisive_gap_report.md`

### Selected-text dense replay

- `/data/chaidi/kvmem_eval/results/agentlongbench_kvmem32k_selected_text_replay_all52_20260718/accuracy_summary.json`
- `/data/chaidi/kvmem_eval/results/agentlongbench_kvmem32k_selected_text_replay_all52_20260718/answers.jsonl`
- `/data/chaidi/kvmem_eval/results/agentlongbench_kvmem32k_selected_text_replay_all52_20260718/eval.jsonl`
- `/data/chaidi/kvmem_eval/results/agentlongbench_kvmem32k_selected_text_replay_all52_20260718/replay_prompts.jsonl`
- `/data/chaidi/kvmem_eval/results/agentlongbench_kvmem32k_selected_text_replay_all52_20260718/replay_manifest.jsonl`

### 代码

- `scripts/kvmem_eval/capture_agentlongbench_rag52_retrieval.py`
- `scripts/kvmem_eval/analyze_agentlongbench_rag52_overlap.py`
- `scripts/kvmem_eval/analyze_agentlongbench_rag52_decisive_gaps.py`
- `scripts/kvmem_eval/run_agentlongbench_selected_replay.py`
- `scripts/kvmem_eval/run_agentlongbench_selected_replay_smoke9.sh`
