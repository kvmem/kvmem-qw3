# MemoryAgentBench × KVMem Context Archive 评测方案

Date: 2026-08-02  
Official repository commit: `455306dcabc3842526eb83cd4e225e5d486c5c5d`

## 1. 评测范围

本地 parquet 与 Hugging Face `ai-hyz/MemoryAgentBench` 的四个 split 对齐：

| Split | Context rows | Questions |
|---|---:|---:|
| Accurate_Retrieval | 22 | 2,000 |
| Conflict_Resolution | 8 | 800 |
| Long_Range_Understanding | 110 | 171 |
| Test_Time_Learning | 6 | 700 |
| **Total** | **146** | **3,671** |

MemoryAgentBench 的一个 row 是“一份固定 context + 多个问题”，正好对应 durable
archive 的 frozen branching：context 只 prefill/build 一次，同一 row 的全部问题都从
完全相同的 archive base state 独立分支，不把前一个答案追加到下一个问题。

## 2. Prompt parity

`prepare_memoryagentbench_archive_row.py` 复现官方 `LongContextAgent`：

1. 使用 `gpt-4o-mini` tiktoken 和 NLTK sentence tokenizer，按官方 4096-token
   control flow 切分原 context；
2. 对每个 chunk 应用官方 `memorize` template；模板中原本存在的字面量
   `\\n<User>` 也逐字保留；
3. 使用官方 system message；
4. archive prefix 在 `<|im_start|>user\n{memorized context}` 后结束，故意不关闭
   当前 user turn；
5. 每个 frozen query 通过 `qwen-user-continuation` 追加
   `\n{official query template}<|im_end|>...assistant...`。

因此 role 结构仍是官方的一条 user message：`memorized context + query`，没有为了
cache reuse 额外增加一个 user/assistant round。为了满足 raw-K 的 2048-token durable
chunk 边界，输入尾部可能补 0–2047 个 newline token；原始 token 数和 padded token 数
都会逐 row 记录。

带 `{time_stamp}` 的官方模板原本在每个 chunk 调用 wall clock。主实验固定为
`2026-08-02 00:00:00`，以免同一个 benchmark prompt 因运行时间不同而改变；这是唯一
有意做成 deterministic 的模板字段，值会写入每个 row 的 manifest。

## 3. 主实验参数

| Parameter | Value |
|---|---|
| Model | Qwen3.6-27B-Q8_0 |
| Archive KV | FP8 immutable raw-K/V |
| KVMem context budget | 204,800 tokens（200K） |
| Generation reserve | 32,768 tokens |
| Retrieval block | 512 tokens |
| Retrieval | Adaptive Multi-Prototype Key-direction，gain 0.10 / 0.06 |
| Retrieval index/query | FP16；index authority on CPU |
| Sink / recent | 2,048 / 16,384 tokens |
| Query replay | on |
| Immutable refresh | every semantic materialization (`=1`) |
| Prefill chunk | 2,048 tokens |
| MTP | chain 4（archive default） |
| Sampling | temperature 0.6, top-p 0.95, top-k 20, no thinking |
| GPU memory ratio | 0.50 |
| Optimizations | stage-out + stage-in + pack, all on |

这里的 Adaptive Retrieval 指每个最终问题触发的 semantic reselection。Archive
顺序构建期间的 pressure reselection 在实现中固定调用 sink+recent，不进行语义评分；
因此 builder 使用较便宜的 mean-K 临时索引不会改变选中的 pressure window。Attach 后会
从 immutable raw-K 按当前 policy 重新构建 adaptive index，随后才处理问题并触发
query-conditioned semantic reselection。

每种 sub-dataset 使用官方 `generation_max_length`：Fact=10、ICL=20、EventQA=40、
Ruler/LongMemEval=50、Recsys=300、InfBench=1200、DetectiveQA=2000。

## 4. CPU-first archive placement

runner 先通过 `qw3 tokenize --token-output` 得到精确 Qwen token 数；同一个 binary
token file 直接交给 archive builder，避免再次 tokenize 超长文本。token file header
同时保存完整 GGUF SHA-256；builder 会拒绝由不同模型/tokenizer 生成的 token stream。

- 估算 archive（raw-K + V + ladder state）不超过 50 GiB 且 `/dev/shm` 空间足够：
  row archive 放在 tmpfs，即实际使用 host memory，不产生 SSD stage-in；CPU tier 16 GiB。
- 超过阈值：仅该 row 临时落到本机 NVMe，CPU tier 64 GiB，并保留 OS page cache。
- row 的所有问题结束且结果 durable 后删除临时 raw-K/V archive；prefix token file、
  answers、logs、逐问题结果永久保留。
- 构建中断时保留 archive，可从最后一个 262,144-token ladder 恢复。

当某个 row 的 aligned context 短于全局 200K budget 时，有效 budget 自动取该 row 的
完整 context 长度，sink=2K、recent=剩余全部 prefix；这等价于 full-context，不执行
无意义的短文本丢弃，同时保证 immutable archive 仍使用合法的 bounded tiered page
pool。超过 200K 的 row 保持主实验的 200K/2K/16K 配置不变。每个 row 的实际三项值
写入 `row_summary.json`。

该策略不会把实现绑定到当前机器：阈值、tmpfs/SSD root、CPU tier 和 GPU ratio 都是
runner 参数；换机后只需根据 RAM/NVMe 容量调整配置。

## 5. 结果与评分

每个问题立即追加一条 JSONL，包含 stable row/question ID、原问题、gold、完整模型输出、
archive/prompt/decoded token 数，以及 wall/prefill/decode 时间。每完成一个 context 就写
`row_summary.json` 和全局 `progress_summary.json`，重启会跳过已完成 row。

评分分两层：

1. `score_memoryagentbench_official_local.py` 直接调用 pinned official repository 的
   `post_process`，覆盖 exact/substr/F1/ROUGE、EventQA 辅助 recall 和 ReDial
   Recall@K；论文主指标仍严格按 README：EventQA/Ruler/Fact 用 substring EM，
   Detective/ICL 用 exact EM；
2. `judge_memoryagentbench_special.py` 使用官方 LongMemEval answer-equivalence prompt，
   以及 InfBench fluency/recall/precision 三段 rubric。judge model 单独记录；若使用
   DeepSeek-v4-pro，这与官方评估流程相同但不应声称 judge model 与论文的 GPT-4 相同。

DeepSeek-v4-pro 的 LongMemEval 主评分显式固定 `thinking=enabled`（V4-Pro API
默认）并预留 1,024 个 completion tokens。官方 GPT-4 脚本的 10-token 上限
不能直接移植到 reasoning 模型：thinking tokens 会先耗尽上限并留下空的
final content。InfBench 的三段官方 rubric 要求返回短 JSON，实测
`thinking=enabled` 会在 precision 评审中用完 4,096 个 reasoning tokens 仍没有
final JSON；因此 InfBench 单独固定 `thinking=disabled`，并在 result/cache identity
中与 LongMemEval 分别记录。
评分器现在把空 final content 视为 API 失败而非错误答案，并将 model、endpoint、
thinking mode、token budget 和 prompt SHA-256 全部纳入逐调用 cache identity。
每次成功的 API response 都立即 fsync 到 sidecar，断点恢复不会重复请求。

## 6. 执行入口

```bash
/home/chaidi/kvmem_eval/KVMem_Motivation/.venv/bin/python \
  scripts/kvmem_eval/run_memoryagentbench_archive.py \
  --out-dir /data/chaidi/kvmem_eval/results/memoryagentbench_kvmem_archive_20260802
```

该命令支持安全重启；已存在 `row_summary.json` 的 row 不会重复执行。

## 7. 当前状态（2026-08-02）

- Official prompt/template parity：已用所有八类 source 做逐字节检查；通过。
- 评测数据清点：146 contexts / 3,671 questions；完成。
- CLI/runner/JSONL/CPU-first placement：完成。
- 真实 smoke：Conflict Resolution row 0，8,192 archive tokens、2 questions；adaptive
  scorer fallback=0，结构化结果 2/2，官方本地 headline substring EM=0.5。
- 10M multi-query 发现并修复 archive SSD→CPU cache admission；完整 20 问平均 CPU
  hit=85.1%，稳态 semantic KVMem 6.225→4.321 s，selected hash 不变。
- 全量生成已完成：146 / 146 contexts、3,671 / 3,671 questions；3,671 次
  semantic scorer event 全部 `fallback=0`。较小 archive 使用 tmpfs，超出
  tmpfs 策略上限的 archive 使用 NVMe；结果根目录为
  `/data/chaidi/kvmem_eval/results/memoryagentbench_kvmem_archive_20260802_full`。
- Accurate Retrieval 已生成 2,000 / 2,000 问；LongMemEval 使用
  DeepSeek-v4-pro thinking-enabled 官方 prompt 评分为 157 / 300 = 52.33%。
- Conflict Resolution 已生成并完成官方 deterministic post-process：
  single-hop 6K/32K/64K/262K 分别为 95%/94%/87%/73%；multi-hop 分别为
  17%/16%/14%/5%。
- InfBench 首条完整输出为 919 / 1,200 tokens，retrieval fallback=0。官方
  fluency/recall/precision 三段 rubric 端到端 smoke 通过。thinking-enabled
  precision 会耗尽 4,096 reasoning tokens 且不返回 final JSON；该失败调用未写入
  judgment，评分器已按上文所述的 source-specific thinking 语义修复。
- 完整 3,671 问 deterministic post-process 与全部 400 条 special judge 均已完成。
  DeepSeek-v4-pro 的 LongMemEval thinking-enabled judge 为 157 / 300 = 52.33%；
  InfBench thinking-disabled 三段 rubric 的平均 summary F1 为 46.59%。审计结果为
  400 个唯一 key（300 LongMemEval + 100 InfBench），无缺失、无多余、无越界分数；
  `special_judge_summary.json` 与 `final_summary.json` 已同步更新。
