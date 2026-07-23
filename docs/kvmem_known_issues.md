# KVMem Known Issues and Follow-up Work

本文档记录 KVMem 当前仍未解决的问题，供后续实现、回归测试和实验解释使用。
它只记录当前仍需处理的事项；已经修复的问题单独列在文末，避免重复诊断。

- 快照日期：2026-07-21
- 代码基线：`4a8c6f3`
- 默认重点路径：`fp16 KV + query-conditioned mean-k + step update`
- 状态约定：`OPEN` 表示尚未修复，`RESEARCH` 表示属于算法研究问题而非明确的实现错误，
  `LIMITATION` 表示已知但暂未支持的配置或性能限制。

优先级定义：

- **P0**：在扩展到真实 agent/multi-turn 使用前必须修复的语义正确性问题。
- **P1**：会造成检索结果错误、静默改变实验语义或影响实验可信度的问题。
- **P2**：已经观察到的算法质量问题，或尚未形成可靠修复方案的问题。
- **P3**：非默认配置、兼容性、性能和文档一致性问题。

## 1. Issue Summary

| ID | Priority | Type | Status | Summary |
| --- | --- | --- | --- | --- |
| KVMI-001 | P0 | Correctness/API | OPEN | 缺少完整的 context/query/control message 语义和硬保留机制 |
| KVMI-002 | P1 | Correctness | OPEN | 多轮 prefix reuse 下 partial block 的 mean-key 索引可能过期或缺失 |
| KVMI-003 | P1 | Correctness | OPEN | query span 跨越 prefix checkpoint 时不能完整重新捕获 |
| KVMI-004 | P1 | Correctness/Observability | OPEN | scorer 失败时会静默 fallback，配置与实际执行可能不一致 |
| KVMI-005 | P1 | Scalability | OPEN | per-token ExactMass 仍有 shared-memory block 上限和不可接受的显存增长 |
| KVMI-006 | P1 | Experiment integrity | OPEN | 默认测试二进制尚未包含最新 prefill-pressure 修复 |
| KVMI-007 | P2 | Retrieval quality | RESEARCH | mean-k 会稀释和碎片化长 session 中的相关证据 |
| KVMI-008 | P2 | Retrieval quality | RESEARCH | retrieval query hidden state 会受临时 prefill pressure window 影响 |
| KVMI-009 | P3 | Compatibility/Performance | LIMITATION | q8/fp8、layered MTP、同步 NVMe 写和 host selector 等限制 |
| KVMI-010 | P3 | Documentation | OPEN | README 和部分方法文档仍描述已经删除的旧 CLI 语义 |
| KVMI-011 | P0 | Correctness/Numerics | FIX PROTOTYPE | fp16 K 反复原地 re-RoPE 会累积漂移；immutable source K 正在真实样本验证 |
| KVMI-012 | P0 | Correctness/Hybrid state | RESEARCH | normal-attention blocks 重选后，DeltaNet recurrent state 仍来自原时间线 checkpoint |

## 2. Detailed Issues

### KVMI-001 — Message semantics and hard pinning

**Status:** OPEN
**Priority:** P0

当前 server 的自动策略是把最后一个普通 `user` message 的完整 content 当作 retrieval
query。API 也支持顶层 `kvmem_query_span`，用于在一个 string-content message 内显式标记
一个 UTF-8 byte span，但它还不是完整的消息语义接口。

当前缺口：

- 没有 `context-only` warm-up 模式；warm-up 必须人为添加一个短 user confirmation query。
- 没有多 message query、context message indices 或 pinned message indices。
- `system` / `developer` control context 只依赖固定数量的 sink blocks；超出 sink 的部分不保证保留。
- 普通路径只硬保留 sink/recent blocks。当前 query blocks 也参加 Top-K 竞争，没有独立的
  hard-pin invariant；只有实验性的 clean-query 路径调用 `kvmem_set_pin_from_block`。
- query span 已取消原来的 512-token 静默截断；超长 query 会按实际长度增加显存与计算量，
  但不再只保留开头部分。

这会在以下场景中造成错误语义：

- 最后一个 user message 实际上是很长的历史上下文，而不是问题；
- agent 的 system/developer/tool instructions 长于固定 sink 区域；
- 当前 query 或关键的 recent tool trajectory 没有进入检索 Top-K；
- 真正的操作指令位于超长 user message 的尾部，因 512-token cap 被截掉。

代码依据：

- `src/qw3_server.cpp`: `kvmem_query_span` 和 last-user 自动 span 检测。
- `src/qwen_executor.cpp`: `kvmem_set_query_span` 的动态 query-row 分配；
  `kvmem_selection_with_pin`。
- `src/kvmem_store.cpp`: `pick_topk_blocks` 只硬保留 sink/recent。
- `src/qwen_native_backend.cpp`: 只有 clean-query PASS B 设置 `pin_from_block`。

建议修复：

1. 实现 namespaced API：
   `kvmem.mode={last-user|explicit|context-only}`，以及
   `pinned_message_indices`、`context_message_indices`、`query_message_indices`。
2. chat rendering 返回每个 message 的 token span。
3. system/developer、当前 query 和显式 pinned messages 对应的完整 blocks 必须硬保留。
4. 只有 corpus blocks 进入 retrieval Top-K；pinned blocks 单独进行 budget accounting。
5. 对 pinned tokens 超过 budget 的请求显式报错或给出明确的 budget expansion policy。

关闭条件：增加覆盖长 system prompt、context-only warm-up、多 message query、超长 query、
`recent_blocks=0` 下 query hard-pin 的 API 和 selector 回归测试。

### KVMI-002 — Stale partial-block mean-key index after warm reuse

**Status:** OPEN
**Priority:** P1

mean-k session continuation 使用固定 stride 保存历史索引，但 checkpoint/suffix 不按 block
边界对齐时，部分索引仍是近似值：

- resume 时把 `ceil(resume_base / block_tokens)` 之前的 blocks 标记为已经捕获；包含
  `resume_base` 的边界 block 保留上一轮 mean。
- misaligned suffix 的开头没有边界 block 的旧 token K，因此直接跳到下一个完整 block。
- 很长的 misaligned suffix 在每个 2048-token chunk seam 都可能留下一个跨 chunk block 的近似。
- prompt/response 混合 block 不会用 decode tokens 重新计算完整 mean。
- sub-block mean-k 的 decode-time content capture 尚未实现。

KV bytes 本身仍然正确，错误发生在 retrieval index，因此可能导致刚追加的短 tool output、
assistant response 或更新事实不参与后续检索。

代码依据：

- `src/qwen_executor.cpp`: `kvmem_set_query_span` 的 resume seeding；
  `kvmem_capture_kbar_multi` 的 misaligned suffix 处理；
  `kvmem_decode_capture_begin/stage` 的 partial-block 和 sub-block 限制。

建议修复：为每个未完成 block 保存可增量合并的 `sum(K_content)` 和 token count，或者在
resume/reselect 时从 tiered KV 精确重建所有受影响的 boundary blocks。跨 prefill chunk 的
partial accumulator 也应保持连续，不能假设每个调用都 block-aligned。

关闭条件：对所有 `resume_base % block_tokens` 取值，比较 cold full-prefill 与 warm reuse
构建的 mean/sub-block index；短 suffix、超过 2048 tokens 的 suffix、prompt/response 混合
block 均需达到数值一致性。

### KVMI-003 — Query span crossing a reused checkpoint

**Status:** OPEN
**Priority:** P1

prefix reuse 只重新 prefill checkpoint `C` 之后的 suffix。当前 reuse 选择没有验证
`C <= query_begin`。若新的显式 query span 完全位于 checkpoint 前，或横跨 checkpoint，
prefix 中的 query rows 无法重新捕获，`g_query_multi_ready` 可能保持 false，随后进入较弱的
fallback scorer。

正常的“在历史后新增一个独立短 user query”流程通常满足 `query_begin >= C`。2026-07-15
的 30-sample last-user 测试中，30 次 prefix hit 的最终 query 均完整捕获；warm-up 和最终
query 合计 60/60 次 `qcap ready`，server error 为 0。因此这是一个已确认的代码边界缺口，
但不是当前标准实验协议中的高频问题。

代码依据：`src/qwen_native_backend.cpp::kvmem_prefix_reuse`。

建议修复：只有在 `checkpoint <= query_begin` 时允许 reuse；否则选择不晚于 query begin
的 checkpoint，若不存在则 cold prefill。未来若缓存 query rows，也必须按本次请求的 span
identity 验证后才能复用。

关闭条件：覆盖 query 完全在 suffix、完全在 prefix、横跨 P、横跨 M 四类测试，并验证
scorer 实际使用完整 query token 数。

### KVMI-004 — Silent scorer fallback

**Status:** OPEN
**Priority:** P1

`kvmem_prepare_reselect` 在 query/index/kernel 不可用时，会从配置的 mean-k 或 per-token
scorer 回退到 single last-token content scorer，再进一步依赖 window-local profile/recency。
默认输出没有统一记录 requested scorer、actual scorer 和 fallback reason。

因此看到“reselection 被调用”或请求中配置了 `--kvmem-retrieval-method mean-k`，并不能单独
证明 mean-k 确实完成了本次打分。q8/fp8 dtype、incomplete query capture、ExactMass shmem
cap、buffer 未 ready 或 kernel error 都可能改变实际 scorer。

代码依据：

- `src/qwen_executor.cpp::kvmem_prepare_reselect`。
- `src/qwen_executor.cpp::kvmem_retrieval_score_mean_softmax`。
- `src/qwen_executor.cpp::kvmem_retrieval_score_exactmass`。

建议修复：每次 reselect 记录结构化字段：

```text
scorer_requested
scorer_used
fallback_reason
query_tokens_expected
query_tokens_captured
indexed_blocks
selected_blocks
```

这些字段应进入 server log、timing/trace summary，并可选进入 API usage metadata。实验模式
可提供 `QW3_KVMEM_REQUIRE_SCORER=mean-k`，一旦 scorer 不匹配就 fail-fast。

关闭条件：所有 fallback 分支都有稳定 reason code，并有测试确认不会静默改变 scorer。

### KVMI-005 — ExactMass scalability

**Status:** OPEN
**Priority:** P1

per-token ExactMass kernel 为每个 query-token/head CTA 在 dynamic shared memory 中保存
`q_vec[head_dim] + block_mass[n_blocks]`，并设置 48 KiB cap。它大约在 12K blocks 后返回
失败；block size 32 时约为 38--39 万 tokens。失败后当前调用链会进入 KVMI-004 所述的
fallback。

更早的限制来自 raw-key buffer：`[L, total_tokens, n_kv_heads, head_dim] fp32` 在约 113K
tokens、16 个 normal-attention layers 时已经约 7.4 GiB，百万 token 上下文不可实用。
此外 above-budget per-token 模式目前拒绝 prefix reuse，虽然保证正确性，但会退化为完整
cold prefill。

代码依据：

- `src/kernels_cuda.cu::launch_block_attn_score_exactmass`。
- `src/qwen_executor.cpp::kvmem_retrieval_score_exactmass`。
- `src/qwen_native_backend.cpp::kvmem_prefix_reuse`。

建议修复：在决定继续支持该 CLI 前，二选一：

1. 实现 tiled/online-softmax ExactMass，并设计分层、量化或 streamed raw-key storage；或
2. 对不可支持的 context/block 配置在启动或请求开始时明确拒绝，而不是运行中静默 fallback。

### KVMI-006 — Canonical binary is older than the pressure-policy fix

**Status:** OPEN
**Priority:** P1

源代码提交 `119ce05` 已实现 deterministic sink+full-recent-tail prefill-pressure policy，并有
独立 smoke build。当前 canonical `build/qw3` 的构建时间为 2026-07-16 08:15，而该提交时间
为 2026-07-16 13:43。AgentLongBench 和 warm-query shell scripts 又直接引用
`build/qw3`，所以按默认命令重新运行仍可能使用旧实现。

这不是未修复的源代码 bug，而是实验发布/可复现性问题。

建议修复：

1. 重新构建 canonical `build/qw3` 并运行完整 KVMem tests。
2. 所有 eval scripts 支持统一的 `QW3_BIN` override，避免硬编码。
3. 每次实验在 `run_config.json` 中记录 git commit、binary mtime/hash 和关键环境变量。
4. 对最新 binary 再运行一次 multi-turn pressure regression；单轮 AgentLongBench 不能覆盖该问题。

### KVMI-007 — Fragmented retrieval and mean-key dilution

**Status:** RESEARCH
**Priority:** P2

这是当前默认 mean-k 的主要准确率瓶颈，不是 KV tiering、window assembly 或 scalable
softmax kernel 的机械正确性 bug。

已观察证据：

- LongMemEval-M 10 samples，2M ctx、200K selected budget、block size 32、sink 8、recent 0、
  mean-k、无 sub-block：人工准确率 6/10。
- 四个错误样本的 gold-session 近似 block coverage 分别为 60/194、30/239、61/256、
  59/233；每个 session 都有高排名 block，但只保留了零散证据。
- 常见错误包括事实与日期/header 分离、旧值和更新值同时存在、相似 distractor 混合、
  temporal order 判断错误。
- AgentLongBench 250 样本中，32K budget 官方分数为 60.87%，64K 为 60.62%；单纯扩大
  budget 没有稳定改善。

建议研究顺序：

1. 对高分 block 做可控的 left/right neighbor expansion。
2. 引入 session/header-aware group selection，确保事件内容与日期/会话头共同保留。
3. 对 update/temporal 问题增加 recency-aware tie-breaking 或结构化 session metadata。
4. 运行 answer-local oracle，拆分 retrieval recall 与 downstream reasoning error。
5. 在上述诊断之后再决定是否继续扩大 sub-block 或改写 scorer。

相关记录：`docs/kvmem_implementation_notes.md` §12.1、§12.7。

### KVMI-008 — Query state depends on the temporary pressure window

**Status:** RESEARCH
**Priority:** P2

在超出 GPU page pool 的长 prefill 中，当前 query hidden state 是在 temporary sink+tail
pressure window 上计算的，而不是在完整历史上计算的。因此 retrieval query Q 可能受到临时
窗口构成影响。deterministic pressure policy 修复了多轮之间的不确定性，但没有证明该 Q 是
最优 retrieval representation。

现有替代方案也没有通过验证：`QW3_KVMEM_CLEAN_QUERY=1` 将 query 独立 prefill，在
LongMemEval-S 500 samples 上从默认路径的 77.2% 降到 57.0%，不能作为默认修复。
`QW3_KVMEM_FULLCTX_QUERY` 只适合显存允许的诊断，不是 bounded-memory 通用方案。

建议继续使用 full-context-query、pressure-window query 和不同 clean-query construction 做
固定样本的 retrieval-rank 对照；在没有稳定收益前保持 clean-query 默认关闭。

### KVMI-009 — Non-default compatibility and performance limitations

**Status:** LIMITATION
**Priority:** P3

- tiered CPU/NVMe offload 不支持 1-byte KV dtype；mean-key/de-RoPE retrieval 期望 fp16/fp32。
- q8/fp8 retrieval kernel 失败时可能退化为 recency/profile，需与 KVMI-004 一起 fail-fast。
- KVMem 与 opt-in layered MTP verifier 不兼容；当前会明确 hard error。默认 ragged verifier 可用。
- NVMe stage-in reads 已重叠，但 stage-out writes 仍同步。
- score 需要 D2H，Top-K selector 在 host 上执行。
- selected block set 在所有 standard-attention layers 之间全局共享，不能表达 layer-specific memory。

这些限制不阻塞当前默认 `fp16 + mean-k + ragged MTP` 实验，但应在扩展配置前分别处理。

### KVMI-010 — Stale public documentation

**Status:** OPEN
**Priority:** P3

`README.md` 和 `docs/KV_Memory_Paper.md` 仍描述已删除的
`mean_attention | content_mean` retrieval-method 值；README 还写着
`recent_blocks=0` 会自动派生 recent allocation。当前 CLI 实际支持
`mean-k | per-token | sub-block-mean-k`，并且 zero recent 是 literal zero。

建议在算法和 API 行为稳定后统一更新 README、paper notes 和所有示例命令；在此之前以
`src/qw3_cli.cpp --help`、`docs/kvmem_implementation_notes.md` 和实际 eval scripts 为准。

### KVMI-011 — Repeated in-place fp16 re-RoPE changes historical K

**Status:** FIX PROTOTYPE / REAL-SAMPLE VALIDATION
**Priority:** P0

默认 window assembly 会直接修改 repository 中的 fp16 K：从当前 `baked_pos`
de-RoPE，再 RoPE 到新窗口位置。单次映射的误差很小，但 transcript/multi-turn replay
可能让相同 block 在多个窗口之间反复移动；后一次映射读取的是前一次已经舍入过的 fp16
结果，因此误差会累积。它影响所有读取该 block 的后续 prefill、query replay 和 decode，
不只是 retrieval query。V 没有位置旋转，不受该问题影响。

当前证据：

- frozen LongMemEval-M 首个错误样本中，Japan gold evidence blocks 已完整进入最终 224K
  selection，但旧 transcript replay 仍回答不存在 Japan 行程，排除了“完全未召回”。
- `qw3-kvmem-immutable-k` 的 1800 轮 CUDA control 中，旧原地路径相对一次直接映射的
 最大单元素误差为 `0.201660`；不可变 source 路径与一次直接映射逐字节一致。
- 这项证据证明旧路径存在真实的数值漂移，但尚不能证明它是十个真实样本的唯一或主要
  准确率根因；不常被选中的 gold block 可能只经历很少映射。

当前默认修复使用 immutable source K；`--no-kvmem-immutable-k` 保留旧原地路径用于
消融，旧脚本也可使用 `QW3_KVMEM_IMMUTABLE_SOURCE_K=0|1` 覆盖。repository K 作为不可变
source，每次 assembly 复制到额外 working K 后只做一次 source-frame -> window-frame
映射；attention 读取 working K，tiering 原样保存 source K，V 仍为单副本。fp16 256K
resident pool 额外占用约 8 GiB。为降低显存，开发分支已补齐 FP8 KVMem re-RoPE、
window k-mean、attention-mass 与 content-k-mean CUDA 路径；配合 `--kv-dtype fp8` 时，
额外 working K 约 4 GiB，source+working 每轮仍保持 one-shot deterministic。FP8 会引入
一次性的量化误差，必须与 fp16 immutable 做真实样本准确率对照后才能设为推荐配置。

若该修复不能恢复 gold-block-complete 的样本，下一根因应转向 source KV 的上下文构建
质量（尤其是 pressure window 下构建的 hidden state）及 first-pass query 所依赖的上一轮
临时窗口，而不是继续调 re-RoPE 数值精度。

### KVMI-011A — MTP prefix used out-of-range logical RoPE positions

**Status:** GUARDED / COMPACT REBUILD OPEN
**Priority:** P0

目标模型在 KVMem pressure 后使用压缩到 256K 内的 attention window，但旧 MTP prefix
priming 仍按完整 trace 的逻辑位置写入 Q/K；1M trace 因而会让 MTP 先在 `>=256K` 的
位置执行 RoPE，再尝试在最终 window assembly 中恢复。这不改变 target verifier 的理论
正确性，但会降低 draft 质量，而且超范围 bake 不应被视为可靠、可逆的表示。

当前保护策略是：KVMem 请求的 `logical_prompt + max_tokens` 可能超过 `n_ctx_train` 时，
不再 prime/use MTP prefix，也不进入 continuous-MTP lane；decode 回退到 target model 的普通 windowed
路径。executor 内还有第二层边界检查，防止遗漏调用源继续执行超范围 MTP RoPE。没有使用
modulo/clamp，因为那会静默破坏相对位置。后续若要恢复超长请求的 speculative speedup，
需要从最终 selected window 在紧凑位置重建 MTP prefix，而不是恢复超长位置的 MTP K。

### KVMI-012 — Selected attention window and recurrent state describe different histories

**Status:** OPEN / DELTANET-STATE DEBUG ARCHIVED
**Priority:** P0

Qwen3.6 是 normal-attention 与 DeltaNet recurrent layers 的混合模型。KVMem 的 block
selection 只重组 normal-attention K/V；recurrent layers 没有 token-wise KV blocks。
当前 query replay 在重放最终问题前，会恢复原时间线 query boundary 的 recurrent/conv
checkpoint。因此最终 forward 实际同时读取：

- mean-K 选出的约 224K normal-attention K/V；
- 顺序处理完整约 1.1M trace 后形成的 DeltaNet state。

这两个状态不描述同一个有效上下文。相同 selected tokens 的 dense text replay 会同时重建
normal-attention K/V 和 DeltaNet state，因而它答对不能单独证明问题只在 K/V。

当前证据：

- frozen Japan 样本的 gold blocks 已完整进入最终 selection，但 transcript cached-KV 路径
  仍回答没有 Japan 行程；相同 selected tokens 重新 prefill 后回答 `two weeks`。
- immutable source K 消除了 1800 次 re-RoPE 数值漂移，但首个真实样本仍错误，说明仅修复
  K 的数值帧不够。
- DeltaNet 单 token 更新含 `S(I-bkk^T)+B`，不是只含标量 decay 的可交换状态。精确组合
  任意 selected blocks 需要保存每 block、每 layer 的完整 affine transform，内存不可接受。

2026-07-21 的 frozen-10 构建消融进一步定位到“新旧表示混合”：

- 原始 cached-KV 为 0/10；仅 rollback/replay query 为 4/10；把同一批 224K selected
  source tokens 解码后 dense prefill 为 9/10。
- 在 knowledge-update 样本 `031748ae` 的真实 session-local score dump 中，新值
  `5 engineers` 的块排名第 3，旧值 `4 engineers` 的块排名第 29；两者都在 top-1024，
  因此该样本不是 retrieval miss。
- 相同 top-1024 tokens 按 source order 离线 dense 构建时，32K、8K、2K、1K 四档都
  正确回答 `4 -> 5`。真实 KVMem 若在 fresh 32K 之外再混入 32K 旧 cached evidence，
  只回答旧值；混入约 192K 时甚至会回答没有相关信息。
- 最终候选保留历史 KV 作为 mean-K 检索库，但 answer window 只保留 8 个稳定 sink
  blocks，并对 top-1024（32K）source tokens 进行一次 source-order fresh prefill。
  它已在 `031748ae` 正确回答 `4 -> 5`，并在 full10 的首条 Japan 回归正确回答
  `two weeks`。其余 frozen 样本仍在后台验证，尚不能将本 issue 标记为关闭。

2026-07-22/23 又完成了两种 DeltaNet recurrent-state 对照，但结果不足以支持继续保留
在线 debug 接口：

- **replacement rebuilt-state**：保持 mean-K 选出的 normal-attention KV 不变，把相同
  约 224K source tokens 按原顺序 dense prefill，并用所得 DeltaNet recurrent/conv
  state 替换长 trace 的累计 state。脚本内联评分为 `1/10`；对完全相同的输出使用
  DeepSeek V4 Pro 重评为 `6/10`。评分口径差异过大，不能把该结果解释为稳定修复。
- **accumulated-state refresh**：先保留 query boundary 的累计 state，再把相同 selected
  tokens 作为普通 DeltaNet updates 追加进去，结果为 `5/10`。它同样没有相对
  query-replay 对照形成稳定、可归因的提升。
- 每条约 229K selected tokens 的 state export 额外耗时约 `91 s`，artifact 大小约
  `157,811,352 bytes`（`150.5 MiB`）；完整回答的平均 TTFT 约 `611 s`。

因此 `kvmem_rebuilt_state_{export,import,capture,seed}` 请求入口现已明确拒绝，executor
state 文件读写和 backend 注入分支均编译禁用；两条实验脚本也标为 archived。这里禁用的
仅是 rebuilt-state debug/ablation，不是模型正常 DeltaNet forward，也不影响 mean-K、
query replay 或 immutable source-K。历史实现仍可从 commit `9792a0a` 恢复。

对应原始记录：

- score dump：`/data/chaidi/kvmem_eval/results/longmemeval_m_transcript_sessionlocal_s9_scores_20260721.jsonl`
- knowledge-update 修复：`/data/chaidi/kvmem_eval/results/longmemeval_m_transcript_blockrefresh32k_cachesink_tb4k_s9_20260721_eval_20260720_224550.jsonl`
- full10：`/data/chaidi/kvmem_eval/results/longmemeval_m_transcript_blockrefresh32k_cachesink_tb4k_full10_20260721_eval_20260720_225437.jsonl`
- replacement rebuilt-state：
  `/data/chaidi/kvmem_eval/results/longmemeval_m_k224k_query_replay_immutable_rebuilt_state10_fixed_20260722`
- accumulated-state refresh：
  `/data/chaidi/kvmem_eval/results/longmemeval_m_k224k_query_replay_immutable_deltanet_refresh10_20260722`

保留的默认关闭 A/B 隔离：

1. `QW3_KVMEM_TRANSCRIPT_STABLE_INGEST=1`：中间 turn 的 durable source K/V 在固定
   sink+recent window 下构建，排除前一轮 semantic selection 污染。
2. `QW3_KVMEM_TRANSCRIPT_RESET_FINAL_RECURRENT=1`：保持最终 selected K/V 完全不变，
   仅让最终 query 从空 recurrent/conv state 开始。
3. `QW3_KVMEM_TRANSCRIPT_RESET_EACH_RECURRENT=1`：每个 durable user turn 形成局部
   recurrent segment；normal-attention 仍读取 KVMem window，不重新计算整个 224K。
4. `QW3_KVMEM_TRANSCRIPT_REFRESH_BLOCKS=1`：从 query-conditioned mean-K 的结果中取
   top blocks，按 source order bounded fresh prefill；该模式默认只把稳定 sink 与 fresh
   blocks 组装到最终窗口。`QW3_KVMEM_TRANSCRIPT_REFRESH_CACHED_TOKENS` 仍可显式扩大
   cached 对照，但当前实验证明大 cached/fresh 混合会显著降低正确性。

关闭条件：在 frozen 10-sample LongMemEval-M 集合上分离 K/V construction 与 recurrent
state 的贡献，选定能恢复答案且不要求全量 selected-token prefill 的路径，并完成
LongMemEval-S 与默认 one-shot/multi-turn 回归。

## 3. Issues No Longer Open

以下问题已有明确修复或归档，不应继续作为当前根因重复排查：

| Issue | Current status |
| --- | --- |
| mean-k scorer 超过 8192 blocks 后 fallback | 已新增 exact tiled two-dot path；8192/8193 CUDA parity tests 已覆盖 |
| `recent_blocks=0` 隐式保留 `budget/4` | 已修复；zero 现在表示不硬保留 suffix |
| warm query 使用 suffix-relative/absolute 坐标混淆 | `716ee65` 已修复；标准 30-sample multi-turn 测试无 server error |
| 旧 warm-query CUDA illegal memory access | 修复 query absolute-coordinate capture 后未在标准协议中复现 |
| prefill pressure 受上一轮 semantic selection 影响 | `119ce05` 源代码已修复；canonical binary 更新另由 KVMI-006 跟踪 |
| large-context CUDA garbage output | 已修复并记录于 `docs/BUG_SUMMARY_large_ctx_garbage.md` |
| DeltaNet retrieval 作为推荐 CLI 方法 | 已归档并从 CLI 移除；保留实验记录供未来研究 |
| KVMem continuous batching 全面不支持 | 该结论已过时；默认 window-aware ragged verify 路径可用 |

## 4. Validation Snapshot

截至本快照：

- `qw3-kvmem-store`、`qw3-kvmem-assembly`、`qw3-kvmem-softmax-pages` 测试通过。
- 独立 prefill-pressure build 中 smoke/store/pinned-tier/NVMe-tier 测试通过。
- 正确 last-user 协议的 LongMemEval-S 30-sample 测试：30 次 prefix hit、60/60 次 query
  capture ready、60 次 mean-k scoring、0 server errors；aggregate accuracy 25/30 = 83.33%，
  与 one-shot 25/30 相同，但逐样本结果并非完全一致。
- 上述测试没有覆盖 KVMI-001 的通用 message pinning、KVMI-002 的所有 misalignment、
  KVMI-003 的跨 checkpoint query 和 KVMI-005 的大规模 ExactMass。

## 5. Recommended Implementation Order

1. **KVMI-011 + KVMI-012**：完成 frozen LongMemEval-M KV-construction/recurrent-state
   A/B，收敛不需要全量 selected-token prefill 的正确构建路径。
2. **KVMI-001**：实现 message-aware query/context/pin 语义和 hard-pin invariant。
3. **KVMI-002 + KVMI-003**：修复 warm continuation 的 partial-block index 和 checkpoint guard。
4. **KVMI-004**：让每次 scorer/fallback 可验证，并为实验提供 require-scorer 模式。
5. **KVMI-006**：更新 canonical binary 和实验 provenance。
6. **KVMI-007**：实现 neighbor/session/header-aware retrieval，并先做 oracle 分解。
7. **KVMI-005**：决定 scalable ExactMass 的实现或明确撤出不可支持配置。
8. **KVMI-008/009/010**：继续研究 query construction，处理扩展配置和文档收敛。
