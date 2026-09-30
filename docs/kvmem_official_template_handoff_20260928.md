# 官方任务模板 + 10% 重算直接替换 KV：复现交接

## 要复现的实验

Qwen3.8-27B-Q8_0，KV Memory 的检索/prefill 窗口为 32768 tokens。Query 触发召回、窗口重定位后，按已有检索分数选约 10% 的历史 token，重新经过模型，用新的 attention K/V 直接覆盖工作窗口对应行，然后完整 replay query 并生成答案。

这里“官方模板”指 RULER 官方 NIAH 任务正文；“直接替换”指重算 KV 100% 替换旧 KV。它不是无重算、不是全窗口重算、不是 50/50 混合，也不等同于 CacheBlend 论文的逐层 KV 偏差选 token。

## 直接使用冻结输入

实验根目录：`/data/chaidi/qw3-experiments/cacheblend_20260925/official_template_20260927`。

- `cases/*.tokens.i32`：180 个完整模型输入，little-endian int32 token IDs。
- `cases/*.json`：实际输入长度、名义长度档、query span、答案和证据元数据。
- `cohort.json`：顺序固定的全部案例与路径。
- `prepare_inputs.py`、`template-audit.json`：模板转换和一致性校验。
- `official-source/provenance.json`：官方源码与哈希。固定提交 `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`。

四类为 `niah_single_2`、`niah_multikey_2`、`niah_multivalue`、`niah_multiquery`。每个长度档每类 15 例：seed=1000/2000/3000，depth=10/30/50/70/90，共 180 例。优先读取冻结 IDs，不重新生成正文或自行套默认聊天模板。跨实现先确认模型词表 ID 映射相同；答案/证据位置仅供评分审计，不能送入模型或用于选块。

本组去掉原 xhigh system 消息，采用官方 NIAH 任务正文、单复数转换以及多名称的 `, and` 连接方式。外层为 Qwen user/assistant 标记，保留 `<think>` 起始标记，无 system 消息、无额外答案预填。正文 token 和全部证据 token 不变。实际长度分别为 65493–65495、131029–131031、262101–262103；证据绝对位置相对原输入前移 42 tokens。因此与原模板的比较也包含块对齐变化。

这是官方任务模板适配实验，保留原有 thinking 与生成预算，不是官方数据生成及 128-token 生成流程的完整复现。

## 精确推理配置

以根目录的 `command.json`、`env.json`、`config.json` 和实际请求 `trials/*/*-request.json.gz` 为准。

- 冻结二进制：`/data/chaidi/qw3-experiments/cacheblend_20260925/bin/qw3`。
- 二进制 SHA256：`48b6415c4adfad971dd4a55896d59ea029a1861a7d8055df8aa834c2214f29cd`。
- 模型：`/home/chaidi/qw3/models/Qwen3.8-27B-Q8_0.gguf`；KV dtype=FP8，MTP4。
- `QW3_KVMEM_BLEND_RATIO=0.1`，`QW3_KVMEM_BLEND_MAX_TOKENS=4294967295`，`QW3_KVMEM_NO_REROPE=0`，`QW3_KVMEM_GUIDED_MTP=1`。
- 冻结二进制为直接替换实现，没有启用混合。若用新增 alpha 参数的当前实现，对应 `QW3_KVMEM_BLEND_ALPHA=1`；不要使用 0.5。
- block=32，history budget=32768，prefill budget=32768，generation reserve=16384，prefill chunk=2048。
- sink=1312，recent=10496，retrieval=mean-k，query-conditioned 开启，CPU index，repository blocks=16384。
- guided reselect=`both`，guided thinking tokens=0，guided query tokens=512，guided MTP 开启。不是仅使用原问题 Q 的配置。
- answering thinking 开启，thinking budget=4096，max output=8192，temperature=1，top_p=0.95，top_k=20，min_p=0，presence penalty=0，repetition penalty=1；请求 seed 使用各样例 seed。
- `--ctx 524288` 是服务配置，不是 512K 工作 attention 窗口；32K history/prefill 与额外 16K generation reserve 要区分。

## 每个案例的请求顺序

1. 创建全新服务进程，避免跨样例状态污染。沿用原 runner 的四 token reset。
2. 取 `boundary = floor(query_begin / 32) * 32`。不提供 query 或答案，按累计前缀逐次 ingest，每段增加最多 65536 tokens，最后到 boundary。请求 `max_tokens=0`、`enable_thinking=false`、`kvmem_reselect=off`。
3. Answer 请求发送完整冻结 token 数组，包括 query 与 assistant 前缀；`kvmem_query_token_span={begin:query_begin,end:query_end}`，`kvmem_reselect=auto`。其余生成参数如上。
4. 保存请求体、响应、日志和成绩，再停止该案例的服务。

`run.py` 是已跑完这组的实际 runner。不要直接重跑到已完成目录；在新输出目录使用独立 runner，并保留输入、模型、二进制与脚本身份记录。

## 召回和重算的内部语义

1. 按现有 guided query / mean-k 流程选择窗口，并完成 K 的 re-RoPE 位置更新。重算发生在选窗之后、正式 query replay 之前。
2. 从选中窗口里排除未改变的连续历史前缀，以及本来就会 query replay 的 suffix。候选块须有对应原始 token。
3. 按 `retrieval_score` 从高到低选块，同分优先较早窗口位置。预算是 `min(floor(history_window_tokens * 0.1), max_tokens)`，按完整 32-token 块装入。本组所有案例的 query 前窗口为 32704 tokens，实际重算 102 块、3264 tokens。不是原始 256K 输入的 10%，也不是随机 10%。
4. 选完后改按窗口位置从前到后执行；只有历史位置和窗口位置都连续的块才合并为一次连续 forward。执行顺序不是分数顺序。
5. 复用 `forward_n_tokens()` 跑这些原始 token 的全部模型层。attention 可读取它前方的完整工作窗口，遵守 causal mask；不是只对选中块互相做 attention，也不访问未来内容。
6. 所选行的 attention K 和 V 均用新值完全替换；其他历史行保持已有 KV。GDN 层不是普通 K/V cache，按下面的独立状态语义处理。

## GDN 与缓存生命周期

- 重算前保存 query 边界的 recurrent、conv、hidden 和逻辑位置。
- 整个重算开始时，recurrent/conv **只清零一次**；各选中段连续更新同一临时状态，跨段不清零。未选中的中间历史不经过 GDN。
- 全部重算结束后恢复原 query 边界计算状态，保留工作窗口的新 attention KV，再执行完整 query replay。
- 这是一种稀疏历史重算近似，不等价于完整窗口 prefill。
- 只临时修改 GPU 工作窗口；不改 immutable raw K、历史索引、CPU/NVMe backing 或 MTP 历史 KV。重排、stage-out 等需要源 KV 时恢复备份，避免把临时修补写回历史存储。

实现参考 `/home/chaidi/qw3/src/kvmem_blend.cpp` 与 `src/kvmem_blend_policy.hpp`。当前源码包含后加的 alpha 混合分支，复现本组须使用 alpha=1；精确已测行为以冻结二进制为准。

## 验收与成绩

每例应有一次有效 `[kvmem-blend]`：tokens=3264、blocks=102、missing_source_tokens=0，随后正常 query replay；不出现 thinking compaction。180 例独立进程、780 个请求体已审计，全部正常 stop。

分别统计最终回答的完整数字答案覆盖率（逐题平均）及整题数字集合全对率。不能用字符串子串匹配把 `84269240` 当作命中 `8426924`。多名称绑定和完整语义另行核对，当前完整语义复核仍标记 pending。

| 名义长度档 | 覆盖率 | 整题数字集合全对 |
| --- | ---: | ---: |
| 64K | 100% | 60/60，100% |
| 128K | 99.58% | 59/60，98.33% |
| 256K | 97.92% | 58/60，96.67% |

本组重算日志平均 2768.62 ms/例，包括该调用内备份、状态操作和重算，不是整个 answer 请求延迟或 TTFT，不包括未来恢复覆盖的耗时。

结果见实验根目录 `results.json`、`audit.json`、`summary-audited.json`、`template-comparison.md`。原样本 50/50 混合实验属于另一组，输入仍含 xhigh system；不要把它的成绩当作官方模板上的混合对照。
