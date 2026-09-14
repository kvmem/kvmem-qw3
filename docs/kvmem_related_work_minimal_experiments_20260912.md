# KVMem：低成本相关工作对比方案

日期：2026-09-12。本文是实验设计，没有启动实验或修改推理代码。

> 优先级已根据用户补充调整：主实验应首先直接比较具名现有方法，本文件中的机制消融降为辅助。当前建议见 [现有工作直接对比方案](/home/chaidi/qw3/docs/kvmem_existing_work_comparison_plan_20260912.md)。

## 优先回答的问题

在相同模型、服务实现和资源条件下，关闭摘要压缩并使用普通 KV 检索，是否已经足以取得当前收益？模型生成的检索 query 和跨轮工作集复用各自贡献多少？

最低成本方案是同框架机制对照，加少量完整 agent 轨迹。此类实验属于消融与受控参考策略，不能标成 InfLLM、LouisKV 或其他原版系统的复现，也不能据此宣称超过这些系统的性能。

## 与相关工作的对应关系

| 工作 | 最值得比较的机制 | 本轮低成本实验能回答什么 |
|---|---|---|
| InfLLM、RetrievalAttention、InfiniGen | 从模型内部信号选择历史 KV，配合分层存储 | 直接 Q 检索能否解释已有收益；这不是三套原版算法的性能比较 |
| LouisKV、FlexiCache | 利用时间局部性减少检索与搬运 | 在同一检索实现中扫描刷新频率，测量跨请求复用的成本和风险；请求间隔不等于其逐层、逐 token 或 head 自适应算法 |
| Practical Online KV Cache Compaction for LLM Agents | proxy Q 来源、延迟决策、压缩后原 KV 不可召回 | 先比较 query 来源；永久丢弃与可召回历史的对照可后补，不能把近期窗口策略冒充其 TE/AM 算法 |

InfLLM 公开 patch 当前列出 Llama、Mistral、Qwen2，没有当前 Qwen3.8 混合注意力和视觉路径的直接适配。RetrievalAttention 当前仓库的说明还包含 RetroInfer，必须固定具体算法和版本。原版移植会涉及模型结构、位置处理、缓存布局、量化、MTP 和多模态接口，不能视为替换一个 top-k 函数。

在线 agent compaction 论文已经覆盖混合注意力模型，并只压缩 full-attention 层；因此不能笼统声称相关工作都不支持混合架构。

来源：[InfLLM patch](https://github.com/thunlp/InfLLM/blob/main/inf_llm/utils/patch.py)、[RetrievalAttention 仓库](https://github.com/microsoft/RetrievalAttention)、[InfiniGen 仓库](https://github.com/snu-comparch/InfiniGen)、[LouisKV §4](https://arxiv.org/html/2510.11292v1)、[FlexiCache 作者实现](https://github.com/NazmulTakbir/FlexiCache)、[在线 agent compaction §3](https://arxiv.org/html/2608.00902v1)。

## 最小方法集合

| 名称 | 策略 | 用途 |
|---|---|---|
| Native compaction | 普通 qw3，Claude Code 原生 auto compaction | 复用已有完整实验，说明应用中的摘要停顿 |
| Recent-window | 保留相同 mandatory/sink 策略和预算，容量压力下选择近期块；不生成 query，不进行历史语义召回 | 检验仅取消摘要并保留近期上下文是否已足够 |
| Native-Q retrieval | CPU 历史、Mean-K 索引、与 KVMem 相同的容量刷新时机和恢复流程；使用当前请求中最近观察/指令的模型 Q，不生成私有文本 query | 检验基础 KV retrieval + offloading 的效果，并分离 guided query 的额外价值 |
| KVMem guided | 当前完整 guided-query 版本，保留完整 query replay | 现有系统 |

Native-Q 基线有意继承 KVMem 的跨轮复用与状态管理，构成一个较强的 query 来源消融；它不能单独证明调度创新。最近观察的 Q 范围需要在实验前固定并记录，不能一直拿任务开始时的旧 user query 冒充当前 query，也不能读取未来工具结果或正确答案。

当前 `kvmem_harness_refresh_decision` 在关闭 guided 时，会让同一 user query 下的工具历史增长进入容量管理路径，不再因增长自动发起语义重选。因此，现有 guided-off 不等于 Native-Q retrieval。实现时应解耦“何时刷新”与“Q 从哪里来”，复用现有 scorer、CPU KV、checkpoint、re-RoPE、query replay；不重写检索内核。Recent-window 也须用日志验证实际只发生容量管理，不能只根据开关名称推断。

代码依据：[刷新策略](/home/chaidi/qw3/src/kvmem_refresh_policy.hpp:186)、[服务层刷新决策](/home/chaidi/qw3/src/qw3_server.cpp:3292)。已有显式 force/query-span 接口可辅助重放验证，但 Claude Code 的 Anthropic 路径需要核对元数据映射，不能假设直接转发所有原生字段。

## 第一阶段：先用请求重放检查效率

1. 使用已有 OSWorld 052 归档的前 260 个请求，历史增长到约 563K tokens，覆盖多次维护。再加一条存在多次维护且无基础设施故障的 DeepSWE 文本轨迹；文本轨迹在本机用相同程序和精度重放，不混入原始不同硬件的绝对耗时。
2. 运行 Recent-window、Native-Q retrieval、KVMem guided，共六次顺序重放。第一遍用于发现机制和正确性问题；需要形成稳定速度结论时，再按交错顺序重复，不把单次运行称为稳定加速。
3. 从轨迹开头依时间顺序建立每种方法自己的 KV、索引和 recurrent/MTP 状态。不要把重选前的大 prompt 当作独立冷请求，也不要跨策略复用包含不同选块历史的 checkpoint。真实冷启动和后续异常冷 prefill 分别报告；不能通过强制恢复不满足 LCP 的状态消除它们。
4. 请求历史、截图、采样参数和输出上限固定为归档值。每条下一请求仍使用录制的历史，当前生成不驱动环境，所以这是开放环路测试。输出可能不同，累计 API 时间也受生成 token 数影响；不能把它当成 agent 成功率或同输出的解码加速比。
5. 保留完整 guided 生成和选后 replay；不能通过只生成一个 token、跳过 query replay 等方式低估维护耗时。

记录 API TTFT、完整请求时间、输出 tokens、guide/score/stage/replay 分项、实际活动 KV 大小、GPU/CPU 峰值、重选次数、guided fallback 次数，以及已有日志能覆盖的 cache miss/搬运量。新增计数器应单独验证测量开销。只在原生客户端完整运行里报告观察就绪到下一动作的等待与 summary stall，重放没有该完整客户端区间。

既有 [052 重放结果](/home/chaidi/qw3/benchmark/osworld_v2/claudecode_guided_query_latest_20260910/results.json) 中，260 次请求的累计 API 时间为 guided-off 11.99 分钟、guided-on 12.23 分钟。这只是旧版本重放的排期参考，未包括所有启动与记录耗时；新策略耗时须由首轮实测确认。已有 [重放脚本](/home/chaidi/qw3/benchmark/osworld_v2/claudecode_guided_query_latest_20260910/run.py) 可以复用。

## 第二阶段：最少量的完整任务能力检查

建议先选 012、015、016、018：规则为当前相同最新程序版本、双方原始轨迹都发生过维护的任务中，按 ID 升序取前四个。该选择用于维护机制诊断，不是总体成功率的随机样本；不按得分或 KVMem 是否获胜筛选。

每个任务先补 Recent-window 和 Native-Q 两组，seed=0，原始任务/判分器、500 动作上限和原生任务计时规则不变，共八次新增运行。已有 KVMem 结果只有在推理程序和采样等条件完全一致时才直接复用。若实现新开关需要重编译或改变共有路径，应让完整 KVMem 也使用同一新程序重跑四次，变成十二次。052 是重放校准样本，053 的异常/补跑另行保留，不用得分选择替代轨迹。

这些任务的已有 KVMem 运行时间约为 18、24、16、38 分钟，四条合计约 97 分钟。可先按新增运行需要约 4–8 GPU 小时安排，首个任务后重新估算；新策略可能产生不同步数，若长时间未结束则仍遵守原始停止规则，不能为了赶时间把失败任务截掉。

报告每任务原始官方得分、完整成功与否、动作数、输出量、总时间、观察到首内容的 TTFT，以及观察到下一动作的等待。任务级展示，不用不同轨迹的第 N 步当作同一输入。四题一个 seed 是诊断证据，不能据此证明一般性的能力不下降；若论文要下该结论，需要扩大任务/seed 并预先定义可接受能力差距。

## 第三阶段：只在强调调度贡献时补充

用 Native-Q 检索固定 query 来源，在重放中比较每 1、8、16 次模型请求触发刷新，以及当前容量驱动刷新。容量安全条件可提前触发；不在低于检索预算时人为制造维护。明确单位是模型请求，不是 token 或桌面动作。

这类对照用来判断降低频率的收益、平均活动窗口增长的代价，以及新证据多久能被纳入。它不是 LouisKV/FlexiCache 原版：后者有逐层/逐 head 的策略与缓存优化。若更频繁刷新已经足够快，或当前策略质量不如较短间隔，应如实报告。不要只挑每 token 强制搬运的弱基线。

只有要宣称优于具体检索系统时，才把该系统原实现或经过忠实验证的移植放入具名实验表。必须使用共同支持的模型和精度、匹配物理 KV/索引/缓存预算，保留其 cache-hit、异步重叠、分层策略等优化。若采用算法适配，明确列出偏离原实现的部分；当前框架的参考策略不能冒用原系统名称。

## 保持比较条件一致

- KV 方法统一 CPU 历史层、无 NVMe、GC 关闭；相同模型权重、KV dtype、视觉编码、MTP 和客户端设置，关闭摘要及子代理。普通组保留原生 auto compaction。
- 主实验继续用 128K 选择预算、128K 生成 headroom、block=32、相同 sink/recent 参数。128K 选择预算不等于全程只使用 128K 活动 KV，必须报告平均和峰值实际活动量。不同刷新频率下平均 attention 工作量可能不同。
- 不把 guided 总计与它的生成/状态子项重复相加。发生在首内容之后的 mid-decode refresh，必须在完整动作等待和总时长中评估。
- 维护事件样本数可能很少；给出全部事件、数量、均值/中位数和最大值，避免以少量事件的 p99 推断总体尾延迟。任务平均和请求合并统计分开标注。
- 记录工具基础设施故障并保持两组一致的排除规则；正常模型失败、无维护任务和 guided fallback 不删除。

## 结果如何影响论文定位

- Recent-window 已达到同样质量：应弱化语义检索必要性的论断，当前优势可能主要来自取消摘要和近期上下文保留。
- Native-Q 与 guided 质量相当且更快：guided 适合作为可选能力增强，不能声称额外 query 是必要条件。
- Guided 的任务表现更好，额外维护仍明显低于摘要停顿：支持为未来任务需求生成 query 的收益/成本取舍，但仍不等于优于 InfLLM 等原版系统。
- 只有频率消融也有收益，才进一步强调跨轮调度的独立贡献。

建议优先实施前两阶段。这样新增的主要工作是一个清晰的 query 来源开关和少量实验，论文可以补上“普通 KV 检索是否已经足够”的直接证据；具名原版系统比较作为不同成本等级单独安排。
