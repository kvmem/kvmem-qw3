# KVMem 后台回收设计分析

状态：完整设计建议 v4；已实现的保守运行时子集、与本方案的差异和验证见 [GC v1](kvmem_gc_v1.md)。本页尚未标为完成的内容不是当前代码能力。分析日期：2026-09-05。依据当前工作区代码（包含已有未提交修改）、对话《论文差异与定位》，以及进一步明确的两个目标：尽量删除未来复用概率最低的 blocks；管理全部在 CPU 上完成，避免增加 GPU 服务延迟。按用户最新要求，第一版改为无需训练的规则评分器；具体规则见[规则策略 v1](/home/chaidi/qw3/docs/kvmem_rule_gc_policy.md)。文件版本/读取范围与 Mean-K 联合信号仍依据现有 82 条完整实验轨迹；见[轨迹分析](/home/chaidi/qw3/docs/kvmem_file_history_gc_trace_analysis_20260905.md)。

## 结论

建议增加独立的 CPU repository 管理器：按硬保护、规则等级和组内热度选择冷 block，以延迟回收协议管理生命周期，提前补足空闲 slots。推理读取已经发布的不可变目录视图和已准备资源；GC 不进入逐 token / decode batch 关键路径。第一版增加工具操作产生的结构化内容来源信息，不训练判别模型、不引入额外 LLM 语义分类，不保留用于召回的 ghost，不实现 GC miss 自动 re-prefill。

v3 的策略改动是：把“修改后再次读取”作为旧观察可能被替代的软证据，按文件身份、版本和实际返回范围判断；结合已有 Mean-K 的有效评分历史、复用统计及错误/回退状态。新版本并不自动使旧版本无用，重复文本也不代表其上下文化 KV 可以严格互换。文件来源信息缺失或没有新评分时保留不确定性，不能当作零风险。

v4 将训练、概率校准和训练集建设移出第一版交付：用可解释的 0–99 保留分数排序，分数不是概率；Mean-K 使用候选集内百分位和有效观测次数，减少对模型分数尺度的依赖。换模型时重置评分历史并做小规模回归验证，不需要重训规则。

相比 v1 有四项修改：

- 衰减 LFU 保留为基线；实际策略叠加有效检索排名、内容替代关系、回退保护和缺失信息处理。
- 撤销在每个 decode batch 安全点执行回收事务的建议；CPU 管理器拥有冷资源生命周期，前台仅在本来就存在的选择/I/O 边界交接视图和 lease。
- 固定 slot pool、空闲储备和资源隔离提前到真实删除之前，避免已启用 GC 却仍在分配时等待回收。
- GPU index 一致性和 checkpoint continuation 是 hard GC 的前置条件，不能留给后续优化。

能够通过设计约束的是：GC 不发起 CUDA kernel、传输或同步；前台不等待 GC 完成；当前执行工作集不被修改。无法无条件保证的是：仅凭历史预测出真正最低的未来复用概率，或在共享 CPU/DRAM/PCIe/SSD 的机器上获得数学意义的零延迟变化。最终验收应同时报告删除风险和完整服务延迟，不能用“CPU-only”替代性能证据。

需要分别交付两个结果：

1. **可安全删除任意历史 block**：被删 block 不再参与检索，K/V 和缓存资源能够释放或复用；当前已安装工作集不变。
2. **固定容量下长期追加**：历史位置与 metadata/index/storage slot 解耦；消除所有按累计历史长度增长的运行时结构和 `--ctx` 容量耦合。

只有第一个结果不能宣称完全消除了 compaction 或支持无限会话。尤其需要注意现有代码不仅有存储限制，还有累计 token 位置和请求长度限制。

## 已有基础与缺口

| 部分 | 当前代码 | 对 GC 的意义 |
| --- | --- | --- |
| Block metadata | `include/qw3/kvmem_store.hpp:36` | 已有 tier、CPU/NVMe slot、工作集标志、`in_flight`，没有永久删除状态和全局 usage |
| Block 注册 | `src/kvmem_store.cpp:157` | `block_id = blocks_.size()`，ID 等于 vector 下标；不能直接 erase 中间元素或改号 |
| 删除路径 | `src/kvmem_store.cpp:185`、`src/qwen_executor.cpp:11057` | 仅支持回滚尾部；executor 路径会等待 capture/写回并重置索引状态，不适合作为后台 GC |
| Host 热度 | `include/qw3/pinned_kv_tier.hpp:177` | 已有 selection epoch、频率衰减和 admission；只属于 CPU tier，永久回收需要全局统计 |
| 统计入口 | `src/qwen_executor.cpp:15120` 附近 | 在 semantic reselect 中对 remaps 记热度，当前还依赖 CPU tier 存在及优化开关 |
| 已有 CPU 数据 | `src/qwen_executor.cpp:23539` | 普通 Mean-K 最终 block scores 已回传 Host，可从现有结果提取特征，无需为 GC 新增 GPU 打分或 D2H |
| CPU index placement | `src/qwen_executor.cpp:19012`、`:20768` | 表示索引驻 Host；打分仍调用 device kernels、传输及同步，不能在 GC worker 中调用这些函数 |
| NVMe slots | `include/qw3/nvme_kv_tier.hpp:305`、`:471` | 在线非 direct-mapped arena 已能释放并复用任意 block slot |
| Raw-K NVMe | `src/qwen_executor.cpp:9937` 附近 | 在线 raw-K 已按 block 独立存储，main/MTP 共用一条记录；archive 仍按 chunk direct mapping |
| Raw-K Host | `src/qwen_executor.hpp:1193` 附近 | 仍按默认 2048-token chunk 分配，多个 block 共享一块 Host allocation |
| Host V | `src/qwen_executor.cpp:2866` | sparse slab 仅在最后一个 slot 释放后归还实际内存 |
| 写回完成 | `src/qwen_executor.cpp:3167`、`:12607` | completion 会重新发布 backed/clean 元数据，必须防止旧写回使已删 block 再次有效 |
| 检索索引 | `src/qwen_executor.cpp:17215` 附近 | 固定层 stride 仍按 `kv_ctx_size_ / block_tokens` 设置，索引与历史 block 下标绑定 |
| 容量校验 | `src/qwen_executor.cpp:9393`、`:10488` 附近 | 按整个 `--ctx` 保留 raw-K 和 V 容量，尚无 GC 下的容量 admission |
| 会话长度 | `src/qwen_native_backend.cpp:3950` | 累计 prompt 超过 `ctx_size` 直接报错；释放 KV 不会增加这项额度 |

当前的 CPU eviction、raw-K chunk eviction 和 stage-out 大多是层间移动或 clean cache 丢弃。它们保存可再次 stage-in 的权威副本，不能等同于永久删除。

## Usage：独立于物理访问与重选次数

在 repository 层维护一份统计，CPU admission 可以读取这份统计。不要把 `attn_score` 直接当作未来概率：它表示当前 query 的相关性；可以利用现有 Host score 提取相对排名及距入选边界的距离。不要用 NVMe read/write/touch 计数：预取、写回和纯搬运会污染实际复用频率。

建议字段：

```cpp
struct BlockUsage {
    uint64_t birth_step;
    uint64_t last_hit_step;
    uint64_t last_decay_step;
    uint64_t last_counted_step;
    float decayed_hits;
};

enum class BlockLife { Live, RetirePending, Free };
enum class RetentionClass { Young, Probation, Protected };
```

物理生命周期与策略等级分开。等级可以由统计及策略推导，不必同时维护多个复杂队列。usage 由 CPU 管理器消费事件后单写；资源 lease / generation 的发布使用单独的并发协议，不能用异步热度事件代替内存安全保护。

一次 hit 定义为：block 进入实际用于执行的历史工作集；同一外部 agent/model step 最多累计一次。即使没有发生 stage-in，持续留在 GPU 的 block 仍要记 hit。新追加的 block 先受 young 保护，不因首次写入或 pressure-prefill 而增加复用次数。

```text
F(t) = F(last_decay_step) * 2^(-(t - last_decay_step) / half_life)
hit: F(t) += 1，更新 last_hit_step / last_counted_step
```

step 从 backend 的真实执行事务边界提供，不能简单使用 HTTP 请求次数（prefill-only append 可能被分片），也不能直接把每次 `kvmem_reselect()` 当作一个 agent step。实现去重，避免 query replay、私有查询和同一轮多次重选增加虚假热度。

衰减 LFU 是需要保留的对照策略。规则 v1 给出可复现的 half-life / protected fraction 起始值，用小规模 shadow 和质量回归检查；这些不是已验证最优值。不同任务的复用间隔可能相差很大；仅按频率和年龄删，可能误伤周期性复用的 block。

周期线程使用 wall clock 唤醒，但推荐第一版以 step 时间老化。用户暂停数小时不应等价于执行了数百轮；需要按实际空闲时间删除时，另加显式 idle TTL。

## 决策目标与无需训练的规则排序

先定义可观测的代理目标：

```text
p_H(b,t) = P(block b 在未来 H 个有效 agent steps 内再次被召回 | 截至 t 的历史)
```

该概率只用于说明理想目标，规则 v1 不估计或输出它。对永久删除，不能只评估很短的 H；shadow 同时报告短期、长期 horizon（例如 8 / 32 steps）及完整任务结束前的后续复用。没有被 selector 选中，不等价于内容对任务无用，因此还需端到端任务质量验证。

使用少量 CPU 统计与工具来源信息；其中全局 usage、评分有效性历史和来源映射仍需补齐：

| 特征 | 作用 |
| --- | --- |
| 最近复用距今、创建年龄 | 区分刚产生、刚用过、长期闲置 |
| 衰减 hit frequency | 避免频繁使用的内容仅因一段时间未访问就快速降级 |
| 最近连续未命中的有效检索机会 | 只有被允许参与且实际被评估，未选中才构成负证据 |
| 现有 score 的排名、距 top-k 阈值的距离、趋势 | 区分持续垫底和几乎入选；避免跨 query 比较未经归一化的原始分数 |
| 内容来源、被后续观察覆盖的比例、版本关系 | 区分同一内容的重复观察、局部重读和真正的新版本；未知来源不推定被替代 |
| 最近修改的验证状态、回退/诊断关联 | 提高未解决错误相关的旧片段、diff 和失败证据的保留优先级 |
| 观察样本量、事件缺失标志 | 避免将缺少观测误判为低概率 |

同一 agent step 内的相关重选需要去重/降权；forced sink/recent、all-fit 和 pressure selection 单独标注，不能制造虚假的独立命中或负样本。原始 score / rank 不存在或语义不兼容时，使用缺失特征标志，不为 GC 新增 GPU profiling。

第一版采用五个规则等级，先判断硬保护，再按下列优先级赋级：有预算的回退/错误证据保护（L=4），刚产生/刚使用/近期高分或高频（L=3），确认有 live 替代者的重复冷内容（L=0），验证后已被覆盖或持续有效低分的冷内容（L=1），其余普通/信息不足的冷历史（L=2）。完整条件与起始参数见规则策略 v1。

```text
heat = max(recency_heat, decayed_frequency_heat, valid_recent_score_heat)
retention_score = 20 * L + 19 * heat
```

heat 的各项在 0–1 内，分数越低越先回收。缺少有效 score 时不计算该项，并禁止凭“低分”进入 L=1；文件来源不明时也不能假设内容已经被替代。每个等级的分数范围不重叠，时间老化不会直接抵消回退保护。硬 lease 不参与评分，不能因分数低就释放仍被引用的资源。

以 ready-slot 目标决定回收量，正常先从 L=0/1/2 取足资源；保护集合超预算和储备不足时按明示的软保护降级/admission 路径处理。限制每批扫描时间、回收字节和触及的保护等级，记录原因。不把规则分数解释成百分比，也不计算 `sum(score)` 作为整批删除风险。实际误删和额外重读成本通过 shadow 与端到端实验评估。

在有空闲储备时增加一个短暂的 `Candidate` 阶段：保持原 KV 和检索资格，起始观察 2 个有效 agent steps，提交前仍满足对应规则才 hard-delete；重新入选或接近阈值即撤销候选。观察期不制造新的低分次数，也不要求一定发生新的 scorer。它不保留删除后的 ghost，不增加额外查询。无新观察时，wall clock 经过不能算新的“不会再用”证据。Candidate 占用必须计入容量，也不能无限延期。

无需额外 LLM、embedding、GPU attention trace 或在线训练。规则沿用统一配置，换模型/scorer 时清空不兼容评分历史；相对排名不能消除所有模型间行为差异，因此仍需小规模 shadow/质量回归。

## 内容来源：按观察片段判断替代关系

现有轨迹包含 285 次显式成功 Read → 同文件 Edit/Write → Read，涉及 63 个任务；264 次（92.6%）前后请求的读取范围不同。这不等于 264 次完全没有覆盖，但足以否定“同文件重读即覆盖全部旧读取”的规则。另有修改后读取 Git 原始基线，以及类型检查失败后恢复旧函数签名的实例。计数及边界见轨迹分析。

建议在已有工具请求/返回处理处产生 CPU side metadata，并异步交给 repository：

```text
Observation {
    observation_id, tool_call_id, append_generation;
    object_identity;       // workspace/repository + path；身份不明则 unknown
    observed_revision;     // 已知版本或局部观察版本，不能冒充完整文件版本
    returned_spans;        // 实际返回范围、截断标志、内容摘要
    token_spans;           // 已有 tokenization 中建立的历史 token 区间
    relation;              // duplicate / overlaps / supersedes_candidate / unknown
    validation_state;     // unknown / pending / failed / passed（含验证范围）
}
```

只使用工具已经返回的字节，不额外读取整个工作区文件。正文摘要在后台计算；利用已有 span 到 token 的映射机制扩展，不在每次请求重扫全部历史。当前 server 的 retrieval-group span 映射可作为实现参考，但并未提供上述文件版本语义，也不应因此强制开启整 message 的 group selection。

显式 Edit 的 old/new 片段可建立局部变更关系；任意 Bash、外部写入、分支切换或 rename 不能仅靠路径和字符串猜测出准确版本。工具包装层没有提供可靠信息时，关系标为 unknown。行号随编辑移动，需要内容/变更映射确认实际覆盖，不能比较两个行号区间就宣称版本替代。

候选优先级建议：

| 已知情况 | 处理 |
| --- | --- |
| 同一对象/版本、同一返回范围、内容相同，并且仍有 live 替代观察 | 较强冗余信号；旧观察可优先进入 Candidate，仍考虑上下文角色和实际检索证据 |
| 新读取仅覆盖旧读取的一部分 | 仅给确认覆盖的片段增加冗余特征；不扩展到整文件或整 message |
| 修改后读取新版本，尚未完成验证 | 标注可能被替代，保留受影响的旧片段/已有 diff 的较高优先级 |
| 修改后失败、回退、比较基线 | 提高关联旧片段、失败证据、原始基线及已有 diff 的优先级；不只看最近文件版本 |
| 验证通过且后续正常推进 | 逐步降低已覆盖旧片段的保护；测试通过也不是语义等价或永不回看的证明 |
| 未知版本、截断、范围映射不确定 | 不用内容规则推断可删，回退到通用规则等级/热度并保留缺失标志 |

不应为每个版本永久保留全文件。原始基线、最近经验证前驱、未解决变更的相关片段可以成为有字节预算的软保护集合；优先保护已有 Edit 中的 old/new、诊断输出及必要源码 spans。没有这些已存在的内容时，GC 不现场生成摘要或 diff 后要求模型 prefill。硬 lease 与语义软保护仍须分开，后者可以在容量压力下按风险降级。

物理删除仍落在 block 上。一个 block 同时包含旧文件内容、用户要求或诊断结论时，不能因为部分 tokens 被覆盖就整块降级；第一版只对整块内容均可归因且满足条件的候选采用强冗余信号，混合块保守处理。多个重复观察需要联合约束：至少保留所依赖的替代观察及其必要结构，防止一批删除把所有副本同时清掉。替代者受保护也必须计入容量。相同输出文本在不同历史位置有不同 KV，内容去重只是策略特征，不能当成精确 KV 去重。

## Mean-K：复用相关性证据，不能充当删除判决

当前 scorer 使用 query 与 block mean-key 的相似度，并在 layer/query/head 内对 blocks 做 softmax 再聚合。它体现模型表征下对当前 query 的相关性，不直接表示未来访问概率、代码正确性，或删除后的任务损失。相近的两个文件版本可能都高分；一个关键错误行也可能被块均值稀释。这些是需要验证的风险，不是本次轨迹已经测出的结论。

推荐使用现有 Host 结果的候选集内排名、距选择边界的距离、最近高分与近几次有效评分趋势，并记录 scorer/query 类型与 eligible 集合。一次当前低分不足以永久删除；对于同一来源片段跨多个 block 的保护，可使用已有分数的高分片段/max 或上分位数作为保守特征，避免再次按整个文件平均而掩盖局部证据。尚无离线结果支持某组固定权重最优。

本次 82 个任务的 `metrics.json` 中，semantic reselection 次数中位数是 1，23 个为 0、26 个为 1。该指标是日志中的语义重选次数，不是逐 block 的完整 score history；也不能把 passthrough、显式裁剪或工作集沿用计作新的语义负证据。没有足够评分时，不能依赖“连续多次低分”作为所有回收的必要条件；使用内容规则、L=2 的通用热度排序和明确的容量预算，否则策略可能完全没有可回收对象。反过来，缺少分数也不能直接降级为低风险。

第一阶段仅在本来就发生的 scorer 完成后，传递现有 CPU score/rank 摘要及 block/observation 关联；不为 GC 增加 query、kernel、D2H 或周期性重打分。已有 score dump 不是每步完整事件流，本次选定样本未找到与文件 spans 对齐的逐 block 分数历史，因此目前无法报告“score 能否识别错误修改”的相关性，更不能估算该规则的质量收益。离线 shadow 保留完整检索候选，真实删除后的不可见性不能充当低风险标签。

## 并发模型：CPU 独立管理，推理读取不可变视图

当前 `KvMemStore` 和 `PinnedKvTier` 没有通用并发保护。NVMe metadata mutex 也只保护映射，不保护已提交的 positional I/O 生命周期。因此不建议让新增线程直接扫描、修改 executor 内的 vectors/maps。

需要先把冷 repository 的 lifecycle、目录和空闲资源管理从 executor 中抽出。只在现有容器外面加一个后台线程，无法同时满足无数据竞争和前台不等待。建议一个服务级有界后台 worker，通过明确注册/注销管理 executor 生命周期：

```text
推理：在已有选择/I/O 边界发布有界事件，获取已发布视图和已准备资源
    ↓
CPU 管理器：更新规则等级和热度、选择候选、发布下一目录版本
    ↓
退休队列：等待旧视图读者及实际资源引用自然完成
    ↓
CPU 回收并补充 ready-slot pool，前台不执行 GC 事务
```

目录用分段不可变快照或等价的版本发布机制，避免每次删除复制整个 repository。新读者看新目录；已有 scorer 继续用与其索引完全一致的旧版本。通过 read lease 保护旧视图，在选择完成后先转移为 selected-block / I/O lease，再释放视图 lease。decode 仅持有其实际工作集引用，不持有整个历史目录的长期 lease。

借鉴 [RCU 的 removal / reclamation 分离](https://www.kernel.org/doc/html/v6.10/RCU/whatisRCU.html)：发布新可见性和释放旧资源分开。但 GPU kernel、DMA、positional I/O 不属于 CPU 读侧临界区，必须通过实际 fence/future 完成及长期 lease 保护。不能只等待 CPU epoch 就释放这些资源。

后台退休前复核状态、usage 事件处理水位和 generation。已有旧视图仍可完成其合法选择；若退休中的候选被该视图选中，则获取长期 lease，取消该次实际删除并在后续目录恢复可见性。它此时尚未进入 Free，不需要重建 payload。不得在 scorer 返回后直接丢弃已选结果而要求前台重试。

固定大小事件队列满时不能阻塞推理，也不能丢事件后继续把 block 当冷数据删除：记录观测缺口，保守暂停受影响 epoch 的退休，待版本/保护集合重新一致。此情况也必须计入 reserve 容量规划。

后台不调用 `DeviceBackend`，不新增 CUDA event 查询、GPU 同步或 tensor 析构；通过现有完成路径发布的 CPU 状态了解资源何时可回收。前台仍有少量视图/lease/事件发布成本，需要测量；CPU-only 不等于这部分成本严格为零。

需要保护的引用包括：

- 当前实际执行工作集，包括 all-fit 时没有显式 `in_working_set` 的 identity window。
- 正在构建的下一工作集、query prefetch、CPU gather、NVMe read、尚未完成的 H2D。
- main/MTP D2H、raw-K writeback chunk、V 写回，以及 index capture/scoring generation。
- 未封口的 append/decode tail、query/replay mandatory 区间、sink/recent 保留区间。
- 仍承诺可恢复的 checkpoint 所需资源。

第一版仅回收已离开 GPU、无在途修改的 sealed blocks。lease 获取与退休必须采用无竞争的协议（例如版本读保护加长期引用转移）；executor 独占使用中的记录，后台只改冷资源和新目录，不能异步读写同一个普通 struct。

工作集切换时先获取新集合引用，再安装新窗口，待旧窗口最后一个消费者完成后释放旧引用。不能提前 unpin 旧集合。

之前讨论的“整个 decode 中工作集不变”只适用于相应运行模式。当前支持 interval 重选，MTP 路径也有 mid-decode guided refresh（`src/qwen_native_backend.cpp:16506`）。因此应以**工作集 generation 的有效期**保护资源。GC 不改变当前 generation 的执行输入；下一次重选及其后的输出允许受到删除影响。

## 删除事务与物理回收

```text
Live → RetirePending：禁止获取新引用，退出可检索目录
    → 等最后的既有引用/transfer fence 完成
    → 回收资源并发布 free slot
    → Free
```

CPU 管理器优先回收已无引用的 block；旧读者、prefetch 或 checkpoint 尚在使用时延迟释放。所有异步结果都携带 `(repository_epoch, slot, generation)`；检查失败的旧结果不能发布 metadata。**generation 检查不能替代 I/O 等待**：旧写请求结束之前绝不能把同一 NVMe slot 交给新 block，否则数据已被覆盖，即便拒绝旧 completion 也无法修复。

一个删除事务必须覆盖：

1. main raw-K 与可选 MTP raw-K 的权威记录。
2. main/MTP V spill record、Host clean copy，以及残余 raw-K GPU cache 映射。
3. Mean-K/prototype index entry 与所有查询结果中的旧 handle。
4. repository usage、tier heat、位置目录、valid/backed 位图。

`PinnedKvTier::release_block()` 故意保留 `heat_` 以便再次 admission；永久删除需要单独的 `forget_block()` 语义，避免热度表泄漏或新内容继承旧热度。

Raw-K 的 chunk 共享要特别处理：NVMe block slot 可独立回收；Host 中删一个 block 只回收该 block 的有效性，不能直接 free 包含其他 live block 的 chunk。chunk 全空且无引用时才释放；NVMe 模式也可以驱逐整个已干净的 CPU cache chunk，因为其他 live block 仍有独立权威记录。重新加载和批量写回都必须跳过已删 block。

无需为了删除而重写/压缩整个 SSD 文件。优先把固定 slot 退回 pool 后复用；服务热路径期间不做 file compaction、hole punching/TRIM、大块清零、pinned allocation/free 或大规模搬运。需要归还 OS 的 slab/file 资源作为独立维护策略，并单独测量性能。占用指标应区分 live bytes、reserved bytes、已分配 slab/file bytes，不能把逻辑删除字节数直接当 RSS 下降量。

## 检索必须完整支持空洞

短期实现采用稳定历史 ID 加 live bit，保留既有位置计算；不能 `erase(blocks_.begin() + id)`。这适合验证 hard eviction，但仅是过渡结构。

所有 selector 都要过滤 dead blocks：普通 top-k、quota、recency、pressure、semantic group、mandatory selection、all-fit 快路径、`set_selection()`、checkpoint 的 identity fallback。真正的 mandatory block 缺失应返回清晰的一致性错误，不能静默缩减必选集合。

仅将删除项最终分数设为 0 或负无穷不够。当前 Mean-K scorer 对每个 layer/query/head 做 softmax 再聚合，已删除项若仍参与归一化，会改变其余项的相对权重；置零向量也仍会产生非零概率。应在 softmax 前 mask，或只扫描 live index slot。空候选集合要有明确处理，避免 NaN。

GC 只修改 CPU catalog 和 slot 状态，不主动更新正在运行的 GPU index/mask。正常的下一次 retrieval 获取准备好的 catalog 版本：Host-resident index 路径在原有 staging 时打包 live entries；GPU-resident index 路径需要在正常 retrieval 的元数据发布中安装对应 live-slot/mask，或使用预先准备的版本。scoring 使用的 payload、handle table、mask 必须属于同一版本。

这意味着可以做到“不为 GC 发起 GPU 工作”，但不能保证“GPU 检索永远无需感知任何删除”：现有 dense index 下仍需改 scorer 输入接口、mask 或间接寻址，数据上传/分支的增量成本应计入 TTFT。若连这部分 GPU 路径变化也禁止，则必须另做真正 CPU-only 的检索并评估其延迟；当前 `index_placement=CPU` 并不满足这一含义。不能通过保留 stale softmax 项、事后过滤结果规避一致性。

GPU index slot 的旧字节可保留在预分配 arena 中，但在旧 reader 结束前不可覆盖；后续新 block 的正常 index capture 才复用这个 slot。无需 GC 发起 cudaFree 或 GPU compact。

语义 group 现有契约是整组选择。第一版使用物理 block 策略并显式限制不支持的 group 路径，后续再定义 group 含已删 block 时的资格规则，不能悄悄变为部分物化。PerToken/实验性 DeltaNet scorer 也应显式验证支持能力，避免退化路径重新访问已删 KV。

## 固定容量：按 live slot 建池

当前 raw-K arena、部分 metadata 和 Mean-K stride 均依赖 `--ctx`。固定 slot pool 和 live-block 上限 P 应在真实删除前建立；NVMe 版可按每个 block 的 raw-K（含 MTP）加 V 实际记录大小计算 P，分别建立 raw-K/V pools，Host cache、index 和 metadata 各有预算。物理预算还须容纳 retired-but-not-free、Candidate、在途分配和空闲 reserve，不能只给目录内可检索 block 计费。

长期结构建议：

```text
单调历史位置 / logical block identity
    → 只包含 live blocks 的目录
    → BlockHandle { metadata_slot, generation }
    → raw_k_slot / v_slot / index_slot / optional cache slot
```

metadata slot 可复用，原始历史顺序不可由可复用 slot 编号推断。选择后按原始位置排序，再沿用当前 window remap；检索器读取 index slot 并返回 handle。

固定 Mean-K 可采用 slot mask 或 live-slot 列表；Adaptive 的 variable prototype arena 需要额外 allocator/分段回收机制。只做 tombstone 或把 prototype 数置零，不会回收 append-only arena 容量。epoch/live-slot 表的更新也必须保持与在途 scorer 的生命周期一致。

GPU 物理 page pool 已有边界，但原始位置页表、raw chunk 目录、valid 位图、usage 表和 token checkpoint 副本仍须审计并改为 sparse/bounded 结构。只改 block vector 不足以保证整个服务内存稳定。

累计位置建议逐步提升到 64 位；GPU 窗口/slot 仍可用 32 位。服务端的 `--ctx` 校验、输出余量、session token 保存与前缀匹配策略需要配套调整。优先用现有增量 session API 验证持续 append，再扩展每轮发送完整 transcript 的兼容路径。

## 触发、预算与无 victim 的行为

建议事件驱动（完成 append / selection / release / occupancy 跨水位）加低频兜底唤醒，并限制每批扫描量、CPU 时间和内存流量。worker 使用 occupancy 增量，不周期性复制整套 KV 或全量历史 metadata。

85% / 70% 仅作为对照参数；正式触发优先看“距离耗尽还有多长时间”和回收完成需要多久。必须分别检查 raw-K/V arena、Host 实际分配、index、metadata。不能只检查去重后的 KV 总量，也不能把 CPU clean cache 满了直接等同于需要丢历史：先回收可再生 cache，再按权威存储压力执行 hard eviction。

分配前预留 upcoming append、writeback in-flight 与下一工作集所需额度。触发依据应包含 `used + reserved`；某个 arena 即将不足，即使全局总量未过高水位，也要触发。分别为每个必需资源维护 ready pool，成套供给新 block。可用以下关系规划储备：

```text
free_reserve >= peak_net_allocation_rate * reclamation_lag + admitted_burst
```

`reclamation_lag` 包含候选观察期、旧视图和最长允许 lease 生命周期、I/O 完成与 worker 调度延迟。无上限 checkpoint 不能放进有限 reserve 的保证。物理回收速度也必须能够覆盖长期净分配速度；退休数量不能当成已释放数量。

由 admission 限定每轮 burst、最大工作集、在途 I/O 和 checkpoint 预算。在满足这些假设的已接纳负载下，前台从 ready pool 取资源，无需等 GC。若负载超过假设，明确暂停接纳/容量拒绝并统计服务影响；不能隐藏在分配路径里的等待，也不能声称任意输入下零延迟。

Young/Protected 是软保护；当前消费者、未提交尾部、必要 sink/query 和在途 I/O 是硬保护。巨大初次 prefill 可能在一个 step 内把所有 block 都变成 Young，必须规定 grace 的字节上限或压力下对最老 sealed/unpinned blocks 的降级，否则“8 steps 内绝不删”会阻止任何回收。

当硬保护资源加本次最小 append 已超过容量，GC 不可能创造足够空间：在接纳边界明确处理。有限内存、任意长度的不可回收集合、任意突发输入和绝不等待不能同时保证。

## CPU 资源隔离与 latency 验收

GC 使用独立的低优先级 CPU worker，绑定预留核心，不与 tokenization、GPU launch、prefetch gather 和写回 helper 共用同步线程池。现有 `KvmemCpuWorkerPool::run()` 会由调用者参与并等待 helpers（`src/qwen_executor.cpp:220` 附近），不适合直接承担异步 GC。

小 metadata 尽量在 GC 的 NUMA/allocator 区域组织；不为了 NUMA 隔离迁移大块 KV。按扫描条目、字节数和时间限额运行，不读取 raw-K/V payload 来估计重要性。即使仅使用 metadata，也需测量 DRAM/LLC 竞争。GC 落后时优先消耗 reserve，不能由 decode 线程帮忙扫描或析构。

需同时记录 GPU kernel / stream 时间、host launch gap、TTFT、inter-token p95/p99、吞吐、prefetch 尾延迟、CPU/DRAM/SSD 使用、ready-pool 低水位和分配等待。仅验证 GPU kernel 数量不变，无法排除 CPU launch 被拖慢；仅验证 decode TBT 不变，也可能把成本转移到下一轮 TTFT。

隔离实验使用固定工作集/输入轨迹，比较 GC off、仅元数据维护、后台完整回收三种负载，避免因删除导致 GPU 实际工作减少而掩盖管理开销。再用真实策略做端到端质量/延迟 A/B。非劣效界限应预先确定，并报告多次运行的置信区间，当前没有实测数字。

## Checkpoint、归档与恢复

`StateSnapshot` 保存 position、selection generation 和 recurrent/window 状态，但不自动拥有全部历史 payload（`src/qwen_executor.hpp:54`）。恢复会回写 selection generation 并截断尾部（`src/qwen_executor.cpp:8171`）；GC generation 必须独立且不可随这种 rollback 倒退。

普通会话 continuation 应基于 live repository 恢复，不得因为 prefix 含 GC 空洞就重新注册/重建全部历史；也不要每次 GC 都整体失效 warm checkpoint，否则会引发本来想避免的 full re-prefill。checkpoint 保存实际必要的资源 lease；支持的冻结分支需要显式保留其依赖并计入预算，过期 checkpoint 及时释放。

`kvmem_content_index_covers()` 当前以连续 token coverage 判断兼容性，需增加 GC-aware 的“活跃目录覆盖”语义。`kvmem_truncate_to()` 的永久删除状态不应被后续 index rebuild 或 restore 复活。

Archive 是 sealed、可共享、direct-mapped 的持久数据，第一版 hard GC 应明确不支持 archive build/attach；仅删 session overlay 不等于回收 archive 本体。后续若支持，需要单独定义持久删除和共享读者语义。

删除的是可寻址的标准 attention KV；已形成的 DeltaNet recurrent state 保留其既有计算结果。这符合缓存管理目标，不代表从模型状态中彻底擦除该信息。

## 建议交付顺序与验收

1. **实现纯 CPU 规则函数与必要事件**：固定规则版本、起始参数、分级/打分语义和原因码。记录真实 selection reason、eligible exposure、已有 Host scores、资源和完成事件；增加工具观察的来源/版本/返回范围、验证状态，以及 token/block 关联。使用有界队列与日志，不建设训练集或训练流程。当前离线文件操作统计只验证候选信号存在，尚不能评价 block 删除收益。
2. **规则验证与 CPU Shadow**：先验证硬保护优先、缺失不等于低分、同分处理、部分覆盖、替代者保留和评分 epoch 隔离等不变量。以衰减 LFU 为基线，比较加 Mean-K 规则、加内容规则和二者联合；测固定容量下 regret、回收量、保护降级和 Candidate 取消。覆盖回退/基线对照、部分重读、重复观察、评分稀疏任务；使用同一套规则做跨模型小规模回归，不训练参数，不为 shadow 再跑一次 GPU scorer。
3. **先实现隔离与容量基础**：CPU repository ownership、视图/lease/generation、固定 slot pool、ready reserve、预算/admission、GPU scorer 版本接口、checkpoint continuation。以延迟 I/O 和空池边界测试证明内存安全与前台无 GC 等待。
4. **启用 hard GC 并验收两项目标**：优先单 executor、在线 immutable raw-K NVMe、已完成 live-view 支持的 Mean-K/SubBlockMeanK；先做固定执行轨迹的隔离测试，再做真实 agent A/B。显式拒绝尚未验证的 archive、group、其他 scorer/执行组合。
5. **扩展长期容量和完整模式**：完成 Adaptive variable index、sparse position/raw directories、64-bit 历史位置和累计 `--ctx` 解耦。metadata/index slot 复用已在前置基础阶段处理，不再把它当 hard GC 后才考虑的可选优化。

推荐代码划分：`src/kvmem_gc_policy.hpp` 放纯 CPU 规则分级、热度评分、原因码和预算策略；新增 `KvMemRepository` 抽离冷资源 lifecycle、目录、retire queue 和 ready pools；`KvMemStore` 保留前台选择与窗口计划；`QwenExecutor` 使用 view/lease 并发布已有 capture/I/O 的完成事件；backend 管 step、checkpoint、会话、admission 和 worker 注册。GC 模块不依赖 `DeviceBackend`。

关键验证：

- Shadow 下统计“模拟删除后未来 k steps 又入选”的比例（regret）及字节量，评估不同容量和阈值。真实 hard-delete 后观测不到该 block 的后续分数，不能把“不再被选中”当作零 regret。没有完整未来 horizon 的轨迹是删失样本，不能直接标成不会复用；确定 task 结束则按事先定义的任务内目标处理。
- 用有未来访问轨迹的离线策略作为参照，明确固定轨迹 replay 的局限：真实 agent 会因删除改变行为，无法用离线命中率取代端到端成功率/额外工具调用。规则分数不代表概率，不能宣称逐 block 最优保证。
- 删除中间 block 后继续 append，所有 selector 与回滚均不复活；同一 block 的 main/MTP K/V 一起回收。
- 故意延迟 raw-K 写回、V 写回、CPU gather/H2D、index scorer，同时请求 GC 与 slot reuse，验证既不提前复用，也不接受 stale completion。
- 当前工作集固定时比较 GC on/off 的 token/logit；跨重选只做质量/性能评估，不要求输出一致。
- 小容量、多轮追加远超 live capacity，Host 实际分配、NVMe 文件/slots、index、metadata 均稳定；同时记录 reclaim、reserved、pending-retire、pin 和 backpressure。
- 用真实 agent 轨迹比较任务成功率、额外工具调用、重读/重跑、TTFT、decode p95/p99、compaction 与隐式 full re-prefill 次数。

本次仅做源码、既有轨迹与设计分析，新增离线统计脚本和分析结果；没有改动运行时代码，也未运行 GPU 性能或 GC 正确性测试。上述阈值、并发预算和收益均须在实现后验证。
