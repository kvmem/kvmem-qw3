# KVMem 图评分与压力重选 V2 实现交接

日期：2026-10-18。目标仓库：`/home/chaidi/kvmem-freetoken/kvmem-vllm`。qw3 参考实现：`/home/chaidi/qw3`。

请在目标仓库实现本说明，包括代码、配置、测试、使用文档和验证记录。目标是：修正 Mean-K 的 mandatory 评分口径，以单一传播系数增强 attention 图传播，补齐真实消息／工具输出边界，并用累计 attention 驱动的图评分替代旧压力排名。

本说明检查的目标 HEAD 为 `c14c6bf`，工作区当时干净。实现开始时重新核对最新提交和工作区，不假设它仍停留在这个版本。该提交已把密集 attention 图聚合移到 native CPU；必须保留这项优化。此前独立副本 `/home/chaidi/qw3/.worktrees/kvmem-qw3-alignment-cz15d7a5` 只能作为部分实现参考，不能整分支合并或整文件覆盖当前代码。

## 1. 已确定的范围

两种评分入口共用一套图传播函数：

| 场景 | 起始分数 | 普通候选范围 | 最终选块 |
|---|---|---|---|
| Query 检索 | 当前 Query 的 Mean-K | 当前 accepted boundary 内的完整历史索引 | 原生 Query picker |
| Graph pressure | 原压力策略的累计 output/prefill attention | 当前逻辑工作窗口；恢复时采用 checkpoint 保存的工作窗口 | 原生 pressure picker |

本次不新增近期 Q 缓存，不把近期 prefill Q 当伪 Query，不增加额外模型 forward，不进行 K 聚类、embedding 聚类、通用句子切分、多跳迭代或依赖闭包。attention 仍用于建边，但 pressure 不再直接按旧累计分数选块，而是先传播一次。

三个修复的准确含义：

1. mandatory 额外必留块不能在 Mean-K softmax 前被随意排除；评分域与图传播域分开。
2. attention 的保留边在本次有效候选中做行归一化，避免 top-k 剩余绝对质量再次压低传播力度；整个 V2 只开放一个新的连续调节量 `graph_alpha`。
3. 自动或通过受支持的结构化调用方生成真实 message/tool-result token spans，使 locality 使用真实边界。这里实现的是有消息边界的相邻关系，不宣称已实现一般语义相似性图。裸 token 无结构信息时仍必须诚实 fallback。

## 2. Mean-K 对齐：区分评分域与传播域

### 2.1 原生评分域

当前入口为 `src/vllm_kvmem/authority.py::RequestAuthority.select_for_queries()`。它用一个 Python `eligible` 同时排除 sink/recent/mandatory，再传入 Mean-K，这与 qw3 的原生评分域不一致。

以 `third_party/qw3/src/kvmem_store.cpp::KvMemStore::repository_score_frame()` 和当前 qw3 对应调用为依据，提供薄 native binding 或共享 helper，取得本次真实 `score_mask`：

- 使用当前 generation-qualified handles、实际 token counts、accepted history boundary，并持有索引／repository lease。
- 使用本次有效 runtime selection budget，保持原生 `n > budget_blocks && sink + recent < n` 等条件及按历史顺序的 sink/recent 处理。
- 额外 mandatory 块不会仅因 mandatory 身份被排除；若它本来属于原生屏蔽的 sink/recent，仍按原生规则屏蔽。
- runtime budget 临时修改必须是有作用域的，正常返回和异常退出均恢复；不得影响下一次选页。
- 不复制一份近似 Python 判断；仅删除 `key not in mandatory` 不能算完成全部对齐。
- torch reference 和 qw3 CUDA 两条评分路径共用这个 mask。保留各 Query token/head/layer 的原先归约语义，不额外改变 kernel。

先在 `score_mask` 上做 per-token/per-head softmax 与原归约，得到完整原始向量 `x`。然后才构造独立 `transport_mask`。对于 Query，它排除本次原生策略实际固定保留的节点；剩余可竞争节点即使原始分数为零，也必须允许接收传播。

不得先把 mandatory 从 logits 删除再在输出补零；也不能认为对最终总分归一化能修复 per-token 分母变化。原始 `x` 的计算不应随 graph off/collect/scoring 改变。`observe_only=True` 保留原 `query_history.preview()` 观测副作用，但不创建图评分／选择事务，不要求尚未提交的图水位追上 archive。

### 2.2 必须加入的排名反转用例

两条 Q、一个 head、scale=1，候选 A/B 与额外 mandatory M 的 logits：

```text
          A    B    M
Q1        2    0   10
Q2        0    1    0
```

排除 M 后计算得 A≈1.149738、B≈0.850262；保留 M 得 A≈0.212277、B≈0.576162、M≈1.211561。预算为两个块且 M 必留，新路径应选 M+B。测试实际 authority/native 路径，不能只测试孤立 softmax。另覆盖小历史全部可容纳、缩小预算、sink/recent 重叠、generation 变化及非法 counts。

这项对齐会改变含额外 mandatory 的旧基线结果，即使 graph=off。记录新的评分域版本，旧冻结实验不能直接充当新基线。

## 3. 统一的一跳传播规则

### 3.1 共同输入与输出

传播函数接收原始分数 `x`、逻辑 block starts、`transport_mask`、不可变 graph snapshot、`mode` 和 `alpha`。不能通过分数是否大于零来推断候选资格。

令可传播节点集合为 E，`Z = sum(x[i] for i in E)`。Z>0 时，`r[i]=x[i]/Z`；非 E 节点不发送、不接收。两个图始终读取同一份不可变 r。输出在 E 内乘回 Z，非 E 节点保留原始 x。原始分数和 mask 在日志中分别记录。

- E 为空、Z=0、alpha=0 或全部算子 identity：原样返回，不制造均匀分布，不因重新计算引入 tie-breaking 差异。
- 拒绝非有限／负输入。检查输出非负、质量守恒；只容忍明确浮点误差，不用最终重新归一化掩盖丢失。
- 一跳：本轮从 A 传给 B 的分数，不继续从 B 传给 C；B 自己原有的分数仍可沿 B 的边传播。

### 3.2 Attention 归一化

继续保存原始观测平均 mass 和已有 top-k，不修改采集、平均、阈值、封存或快照内容。`csrc/block_graph_accept.hpp` 与 Python reference 的聚合语义保持一致。

对每个可传播来源 i：

1. 读取其已保存的 top-k 出边，保留目标属于 E、权重大于零、且非自身的有效边；生产建图仍要求目标早于来源。
2. `z_i = sum(raw_edge_mass)`。
3. 若 z_i>0，令 `PA[i,j]=raw_edge_mass[i,j]/z_i`；这些外边合计为 1。
4. 若没有有效边、没有样本或来源尚未封存，令 `PA[i,i]=1`。

必须先过滤再归一化；不从已丢弃的第 5 名边补位。没有有效边时，不伪造连接，不扩大候选范围。每次评分使用临时边权，禁止修改共享 snapshot，因为 Query 与 pressure 的候选集合不同。

例：原边 `[.03,.02,.02,.01]` 变为 `[.375,.25,.25,.125]`；若前三个目标被 mask，第四条有效边变为 1。这会放大弱关系，是明确的策略变化，需通过质量实验检验，不宣称归一化必然提高准确率。

### 3.3 Locality 与单一 alpha

Locality 保持原始位置 `p-B`、`p+B` 两个方向槽，各基础概率 .5：

- 同一个已知消息／工具结果完整包含两个 block 的联合区间：系数 1。
- 联合区间内部跨越已知边界，或 block 本身横跨边界：系数 .25。
- 元数据不足：系数 1，明确标记 unknown fallback。
- 邻居缺失、不可传播或边界衰减扣除的质量均留在自身。不跨洞找下一个现存块。

不要把 locality 剩余外边再次行归一化，否则会抵消边界降权。

V2 固定：

```text
locality:  s = (1-alpha)r + alpha * PL^T r
attention: s = (1-alpha)r + alpha * PA^T r
multi:     s = (1-alpha)r + (alpha/2) * PL^T r + (alpha/2) * PA^T r
```

multi 中某个算子没有可用边，其份额作为自环保留，不自动转交另一个算子。初始实验 `alpha=.2`，Query 与 pressure 共用这个值；不另加 pressure alpha、各层权重、温度、top-p 或多跳次数。沿用 top-k=4、min_mass=0、cross-message=.25 作为固定初值，不新增调参矩阵。

数值 oracle：`r=[.1,.2,.7]`，starts=`[0,4,8]`，唯一 attention 边 `8→0`、原 mass=.8，全部可竞争、无已知消息边界。V2 multi alpha=.2 应得到 `[.175,.220,.605]`。

## 4. 累计 attention 驱动的 graph pressure

### 4.1 精确复用旧 seed

从 `src/vllm_kvmem/prefill_attention.py::select_pressure()` 提取无副作用的共同评分 helper。对原始工作候选集合 C 中每个 block：

```text
如果 state.samples > 0:
    h_i = action_decay * state.scores.get(start_i, 0)
          + state.mass.get(start_i, 0) / state.samples
否则:
    h_i = state.scores.get(start_i, 0)

仅当 C 内所有 output h_i 都没有正值时：
    h_i = prefill.scores().get(start_i, 0)  # 原有 mass / exposure
```

先在完整 C 上决定 output→prefill fallback，再套图传播 mask。只有 sink/mandatory 有正 output 分数时，也必须沿用旧 fallback 判断，不能私自改成 prefill。

pressure 不调用 `ActionHistory.finish()`，不再次衰减，不增加样本，不混入旧 Query 分数，不把传播结果写回 `scores/mass/prefill` 或图边。相同状态上的重复评分应完全一致。

### 4.2 候选与原生压力选块

- 正常压力采用 `memory._store.working_records()`；checkpoint 恢复采用传入的 `candidate_records`／`checkpoint.window.records`。不能在恢复后用已清空物理 residency 的状态重建候选。
- 校验 generation、逻辑起点、counts、accepted boundary 和 owner。普通冷历史块即使图边或历史分数很高也不能被召回；原生允许的 sink/mandatory 单独保障。
- pressure 的传输集合为 C 扣除它本次真正硬保留的节点。原生 pressure 本身只硬保留 sink/mandatory，不能直接照搬 Query 的 recent mask；如调用方已把某个 recent/tail 块纳入 mandatory，则按实际 mandatory 处理。
- 调用共同传播函数，将 h 变成 s，然后使用 `memory.select(..., pressure_scores=..., action=False)` 的压力分支。不要调用面向全历史的 `select_for_queries()` 或复用其整套 Query picker。
- **传给 native 的列表必须包含 C 中每个节点，零分节点也保留。** native 用这个列表约束压力候选；删除零分节点会改变时间顺序补齐，误传空列表可能重新打开全 repository 候选。
- 全零 seed 不生成均匀分数；携带完整 C 的全零列表执行原生时间优先补齐。非空历史但 C 异常为空，保留明确错误。
- alpha=0 必须逐块恢复旧 attention pressure 结果，包括全零、只有必留块有正分、partial tail 和缩小 runtime budget。

### 4.3 三个入口与事务

接入同步 `bounded_runtime.py`、异步 `async_refresh.py` 和 `prompt_checkpoints.py::prepare_claim_window()` 的压力路径。保留现有 trigger、accepted-prefix barrier、GPU fence、预算与装配流程；不新增 probe 或模型 forward。

评分使用与候选快照一致的 graph owner/epoch/branch/revision/accepted boundary；评分前、装配发布前都校验。沿用图 snapshot、scratch 和结果的配额计费，trace 失败应发生在原生装配事务开始之前。压力选页本身不改变图 accepted 水位或追加采样。

两项恢复顺序约束必须显式处理：

- 当前同步／异步恢复可能先 `prepare_claim_window(...pressure=True)` 选页，再调用 `prepare_restore_metadata()` 合并消息 spans。新 graph pressure 不能沿用这个顺序：先验证 incoming metadata，恢复正确的图／action 状态并安装本次 locality spans，再捕获评分快照和开始选择。否则选页使用旧边界，或随后 metadata revision 改变导致自身 commit 失效。不能通过放松 stale-binding 校验绕过；增加带新 metadata 的 pressure restore 回归。
- 完整 M 前缀已验证但模型退回 P 时，允许保留的图／attention 水位大于当前 authority P 水位。不要把“快照一致”实现成两者无条件相等。候选位置与 counts 必须在当前 authority accepted prefix 内，保留到 M 的统计须有完整前缀验证依据，owner/epoch/branch/revision 必须匹配；真实后缀编辑不能继承未来统计。

## 5. Locality 的真实消息边界

### 5.1 交付范围与入口约束

当前 `http_sessions.normalize_query_plan()` 接受 token-ID completion 的显式坐标；结构化 chat/text 请求不能直接携带这些坐标。elastic gateway 也主要接受 `append_token_ids`，仓库内没有可直接假设已完成的通用 chat renderer。

优先利用已有的真实结构化入口 `benchmarks/deepswe/claude_bridge.py`，不要求重写整个 chat 服务：`Protocol` 已返回 canonical prompt、消息 byte spans 和 content spans；`Planner.prepare()` 生成实际发送的 token IDs；`Trajectory` 与 `Planner.commit()` 保留实际接受的 assistant output。当前 plan 尚未提供图边界。

新增共享 `src/vllm_kvmem/prompt_spans.py` mapper，输入权威 renderer 的文本与消息 byte spans，以及实际完整编码的 token IDs／offset mapping，输出 token 坐标的 graph-only locality spans 和 provenance。接入 bridge 和至少一个可运行的结构化验收入口；仅给测试手写 spans、仅保持已有透传，不能算修复这个问题。

- 直接使用 renderer 已提供的每条消息／独立 tool-result 区间；正文与现有模板保持逐字节不变。其他支持的结构化入口可复用 mapper，无需首版开发通用模板插桩。
- 把 UTF-8 byte offsets 正确转换为 tokenizer offset mapping 使用的字符坐标，映射到实际一次完整编码的 token 半开区间。
- 核对 token IDs 与实际正常生产路径逐个一致，包括 special tokens、thinking 参数、工具声明、generation prompt 等选项。已复用的历史与新 suffix 分别使用它们实际发送的编码，不能重编码旧历史代替真实 output IDs。
- 不靠正文 substring 查找、不单独 tokenize 每条消息或前缀推算长度、不向实际模型输入插入标记。
- 重复正文、中文、多字节字符、空内容、连续 tool messages、模板可能重写历史等情况不能造成坐标错位。
- 跨字符边界的 token、缺少可靠 offsets、未支持的模板、多模态改写、未完成 assistant header 等无法精确确认的范围标为未知；不伪造“精确”区间。helper 严格模式可明确拒绝，宽松模式保留正常 token IDs 并说明 metadata 不可用。

现有 `PrefixTokens.count()` 及其测试保证的是独立文本前缀的 token 数，不保证任意端点都是完整编码中的边界。新增 locality mapper 不能直接把所有 byte endpoints 交给这个函数。对于经验证的 Qwen special delimiter 硬边界可以复用其证明；其他端点使用完整编码 offset mapping。保留原有 Query/mandatory 坐标生成行为，本任务不顺带重写它们。

续接时保留历史 locality spans，并在 `Planner.commit()` 用实际 prompt/output token 索引记录已接受 assistant 区间；`Trajectory.locations` 目前没有保存所有 assistant 坐标，需增加独立图元数据状态。新 suffix 的区间只加实际历史长度一次。更新 `scripts/freeze_timed_agent_trace.py` 使用的新 trace schema/provenance，旧 frozen traces 不原地改写。

对于原始 token-ID 输入，继续允许调用方提供已知精确区间。没有 messages 或结构化来源时，不能从 token 顺序凭空恢复句子／工具边界；fallback 日志与验收必须明确区分这种输入。

### 5.2 边界元数据不能改变 mandatory 策略

**重要：现有 `kvmem_message_spans` 不只是图元数据。** `RequestPlan.parse()` 会据此解析默认 system/root/query 等 mandatory，并改变 `explicit_mandatory`。自动添加它可能意外增加必留块，干扰评分实验。

建议新增 graph-only 的 `kvmem_locality_spans`，元素沿用 `{begin,end,role}`，或采用等价的明确独立字段。它只服务图 locality：

- 不改变 mandatory 集合、explicit_mandatory、Query/replay 坐标或原 token IDs。
- 保留调用方原有 `kvmem_message_spans` 的旧契约。两类元数据同时出现时，已知重叠区域的边界／角色必须一致，否则拒绝。
- 更新 `RequestPlan → payload → scheduler/connector → SessionBinding → authority → checkpoint` 全链路；新字段放在 dataclass 末尾以兼容既有位置参数调用。
- 服务端验证区间范围、排序、非重叠和消息角色；不同消息不能仅因 role 一样就合并。
- elastic 追加协议若采用相对 append 区间，字段名／协议必须显式区别绝对坐标，并在验证该 session 原 token 前缀后转换一次。不能客户端、gateway、worker 重复加 offset。
- 后续完整历史重渲染改变了 token 前缀时，沿用真实分支编辑／恢复逻辑，不能继续沿用旧 spans。

保留 `message_end_known`：仅验证到一条消息中间时，截断点不是已知消息终点。完整前缀恢复、真实编辑、跨响应的 partial block 均要正确继承／裁剪。仍只连原始位置相邻块，不把整个工具输出展开成全连接图。

## 6. 配置与兼容

建议接口如下，允许遵循仓库现有命名调整，但语义与迁移规则必须明确：

| 字段 | 规则 |
|---|---|
| `graph_mode` | 保留 off/collect/locality/attention/multi。默认 off。 |
| `graph_scoring_version` | v1/v2；未指定仍按 v1 解释旧配置；新增目标 profile 显式 v2。 |
| `graph_alpha` | V2 唯一新连续权重，默认 .2，有限数且在 [0,1]，不接受 bool。 |
| `pressure_policy` | recency/attention/graph；不指定时由旧 action_pressure 解析。 |

具体兼容要求：

- v1 保留旧双权重和原 attention 绝对质量传播。v2 使用 alpha 与固定等权，不可把旧 `.1/.2` 静默求和映射为新 alpha，二者不等价。
- 显式同时指定 alpha 和旧 locality/attention 权重时拒绝。v2 显式指定旧双权重也拒绝；解析器必须区分用户显式字段与系统补入的旧默认值，不能误报冲突。
- V2 resolved profile／worker transport 序列化使用规范字段，省略旧双权重及旧 `action_pressure`，保留 version、alpha、pressure_policy。否则旧默认值写进 JSON 再加载后会被当作显式冲突；增加 `load → resolve → serialize → reload` 等价测试。
- 未指定 `pressure_policy` 时，旧 `action_pressure=false→recency`、`true→attention`。指定枚举后由其决定执行路径；若还显式指定旧 bool，只有语义一致的别名组合可接受，graph 与旧 bool 的含糊组合拒绝。默认值注入不算用户冲突。
- `pressure_policy=graph` 要求 v2 且 graph_mode 为 locality/attention/multi；首版不为 off/collect 单独创建隐式传播模式。
- graph pressure 的 seed 依赖 action_attention/prefill_attention。所有新目标 profile 开启这两项，保持 action_scope=non-thinking。
- Query 图评分继续保持 action_fraction=0、query_history=false、repository_gc=false、request_planning 与 owned sessions。action_fraction=0 不妨碍 pressure 使用 attention seed。
- 解除原来 graph scoring 一概禁止 action_pressure 的限制，改成按解析后的 pressure_policy 校验。attention 压力与 V2 Query 图评分的组合也应可用，便于消融。
- direct、elastic、worker connector 的解析／resolved profile 全链路一致。策略版本与 alpha 在会话生命周期内固定；不允许恢复同一会话时静默切换策略。

新 direct profile 的 `kvmem` 片段示意（elastic 对应字段位于顶层；其余模型／预算／会话配置沿用合法完整 profile）：

```json
{
  "graph_mode": "multi",
  "graph_scoring_version": "v2",
  "graph_alpha": 0.2,
  "pressure_policy": "graph",
  "action_attention": true,
  "prefill_attention": true,
  "action_scope": "non-thinking",
  "action_fraction": 0.0,
  "query_history": false,
  "repository_gc": false,
  "request_planning": true
}
```

请求层仍使用现有 `kvmem_prefill_window=pressure` 等合法计划。本次不顺带开启 semantic_chunk、rolling active window 或改变 K/G、batch budget、采样密度。

## 7. 代码入口与实现顺序

1. 阅读目标最新 AGENTS、git diff、graph 文档及 native source；确认与正在进行的优化不冲突。独立分支／副本只提交本任务增量。
2. 为原生 Mean-K score mask 增加薄 binding，修改 authority/retrieval 接线及独立 oracle 测试。若当前第三方 native 已有 helper，优先复用，不重复实现策略。
3. 在 `graph_scoring.py` 增加版本化传播；`graph_integration.py` 保留 Query wrapper，抽取共用的版本校验／诊断，新增压力 wrapper。不要把两个 picker 混成一个。
4. 重构 `prefill_attention.py` 的只读 seed 生成，分别接 legacy attention 与 graph pressure。同步、异步、P/M restore 均调用统一入口。
5. 实现 graph-only spans 协议、精确 producer helper 和实际调用方接线；沿用 `block_graph.py` 的消息合并／快照约束。
6. 更新 `options.py`、`elastic_serve.py`、serving launcher、connector 配置、示例和文档。
7. 更新 `scripts/analyze_block_graph.py`、图输入导出和新实验 qualifier；为新版本创建独立实验目录与 manifest。

保留 `csrc/block_graph_accept.hpp`、`block_graph.py::_DenseSamples` 及 `tests/test_block_graph_dense.py` 的 native 聚合优化、stride 支持、配额和数值对照。归一化发生在稀疏评分阶段，不要把密集采样矩阵重新展开成 Python 标量列表。

## 8. 诊断与离线回放

每次实际选择至少记录：

- phase=query/pressure，实际 seed_source=mean_k/output/prefill/none。
- scoring version、Mean-K score-domain version、pressure policy、mode、alpha，以及 V2 固定算子份额。
- 本次 runtime budget、候选数量、评分 mask、传播 mask 的统计和 owner/epoch/revision/accepted boundary。
- 有效 attention 出边覆盖的 seed 质量、保留原边质量、归一化后实际外传质量、locality 实际外传质量、self-retained 质量。
- 传播前后 L1 净变化、入选集合替换数量；trace 同时显示原边权和本次归一化权重。
- locality 同消息／跨边界／未知 fallback 的比例、元数据来源和模板／tokenizer provenance。未知不能算语义覆盖。
- 图 live/snapshot/scratch 配额与评分耗时；不能以收费字节等同 RSS。

外传总质量不等于净变化，也不等于答案质量。不能保证所有事件外传恰好 alpha：identity rows、locality 边界和未覆盖来源会保留自环。

更新导出 schema 版本，使 replay 携带实际传播版本、参数、候选列表／generation 信息、原始 seed、mask、消息区间与 selected baseline。V1 旧记录用显式兼容解析；未知版本拒绝。Pressure replay 必须调用 pressure picker，并恢复完整候选列表，不能套用 Query replay。

继续保留日志条数／字节限制和评分 scratch admission，不为回放强制全量转储长历史。固定旧 trace 可以验证新算子，但无法重新生成 mandatory 对齐后的 Mean-K softmax 分母，也不能替代 pressure 改变后的模型闭环。

## 9. 验收测试

### 9.1 评分数学与 native 边界

- mandatory 排名反转；与当前 qw3 native score-mask 对照；小历史、runtime budget、sink/recent、count/generation 失效。
- Query off/collect/V2 使用同一份对齐后原始 Mean-K；observe-only 观测链仍正确。
- attention 按有效边归一化、全部屏蔽、只有一个有效目标、没有样本、未完成来源、极小但正的边权、零分目标接收传播。
- alpha=0、Z=0、单节点、identity 精确退化；多图固定半份额；缺边不转移到另一个算子；一跳与质量守恒。
- locality 跨边界衰减不被归一化抵消；缺块不跨洞；上述 `[.175,.220,.605]` oracle。
- 随机小图与独立矩阵／线性算子 oracle 对照，包含 holes、masks、不同 seed 来源；不能让测试只是调用同一实现两遍。

### 9.2 Pressure 与生命周期

- alpha=0 与旧 attention pressure 逐块一致，含 output 优先、当前轮增量、prefill exposure、只有必留块有正 output、全部零分。
- 零分候选仍参与原生补齐；冷历史高分不能被召回；实际 mandatory/sink/tail 与缩小预算正确；不擅自增加 recent 保留带。
- 重复压力不反复衰减，不修改累计分数和图边，图水位不因选择推进。
- 同步、异步、恢复后 pressure、M→P 完整前缀保留、真实后缀编辑、partial block、取消／失败／配额不足重试与 release。
- 恢复请求携带新增 locality metadata 时，先完成边界安装再选择，选择提交不被自身后续 metadata 更新失效；合法的 graph M 水位领先 authority P 不误报错误。
- 私有 Query、公开 replay、rejected MTP 不增加图／attention 样本；旧 epoch 结果不能提交。
- c14 dense/native 与 scalar/reference 聚合测试继续通过。

### 9.3 语义边界与协议

- 至少一个实际结构化调用方自动生成 spans，并且发送 token IDs 与原生产模板完全一致。
- 重复正文、多字节字符、空消息、连续 tool results、完整／未完成 assistant 后缀、模板选项、append 绝对坐标和前缀重写。
- 仅增加 graph-only metadata 时，token IDs、Query/replay 范围、mandatory 集合和 explicit_mandatory 均不变。
- 相同已验证前缀的边界／角色冲突拒绝；中途截断不是消息终点；两类 spans 冲突拒绝。
- 裸 token 无结构信息与不支持模板明确 fallback，不伪造语义单元。

### 9.4 真模型与对照

至少完成一个多次 pressure 的长 prompt，以及含工具结果的多轮会话，覆盖同步／异步和 checkpoint 恢复；核对选页预算、图水位、压力分数来源、实际边界覆盖与资源释放。随后开展同输入、模型、K/G、batch budget、采样和 attention 采集配置的闭环比较。

最小主实验矩阵：

| | 旧 attention pressure | 新 attention-seeded graph pressure |
|---|---|---|
| 对齐后的 Mean-K Query | A | B：Query 传播 alpha=0 的受控测试路径 |
| V2 Query 图评分 | C | D |

产品只提供一个共享 alpha，不为这个矩阵新增第二个生产权重。B 可以由测试 harness 显式禁用 Query 传播实现，并在 manifest 标记；不能偷偷改变产品配置含义。若初版不做 B，至少 A/C/D 三组，明确它们分别测量 Query 图增益和 pressure 图增益。

另用固定输入比较 V1/V2 attention 归一化、V2 attention-only/multi，以及真实消息边界／未知 fallback。为隔离边界效果，不能同时改变 mandatory、tokens 或压力策略。输入没有结构的 NIAH 只能验证位置 fallback，不能冒充语义 locality 验收。

旧 qualifier 在 `qualify_block_graph.py`、`qualify_graph_niah.py` 等处写死 pressure=false，需要新增协议／版本分支。保留旧实验 manifest 和失败记录；所有新结果写入新目录，不覆盖旧 capture。最终分开报告证据命中、回答质量、外传／净变化和 host/GPU 成本，不用 CPU 测试通过宣称质量提升。

## 10. 完成标准与交付

交付代码增量、完整新示例、配置迁移说明、单元／native／生命周期测试、精确边界 producer 示例、版本化离线回放和模型验证记录。说明哪些验证实际运行、哪些尚未运行及原因；不把设计方案或本交接文档当作已实现功能。

新 V2 profile 应能在同一服务中完成：真实消息／工具边界建图、累计 attention 驱动的图压力压缩、当前 Query 驱动的全历史图检索。旧合法 profile 仍可解析；Mean-K mandatory 对齐导致的有意基线变化必须明确记录。

执行中可以自行完成必要实现与验证；不扩大为新的检索算法研究，不引入多组可调权重，不用旧独立副本覆盖最新主线，不改动与本任务无关的并行工作。
