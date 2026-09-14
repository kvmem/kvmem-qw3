# Step-level memory scheduling：补实验与改图方案

本文依据附件 `attention_step_scheduling.pdf`、现有分析脚本和已保存结果制定。本文列出的新实验均为待执行方案；没有把已有 summary 的核对当作重新采集或重算结果。

## 1. 现有证据与缺口

附件上图说明某条轨迹的跨步 attention 变化大于步内变化，下图说明 top-k block 的 attention mass 集中程度。论文需要进一步验证三件事：

1. 这种跨步/步内差异是否在多条独立 agent 轨迹中成立？
2. **只用本步 prefill 中可获得的信号选出的历史集合，能否覆盖后续整个 decode？**
3. 在相同检索器和 KV 预算下，每步更新一次是否具有合适的质量与开销折中？

第 2 项是目前最直接的证据缺口。每个窗口根据该窗口真实 attention 重新选 top-k，只能测 oracle concentration，不能验证 prefill-only selection；相邻窗口变化小也不能排除一整步内缓慢累积的漂移。

### 已核对的本地产物

- 附件上图的 `within=0.0655902`、`boundary=3.103397` 与 legacy 结果中 `django__django-11019` 精确对应。该项记录有 28 个 within 转换和 7 个 boundary 转换。这个对应关系仅核对了上图，未确认附件下图的完整数据来源。
- 已有多轨迹图：`benchmark/results/motivation_v2/partial/figures/main_motivation_partial.pdf`。
- 已有多轨迹结果：`benchmark/results/motivation_v2/partial/analysis_partial.json`，19 条 eligible rollouts。

| 现有 partial 指标 | 保存的结果 |
| --- | ---: |
| rollout-level within KL 均值 | 0.06958 bits |
| rollout-level cross KL 均值 | 1.15550 bits |
| 逐 rollout cross/within 比值中位数 | 15.7156× |
| rollout-level cross > within | 19/19 |
| Oracle top-8 coverage 均值 | 61.196% |
| Oracle top-16 coverage 均值 | 75.048% |

注意：19/19 表示每条轨迹的聚合量满足这个关系，不能写成所有 step boundary 均超过所有步内窗口。15.72× 是逐轨迹比值的中位数，不是两个总体均值之比。

19 条轨迹涉及 django 10 条、pytest 6 条、pylint 2 条、pydata 1 条，因此还需要扩充任务分布。

当前 `manifest.jsonl` 有 49 条 collected、42 条 pending、9 条 failed。49 是状态记录，不等于已重新验证过的 49 条有效分析样本。原始 shared trace 约 2.1 GB，仍在磁盘上；`scripts/motivation_v2/` 中多数分析/采集模块目前只有 1 字节，重分析前需要恢复。已有 64/128/256 窗口稳健性结果只覆盖 legacy-8，不能当作 49 条的结果。

## 2. 建议的主图结构

正文的 scheduling motivation 图建议使用两个横向 panel；完整调度消融放到 evaluation 的独立图或表。

### (a) Across-step versus within-step historical attention drift

- 横轴：Within-step / Across-step。
- 纵轴：历史 attention 分布的 JS divergence，或清楚标注方向与平滑参数的 KL。
- 每条 rollout 产生一对聚合值，画成配对散点/连线，叠加总体统计与 bootstrap CI。
- 按 workload 分面；模型多时按模型分行。不要把几十条时间曲线叠成一团。
- 原附件时间曲线可以移至附录作为示例；按预先规定的规则选取，比如最接近总体比值中位数的轨迹。
- 图注同时给出轨迹数、有效 step 数、boundary 数及窗口定义。

### (b) Historical coverage of a working set selected at prefill

- 横轴：本步 decode 进度，0–100%；附录另外画绝对 token 距离，避免长 decode 的退化被归一化掩盖。
- 纵轴：历史 attention mass coverage。
- 曲线：本步 prefill 选一次并固定、沿用上一步集合、同预算当前窗口 oracle。
- 补充每 128 或 256 decode tokens 更新的实际检索器作为工程对照。
- 主图画 rollout 等权均值与明确标注的区间；另报 coverage 的低分位和相对 oracle 的差值。

原附件的 `Coverage vs top-k` 可以放到检索设计/稀疏性部分；它回答的是选多少 block，而新 panel (b) 回答的是何时更新。

## 3. 实验 A：多轨迹历史关注变化

### 采样与聚合

- 先修复分析链路，验证现有 collected 数据的请求/step 对齐，再分析所有合格轨迹。
- 扩采目标可设为 50–100 条独立轨迹，这是资源规划起点，不是统计充分性的保证。根据实际轨迹间方差和 CI 宽度判断是否需要更多样本。
- 固定任务 manifest，覆盖不同 repository、上下文长度和 decode 长度；保留失败、超时、截断状态，不能按是否支持假设筛选。
- 算力允许时，增加第二种真实 agent workload，例如已有 GUI agent 场景，并在第二个模型上重复核心指标。资源有限时优先做同模型上的严格控制和冻结集合验证。
- 先在 step 内聚合窗口，再在 rollout 内聚合 step，最后在 rollout 间聚合。bootstrap 以 rollout 为重采样单位；同任务多 seed 时应保留任务层的聚类关系。
- 短 decode 可以用于覆盖率分析；不足两个分析窗口的 step 不贡献步内 divergence。排除数和长短 step 分布都应报告。

### 必做的测量控制

1. **等窗口、无重叠对照。** 当前窗口长 128、stride 32，步内相邻窗口共享 96 个 token，而边界两侧不共享 token。主分析或至少关键对照使用 128/128；64/64、256/256 作敏感性检查。短 step 的窗口大小要匹配，不能将不同长度的窗口直接混为同一对照。
2. **共同历史支持集。** 比较相邻 step 时，只在两者共有的、身份一致的旧历史 blocks 上衡量重新分配；新增 tool output、新 step 输入和新增 tail 的质量单独报告。以稳定逻辑位置/块身份对齐，不能直接假定重排后的执行位置等同于历史 block ID。
3. **边界必须真实相邻。** 不跨任务拼接；过滤短 step 后，不能把原本隔了若干 step 的两段误记为相邻 boundary。优先使用 request_id / rollout_id / step_id，而非仅用 sample 计数回绕。
4. **记录历史总质量。** 对历史内部的条件分布重新归一化之外，报告历史占全部 attention 的比例。历史质量接近零时，条件分布变化的解释要单独处理。
5. **避免可见集合造成偏差。** 分布基准优先采自 dense/full-visible 运行；已经剪枝的执行视图无法显示被删掉的历史是否重新相关。无法 full-visible 的超长 workspace，要明确参考测量范围。
6. **补充分层结果。** 现有 trace 对 heads 求和、分析又合并 layers；补 layer 维度和长 decode 子集，防止平均值掩盖局部变化。若系统按 layer/head 选集，测量粒度应与调度粒度一致。

JS divergence 可作为 KL 的稳健性对照：它对称，采用 log2 时范围为 [0,1]，不需要给零概率历史块加入任意 epsilon。它仍不能替代共同历史支持集和窗口匹配。

## 4. 实验 B：prefill 选一次是否足够

这是优先级最高的新增实验。

对 step s，令 H_s 为本次 recall 的候选历史块集合，不包含必留的 sink/recent 部分和本步新输入、decode tail。令 S_s(k) 为系统仅依据当前 prefill 已获得信号选出的 k 个候选块。令 a_{s,t}(b) 为参考运行在 decode 位置/窗口 t 上分配给块 b 的 attention mass。

    C_frozen(s,t,k) = sum_{b in S_s(k)} a_{s,t}(b)
                      / sum_{b in H_s} a_{s,t}(b)

对同一参考分布、同一候选集和同一预算计算当前窗口 oracle：

    C_oracle(s,t,k) = max_{S subset H_s, |S| <= k}
                       sum_{b in S} a_{s,t}(b)
                       / sum_{b in H_s} a_{s,t}(b)

主报告 `C_frozen`、`C_oracle - C_frozen`、低分位 coverage、以及它们随 decode 距离的变化。分母为零的窗口标为不可定义并报告数量，不能填成 100%。必留部分单独给出质量/内存预算，同时可补包含必留部分的总覆盖率。

### 对照与信息边界

| 方案 | 选集时可用信息 | 目的 |
| --- | --- | --- |
| 本步 prefill 后选一次 | 本步 prefill 截止时已产生的信号 | 直接对应论文策略 |
| 沿用上一步集合 | 上一步已选的历史块；保持同一候选历史预算 | 衡量本步 refresh 的收益 |
| 每 128/256 decode tokens 更新 | 截止本次更新时已有的因果信号 | 衡量更频繁更新的边际收益 |
| 当前窗口 oracle | 当前窗口完整参考 attention | 同预算 coverage 上界，仅作离线参考 |

**不能用前 128 个 decode tokens 选集合，再把它标成 prefill selection。** 同样不能用全步平均 attention 来构造 supposedly 在线的一次性选集。现有 `kvmem_attention_stability.py` 中的 `observe_samples` 属于 decode 观察前缀，应清楚标注为诊断基线。

需要额外记录真实 prefill 检索信号与选集。如果采用 dense 轨迹离线 replay，所有策略评估相同 token 序列以隔离集合选择；这只验证 coverage，实际剪枝引起的轨迹变化交给实验 C。

为了进一步定位 recall 时点，可加“本步 prefill 早期选集”和“prefill 结束选集”的因果对照，检验吸收完整新输入的收益。

预算至少覆盖偏紧、中等、宽松三档；以系统实际 KV token/byte 预算为准。对于不同历史规模，同时报告绝对预算与候选历史保留比例。

## 5. 实验 C：调度频率的质量—开销消融

固定模型、检索器、索引表示、候选历史定义、KV 预算、硬件和任务，只改变更新时机。

| 策略 | 更新时机 |
| --- | --- |
| Stale / every-two-step | 跳过部分 step refresh |
| Step-level | 每个 step prefill 后一次 |
| Step-level + periodic | 本步初始 refresh，加每 64/128/256 decode tokens refresh |
| Token-level（小规模） | 使用已计算 query 等因果可用信号，每个 decode token refresh；若实现是先生成该 token 后才 refresh，要记录一 token 滞后 |
| Dense/full-context reference | 模型支持且资源可行的上下文范围内运行 |

指标：任务成功率/答案正确率、step latency、TTFT、decode latency/TPOT、检索次数、检索计算耗时、搬运与组装时间、实际读取 bytes。TTFT 必须包含首次 decode 前的 recall；只比较 decode TPOT 会漏掉本方案的主要固定开销。

以任务质量为纵轴、端到端延迟为横轴画散点，标注调度策略和预算。长 decode 子集单独报告。是否“质量接近”应预先定义实际可接受的差异并给出 CI，不能用显著性检验未显著替代等效性结论。

## 6. 与文中其他论点分开验证

这张 scheduling 图无需同时承担以下两项验证，但论文若保留对应主张，应在别处补证据：

- **Model-native retrieval：** 固定预算，比较系统真实 prefill 检索、词法检索和独立 embedding 检索的历史 attention coverage、证据召回与任务质量。attention coverage 本身不等于答案质量。
- **Millions-token scalability：** 在 64K、256K、1M 等实际支持的 workspace 长度上报告索引 bytes/token、峰值索引内存、selection latency 和 stage-in bytes；说明上下文外历史的参考测量方式。旧 partial 的平均窗口只有约 103 个 128-token 历史块，其 top-128 接近 100% 不能单独证明百万 token workspace 的稀疏性。

## 7. 执行顺序与论文措辞

1. 恢复分析源码并核对当前 collected 数据，对不重叠窗口和共同历史支持集重算。输出多轨迹 panel (a)。
2. 采集 prefill 选集，补冻结集合 coverage 曲线。输出 panel (b)。
3. 运行相同预算下的调度频率消融，形成 evaluation 图/表。
4. 按 CI 宽度、任务覆盖和尾部结果决定扩采；再补第二 workload/model 与规模测试。

现有数据可以支持有限范围内的观察性描述，例如：“In the measured agent rollouts, historical attention changes more across step boundaries than within decoding steps.” 但新控制尚未完成，当前数值不应被写成普适规律。

最终正文应分开陈述观察与验证：先报告真实 step boundary 上的变化，再报告 prefill-selected working sets 在后续 decode 中的保留能力，最后给出 once-per-step 相对高频更新的实际质量与开销。若长 decode 明显退化，应据实收窄适用范围或考虑按需 refresh，不能通过删除这些样本维护原结论。


## 8. 2026-09-11 恢复执行后的实际方案

- 旧 49 条已完成原始 trace 核验，对应 343 次唯一匹配的模型请求；全部为迭代上限截断的历史 OpenHands rollout，不是 49 个成功解决的任务。完整分析与控制图位于 `benchmark/results/motivation_v3_20260910/legacy49/analysis/`。
- 新样本冻结为四类各 20 条源轨迹、合计 957 个连续调用前缀：软件代理、AgentLongBench 工具/状态历史、LongMemEval 多会话对话、OSWorld GUI。统一使用 Qwen3.8-27B-Q8_0、FP16 KV、64K 上限，和旧 Qwen3.6 队列分开报告。
- 新测量属于 recorded-input replay：为每个记录输入生成新 continuation，但不执行输出动作，也不将其写入下一步记录历史。因此不会产生新的端到端任务成功率。
- 生成上限在正式采集前定为 2048、thinking budget 512；自然短输出保持原样，截断单独报告。候选历史固定排除首个 128-token block 和近期 8 个 prompt blocks。
- 首版 scalar prefill-tail 试采因改变生成结果而被排除。正式采集改用原批量 prefill 中的 Q/K 旁路诊断；文本和图片请求均通过与普通 prefill 的完整输出一致性检查，8 个试采请求通过对齐和因果性检查。试采不计入正式样本数。
- 配对主指标使用非重叠 16/32/64-token 窗口，128 为补充；严格比较相同两步、相同已核验输入前缀。聊天模板可能删去旧 reasoning，故逐请求记录实际 token IDs、mRoPE 和图片 hash，不能仅凭消息数组验证历史一致性。
- 冻结集合来自最后 64 个真实 prefill 查询的精确 attention，只作为调度诊断，不等同于实际 compact-index selector。预算报告 5%/10%/20% 和绝对 8/16/32/64 blocks；与同预算 per-token oracle 对照。
- 离线高频对照使用过去 32 个查询，在每 32/128/512 个 decode token 更新；只报告 coverage 和更新次数，不把次数当作延迟。另按输入长度、decode 长度和自然结束/截断分组。
- 正式任务由 `scripts/motivation_v3/run_campaign.py` 执行，输出目录为 `benchmark/results/motivation_v3_20260910/formal80/`，四批各 20 条源轨迹，每批后自动生成主图、控制图和报告。`campaign.json` 是批次状态，有效样本数以每个请求的 `status=collected` 为准。
- 上文实验 C 的在线质量—延迟消融、实际 compact-index 比较和百万 token 扩展性实验仍属于独立验证，不能将当前 attention 诊断表述成这些实验已经完成。
# Source-identity amendment (2026-09-12)

采集中发现软件源目录解析问题：`pier_jobs` 前可能是 `attempt_N` 重试目录，真实任务名位于其后。原 20 条软件轨迹实际覆盖 19 个任务；`code_attempt_04` 与 `code_wasmi-trap-coredumps` 属于同一任务。原始采集和清单保持不变。

主分析按原清单顺序保留首次出现的任务，保留另一尝试为辅助数据；补充任务按未入选真实任务名称的字典序、原文件顺序和原长度资格规则确定，不依据 attention 结果或任务评分。已冻结 `code_aiomonitor-task-snapshots-diff` 的 12 个前缀。最终共采集 81 条新轨迹（969 次调用），其中 80 条（957 次调用）构成四类各 20 个独立源任务/轨迹的主分析。

修正清单位于 `benchmark/results/motivation_v3_20260910/corrected80/manifest.json`，记录原清单及补充清单的完整来源与哈希。旧测量哈希不改写；逐请求核验验证来源清单、源文件与帧身份、协议一致性和真实任务唯一性。采集后补充这一事实将明确披露。前 40 条中期样本未含重复任务，其数值不变，且已通过加强后的来源校验。

最终分析、完整性校验、图表审阅和论文措辞以修正后的主队列为准。原始批次 3/4 自动生成的未修正图表仅保留作辅助记录。详细操作与当前状态见 `benchmark/results/motivation_v3_20260910/source_identity_correction_plan.md`。

## 9. 2026-09-12 完成记录

修正主队列已完成：四类各 20 条、共 80 条轨迹和 957 次调用；另保留 1 条同任务辅助尝试、12 次调用。逐请求核验与源任务身份核验均通过，十项模型/配置文件在最终采集后重新计算 SHA256，与最初记录一致。实际采集耗时约 13.34 小时，包含共享 GPU 时段和 attention 记录开销，不作为服务性能结论。

五张全量图已经目视检查，主图由归档脚本重绘后 PNG 与图注逐字节一致。80 条轨迹在 W=32 的配对差值均为正；固定集合仍有 decode 内漂移。工具 W=16、软件 W=128 的 CI 包含零，长输入和稀少的长 decode 样本单独报告。论文表述采用更新频率与覆盖率之间的权衡，不声称固定集合无损。

最终产物位于 `benchmark/results/motivation_v3_20260910/corrected80/analysis/`：正文主图 `multi_domain_scheduling.pdf`、四张控制图、中文报告 `paper_report_zh.md` 和英文稿 `paper_design_draft.md`。复核与校验值见 `corrected80/final_review/`，重绘命令见 `corrected80/reproduce_figures.md`。旧 49 条仍单列于 `legacy49/analysis/`。

上述完成范围是第 8 节确定的 attention 调度诊断方案。第 5/6 节的闭环质量—延迟、生产 compact selector 和百万 token 扩展性属于独立系统实验，本轮没有执行，也不据此提出对应实验结论。
