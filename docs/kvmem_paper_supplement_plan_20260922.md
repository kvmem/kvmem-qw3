# KVMem 论文现有证据与补充实验计划

日期：2026-09-22。状态：整理与排期方案；没有启动新实验。本文按现有原始记录判断证据可用范围。以下 GPU 时间均为单张 RTX PRO 6000 的规划估计，不包含实验接口开发、编译和故障修复；完成首个单元后重新估时。

## 一、现有证据的使用方式

| 证据 | 入文位置与可支持结论 | 需要保留的限定 |
|---|---|---|
| 四系统吞吐：240 次正式测量，5 个长度、MTP 0/4、3 次重复 | 系统效率主表；128K、MTP=0，KVMem/InfLLM/InfiniGen 为 39.32/19.74/15.44 token/s；MTP=4 为 76.91/19.02/10.11 | Qwen 适配版本，实际 KV 窗口不同，只有一个历史来源；属于这些实现和配置的系统比较，不能推导同资源同质量的算法加速 |
| 新用户 query 的刷新开销及固定轨迹 guided 重放 | 效率分解；128K 新 query 的 TTFT 中位数 0.449 s，selection 244.7 ms，replay 85.5 ms；052 的 260 请求有 3 次 guided 维护事件 | 新用户 query 与内部 hint 分开；guided 的 128K/240K 微基准各仅一次，052 每开关也只一轮；嵌套阶段不能重复相加 |
| re-RoPE/raw-K 数值实验 | 机制图或附录；1800 次重映射旧路径最大误差 0.201660，immutable 路径与直接映射逐字节一致；一个 FP16 样本在 32 次变换后答错，raw-K 重建恢复 | 历史模型、精度与二进制需标注；可证明漂移存在及缓解机制，不能由单题推导全任务质量 |
| 80 条轨迹、957 个请求的注意力分析 | 动机分析：step 内外变化与复用局部性 | 使用 full-visible attention；不是生产 Mean-K 或闭环调度效果实验 |
| 4B 有界仓库 off/shadow/delete 验收 | 实现与容量图、开销附录；固定仓库持续运行并实际释放、复用物理槽位 | 与 27B agent 成绩分开；CPU/KV/index 有界不等于完整服务进程恒定内存 |
| 4B 文档延后提问：2 个文档、32 题 | 回收质量边界；相同 16K active/8K generation 下，512K off/shadow 为 20/32，64K delete 为 12/32；1029 次非首次请求均 HIT | 是独立在线保留消融，不是正式 MemoryAgentBench 总分；保留负结果，不称为无损回收 |
| 512K GC 的 OSWorld 45 题重复 | 全任务回收压力的描述性结果；官方均分 0.2276，历史 2M 为 0.2921；24 题触发删除 | 有随机轨迹及版本差异，无法把全部差额归因于删除；官方分与 task-010 数字格式修正分分别展示，不取两次最好结果 |
| DeepSWE 16 题×4 seeds、OSWorld 45 对轨迹 | 现有端到端主结果的逐任务表及配对统计 | 先核对版本、硬件、失败及重复尝试；耗时仅同硬件配对；按任务聚类估计 CI |

主要来源：

- `baselines/kv_retrieval/length_sweep_breakdown_20260913/REPORT.md`
- `benchmark/osworld_v2/claudecode_guided_query_latest_20260910/REPORT.md`
- `docs/kvmem_known_issues.md`，KVMI-011
- `benchmark/results/motivation_v3_20260910/formal80/analysis/report.md`
- `docs/kvmem_bounded_repository_validation_20260917.md`
- `benchmark/osworld_v2/gc512k_all45_repeat2_20260918/summary.json`
- `benchmark/osworld_v2/gc512k_all45_repeat2_20260918/repeat_summary.json`
- `benchmark/claude_deepswe_ab/analysis/burst_latency_20260907/report.md`

现有单题 no-query-replay 结果、预算和 block 大小的跨版本历史结果可以放诊断附录。InfLLM 四题 pilot 未形成完整有效配对，不作为主文质量对照。历史 compaction/cold-prefill 缺陷存在的轨迹需披露其版本；不用于声称修复后的系统具有相同停顿。

## 二、先做数据整理：不占 GPU

预计 2–4 小时分析工作，可与现有推理并行。

1. 冻结 task ID、seed、原始结果路径、运行配置、binary SHA、硬件与排除理由。识别 INFRA、正常模型失败、用户中断和重试，不按结果筛选。
2. DeepSWE 以任务为单位 bootstrap 10000 次，每次保留该任务四个 seed 及方法配对。报告平均单次通过率、pass@4、配对差值及 95% CI；16 个任务是统计聚类单位。
3. OSWorld 以任务为单位对平均官方分、完整成功率和耗时差做配对 bootstrap。所有完整有效配对进入总体表；维护触发子集作为附表。
4. 分别汇总 user-query refresh、hint refresh、mid-decode refresh。提供事件数、均值、中位数、范围；样本很少时不强调 p99。未知分项留空，检查阶段包含关系。
5. GC 同时给完整任务集、触发/未触发子集，以及官方与授权修正两套分数。已有 4B 延后问答用于解释容量—保留质量权衡。

产出：逐任务 CSV、配置和排除清单、CI 表、刷新阶段表及图。统计改善证据表达，不消除非随机任务集、版本或硬件差异。

## 三、短实验队列

### P1：补齐 guided 刷新计时重复

- 三个预先固定的历史/问题来源，128K 与 240K 两档历史长度，直接用户 query / 内部生成 hint 两条路径，各三次，共 36 次测量。
- 使用现有新 query 微基准，保持 Q8_0、FP8 KV、MTP=4、128K selection、128K generation、B32、完整 replay；每次从规定的历史准备边界开始。
- 记录 hint tokens、direct/fallback、完整请求时间、TTFT、guided/selection/replay 及实际资源。fallback 保留，不补抽成功 hint。
- 计划 GPU 时间：0.5–2 小时。现有脚本主要需要扩展输入和重复矩阵。
- 产出：可报告重复性的两类刷新阶段表；这仍不回答 hint 的任务质量收益。

### P2：受控的重复 refresh 与 query replay

- 在固定的短答案长上下文任务池中，按长度分层、固定种子选择四个前缀/最终问题，保存 ID；不按原来答对与否选取。
- refresh 次数 1/2/4/8，replay on/off，共 32 次 continuation。相同 refresh 计划内共享预先记录的 selected block IDs 与位置映射，保持 immutable raw-K、源 KV、输入及所有其他配置一致。
- 正式续写前检查选择哈希、位置哈希和输入哈希。不同 refresh 次数之间允许按预先冻结的不同计划执行。
- 使用 greedy、MTP=0 做机制控制；记录官方答案分、共同最终前缀的 next-token 分布差异和生成 token agreement。四题作为机制诊断，不外推总体准确率。
- 计划 GPU 时间：1–3 小时。需要选择计划记录/回放接口及验证，开发成本高于 P1；接口不完备时顺延，不能用不同 live 轨迹替代。
- 已有 raw-K 数值结果优先复用，仅在无法恢复其 provenance 时补跑数值 sweep。

### P3：固定检查点的三种 hint 来源

- 首轮采用已有 OSWorld 052 的同一 260 请求序列，冻结三个刷新边界，比较 original task / latest observation / generated hint。
- 每种来源三轮，共九次顺序重放，轮换运行顺序。每轮从头建立自己的状态，使用相同 raw 请求、输出上限、模型和采样；生成结果不驱动下一条录制请求。
- 三组必须在相同边界执行语义选择；不能直接将现有 guided-off 当作 original-query 组。最近观察的取值范围和过长处理规则在运行前固定并公布。
- 记录完整维护成本、hint 长度、fallback、选块、实际 active tokens、模型输出与冷恢复。选块重叠或输出一致性只作诊断，不当任务成功率。
- 计划 GPU 时间：2–4 小时。历史 260 请求一次 API 累计约 12 分钟，九次约 1.8 小时；预算加入准备、记录和版本差异。需要解耦触发时机与 query 来源。
- 产出：hint 来源的受控成本表，为后续闭环实验检查机制。

### P4：刷新频率的固定轨迹成本

- 同一 260 请求序列，当前容量/step-boundary 策略、每请求边界刷新、每累计 16K 个对外生成 tokens 后在安全边界刷新；每组三轮，共九次。
- query 来源统一为 original task，隔离刷新频率。新用户 query 触发规则相同；预算以下不制造选择，容量安全规则保留。私有 hint 不计入生成 token 计数。
- selection/generation 上限相同；记录平均/峰值 active tokens，因为频繁刷新可能改变实际 attention 工作量。同步报告刷新次数、总刷新时间、总请求时间、输出量和异常冷恢复。
- 计划 GPU 时间：2–6 小时。每步刷新组成本需首轮实测校准；高频选择可能大幅增加耗时。
- 产出：调度成本表。闭环质量由下一阶段补充。

## 四、最小闭环质量实验

上述短重放能解释成本和状态变化；若论文要声称 hint 或调度改善任务质量，还需要真实动作反馈。

先用 OSWorld 012/015/016/018 做机制 pilot。这个集合沿用已有按维护事件和 ID 选择的规则，不代表全数据集；历史四题 KVMem 耗时约 18/24/16/38 分钟，仅作为排期依据。

第一轮做三种 hint 来源、固定当前调度，四题×三组×一个预先固定 seed，共 12 条完整轨迹。所有组使用同一个冻结 binary、采样、模型、评估器和原有任务停止规则。预计 6–12 GPU 小时，失败策略可能更久；不为缩短耗时临时降低动作/时间上限。报告原始官方分、成功率、总时长、输出量、刷新次数及实际 active tokens。

调度闭环作为下一批，固定 query 来源，比较三个调度策略。同样先四题×三组，一个 seed；计划另需 6–12 GPU 小时，不做 query 来源×调度全组合。小样本不够稳定时预先追加共同 seed，并保留全部运行，不能因结果不好换题。泛化结论需要扩大任务集合或收窄表述。

## 五、暂缓项目

- InfLLM/InfiniGen 同资源质量对照：有重要性，但不是最省时的一项。先用一个共同短答案任务做各方法可运行性和耗时校准，再锁定 GPU/Host 上限、实际 KV/索引占用与原评分器，关闭 MTP，做小规模完整对照。外部适配和预算匹配通过前，不承诺总 GPU 时间。
- 全量 active-budget×block-size 网格：优先整理历史可比结果；后续选两个预算和三个 block 大小，避免完整笛卡尔积。
- GC 全量 512K/1M/2M OSWorld：已有全任务压力与 4B 延后问答结果，优先分析。新增预算先在同一 32 题保留实验上补 128K/256K 档，验证质量曲线，再决定是否扩展 27B 闭环。

## 六、推荐首批范围与资源

推荐先完成“不占 GPU 的整理 + P1 + P3”，新增 GPU 预算约 **2.5–6 小时**，可以先交付配对统计、两类刷新分解和 hint 成本三项证据。P2 在选择计划回放接口验证后加入，P4 随后执行；全部短实验约 **5.5–15 GPU 小时**，不含开发。

新的 query-source/schedule/forced-selection 接口需要独立实现与回归，合计先按 **1–2 个工程日**安排；这是估计，当前没有将它们视为现成开关。先交付不需这些接口的统计和 P1。

使用一张 PRO 6000 顺序进行纯延迟测试，保留现有实验的正常运行安排；CPU 分析可并行。冻结独立实验 binary 和完整 dirty diff，不替换在跑服务。每项保存来源、参数、随机种子、脚本、全部重复、失败状态及完整原始结果。不同 GPU 的绝对耗时不合并。上述计划没有启动测试或修改 serving。
