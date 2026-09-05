# 文件历史与 Mean-K 用于 KVMem GC：现有轨迹分析

日期：2026-09-05。结论：文件操作提供有用的候选信号，但“修改后重读”不能直接触发旧 blocks 的永久删除。推荐结合观察范围/版本、未解决错误与回退关系、已有 Mean-K 评分历史及复用统计，在 CPU 上估计删除风险。当前数据足以发现反例，不足以证明某项 GC 策略能提高成功率或降低 re-prefill。

后续设计已按用户要求改为无需训练的[规则策略 v1](/home/chaidi/qw3/docs/kvmem_rule_gc_policy.md)。本报告的轨迹事实与数据边界不变；实施策略以最新设计为准。

## 样本与方法

固定分析 cohort：`kvmem3m_128gen64_full113_claude_pass4_pro_reuse_passmajor_s1000_4000_20260829`，seed 1000，`claude_kvmem_128k_gen64k` arm。canonical 路径中发现 83 个 stream，排除没有 terminal result 的 `tengo-destructuring-bindings`，实际分析 82 条，约 2.05 GB。13 个 canonical 入口复用历史目录；结果 JSON 同时保存入口路径与解析后的原始路径，不能把这些描述为全新重跑的 82 次实验，也不能代表全部 113 个任务或全部 seed。

terminal `success` 仅表示 agent 流正常结束，并非任务验证成功；样本包括 reward 0 和 reward 1。

[脚本](/home/chaidi/qw3/scripts/kvmem_eval/analyze_file_history_gc.py)仅读取 JSONL 的完整 message，跳过重复的流式增量，按 tool call ID 去重，以 tool result 完成顺序统计。模式识别排除报告 `is_error` 的工具返回。没有执行录制中的任何命令，没有启动模型或 GPU。

- 自动识别显式 `Read/Edit/Write`；不把任意 Bash 字符串解释成可靠文件版本事件。因此统计只覆盖可直接识别的操作。
- “修改后重读”指同一路径相邻两次显式 Read 之间存在成功 Edit/Write。统计读取参数及实际显示范围，不假定读到了整文件。
- “相同可见读取”要求同一路径、相同请求 offset/limit、去除行号与工具 reminder 后的返回正文 hash 相同。该 hash 不是完整文件版本 hash，也不证明对应 KV 等价。
- “反向编辑”要求同一路径的两次 Edit 恰好交换非空 old/new 字符串；包含调试代码的添加/移除，不能全部叫作错误修复或回滚。
- Git 命令只用于人工定位基线对照与回退实例，通用 `git diff/checkout` 计数不代表发生了回滚。

复现命令（仓库根目录）：

```bash
python3 scripts/kvmem_eval/analyze_file_history_gc.py \
  --root benchmark/claude_deepswe_ab/recordings/kvmem3m_128gen64_full113_claude_pass4_pro_reuse_passmajor_s1000_4000_20260829 \
  --seed 1000 \
  --output docs/kvmem_file_history_gc_trace_analysis_20260905.json
```

机器可读结果：[统计和逐例源位置](/home/chaidi/qw3/docs/kvmem_file_history_gc_trace_analysis_20260905.json)。`first_line/again_line` 等是源 JSONL 的一基行号；`first/again` 等是脚本按工具完成顺序分配的事件号。

## 统计结果

| 观察 | 次数 | 涉及任务数 | 可以支持的判断 |
| --- | ---: | ---: | --- |
| 成功显式 Read | 2,481 | 82 | 文件观察是频繁的输入来源 |
| 成功显式 Edit / Write | 2,918 / 444 | — | 有丰富的局部变更来源信息 |
| Read → Edit/Write → Read | 285 | 63 | 版本/覆盖关系值得作为策略特征 |
| 上述模式中前后读取参数范围不同 | 264（92.6%） | — | 不能由同一路径推断旧观察全部被覆盖 |
| 严格相同的可见读取 | 6 | 2 | 存在较强冗余候选；此严格子集不能估计总体重复率 |
| 字符串严格反向 Edit | 16 | 7 | 确有恢复旧片段的行为，但含调试清理 |

纯显式工具匹配中，“中间已经重读同文件，再发生严格反向 Edit”为 0；相同请求范围的可见内容 A→B→A 也为 0。不能声称自动计数直接证明了这两个精确模式。修改后重读再访问基线的证据来自下面人工核对的 Bash 实例。

读取参数不同也不一定表示内容不相交或新观察更小；需要逐片段验证覆盖。文件路径相同也不能排除 Bash/外部进程的修改，分析没有恢复完整版本图。

## 三个影响设计的实例

### 1. 局部重读不能替代整份旧观察

`python-statemachine-state-data-scoping` 对 `/app/statemachine/engines/base.py` 的操作：

1. [初次 Read][state-first]：显示第 1–952 行。
2. 事件 94、95、96：修改该文件。
3. [再次 Read][state-again]：请求 offset 560、limit 45，实际显示第 560–604 行。

如果此时删除初次 Read 的全部 blocks，会同时删除新观察并未包含的大部分源码。适当的处理粒度是确认覆盖的内容 spans；还要处理编辑导致的行号移动。该例来自 cohort 复用的 20260826 历史运行。

### 2. 读过修改后的文件，仍然查看原始基线

`clack-async-autocomplete-options` 对 `packages/core/src/prompts/autocomplete.ts` 的操作：

1. 早期读取原文件，随后添加 async autocomplete 相关实现。
2. [事件 117][clack-new] 再次 Read 当前文件。
3. 紧接着的[事件 118][clack-base] 使用 `git show HEAD:...` 查看原始代码、原始文件行数和原始测试条目；[返回][clack-base-result]包含原来的同步 `FilterFunction` 定义。

该时点早于轨迹中的首次 commit，HEAD 仍是任务基线。它表明新版本出现后，旧版本仍可能是比较与核查对象。它不证明保留旧 KV 就能省掉这条 Git 命令，更不证明模型生成此命令是由历史缺失造成；这里只用于反驳“新读到即旧信息失去用途”。

### 3. 实际存在改错后恢复旧片段

`query-persist-restored-query-state` 对 `createPersister.ts` 的操作：

1. [事件 155][query-edit] 给 `persisterFn` 增加 `TPageParam = unknown`，并改变 `QueryFunctionContext` 的泛型参数。
2. 随后进行 TypeScript 检查，[结果][query-fail]包含 TS2345：原测试传入的 context 缺少 `pageParam` 和 `direction`。
3. 查看相关类型定义后，[事件 160][query-revert]把该函数签名恢复为事件 155 之前的文本。

这个实例支持提高未解决修改的旧签名/旧片段和失败证据的保留优先级；并不要求保留该文件每一版的完整输出。现有 Edit 的 old/new 已包含部分回退信息，可以优先保护这些已存在的片段。

这里 Bash 工具的 `is_error` 是 false，但输出有编译错误；命令使用了管道和后续 echo。不能仅凭 shell 退出状态、工具成功标记或文本中的 `_OK` 就认定代码通过验证。运行时应从可靠的测试包装层接收验证结果与覆盖范围，无法判定则标为 unknown。

## Mean-K 能补充什么，现有数据缺什么

[现有 scorer](/home/chaidi/qw3/src/qwen_executor.cpp:23322)针对当前 query 计算各 block mean-key 的相关性，在 layer/query/head 内对 blocks 做 softmax 再聚合。[最终分数已有 D2H](/home/chaidi/qw3/src/qwen_executor.cpp:23539)，可以在 CPU 消费这个既有结果。它是模型表征驱动的当前检索信号，不能直接当作“模型决定可以遗忘”、代码正确性分数或未来访问概率。

值得验证的特征包括：候选集合内的相对排名、距入选阈值的距离、最近高分、跨不同有效 query 的趋势，以及这些特征与版本变化/失败状态的交互。两个相近文件版本可能都高分；块均值可能稀释很小但重要的改动。这些是机制上的风险假设，本次尚未用逐块分数验证。

82 份 `metrics.json` 的语义重选统计：

| 每任务 semantic reselection 次数 | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 任务数 | 23 | 26 | 16 | 8 | 5 | 2 | 2 |

中位数为 1。该指标由[已有日志分析器](/home/chaidi/qw3/benchmark/claude_deepswe_ab/analyze_arm.py:478)统计语义重选，并不覆盖所有逐 block 的评分观测。以 clack 为例，52 个 selection passes 中 51 个为 passthrough，semantic reselection 只有 1 次。工作集沿用不能算新 query 下的低分，未发生评分也不能填 0。

在这批选定 arm 的输出中未找到与文件 spans 对齐的逐 block score history；现有[可选 score dump](/home/chaidi/qw3/src/qwen_executor.cpp:15192)也不能直接等同于完整 GC 训练数据。因此目前不能报告 Mean-K 对错误修改/回退的识别准确率，不能计算候选删除后的未来误删率，也不能宣称联合策略优于 LFU。

下一步需要被动采集：真实 query/selection 类型和 epoch、eligible block 集合、已有 Host score/rank、选中结果，以及工具观察到 token/block 的对应关系。只在既有检索边界采集，不为 GC 额外打分。若数据稀疏，保持不确定性并结合来源/usage；不能假造“连续低分”，也不能无上限等新评分而耗尽容量。

## 对实施计划的调整

1. 在 CPU 策略层新增观察来源关系：对象身份、已知版本、实际返回范围、内容摘要、工具调用和历史 token spans。先识别可靠的显式工具，不做复杂 Bash 语义执行或额外 LLM 推理。
2. 使用有预算的软保护，保留未解决修改的相关旧片段、已有 diff、错误证据及必要基线；验证通过后逐步减弱保护。不可将验证通过等同于永不回看。
3. 把 supersession/duplicate 作为概率特征，与已有分数和复用统计联合；混合块保守处理，一批删除不能同时消除全部替代观察。同样文本的 KV 有不同上下文，仍需质量实验。
4. Shadow 比较 LFU、LFU+score、LFU+来源、三者联合。重点报告部分重读、回退/基线比较、评分稀疏子集的风险。之后才启用真实删除，并测任务成功率、额外重读/工具调用、prefill tokens 与延迟。

来源分析和风险更新在有界 CPU worker 中完成；不额外扫描 payload、不调用 GPU、不生成需要重新 prefill 的摘要、不让前台等待分析结果。队列缺失时降低证据置信度，而不是把缺失视为冷数据。该方案仍需测量 CPU/内存竞争与前台事件发布的增量开销，尚没有“零 latency 增量”的性能结论。

完整并发、slot 复用与容量设计已合并到[设计文档](/home/chaidi/qw3/docs/kvmem_background_gc_design.md)。本次交付是离线证据和设计修订，没有实现或启用运行时 GC。

[state-first]: /home/chaidi/qw3/benchmark/claude_deepswe_ab/recordings/kvmem3m_128gen64_full113_claude_pass4_taskmajor_s1000_4000_20260826/trials/python-statemachine-state-data-scoping/seed-1000/runs/python-statemachine-state-data-scoping/claude_kvmem_128k_gen64k/claude.stream.jsonl:933
[state-again]: /home/chaidi/qw3/benchmark/claude_deepswe_ab/recordings/kvmem3m_128gen64_full113_claude_pass4_taskmajor_s1000_4000_20260826/trials/python-statemachine-state-data-scoping/seed-1000/runs/python-statemachine-state-data-scoping/claude_kvmem_128k_gen64k/claude.stream.jsonl:47183
[clack-new]: /home/chaidi/qw3/benchmark/claude_deepswe_ab/recordings/kvmem3m_128gen64_full113_claude_pass4_pro_reuse_passmajor_s1000_4000_20260829/trials/clack-async-autocomplete-options/seed-1000/runs/clack-async-autocomplete-options/claude_kvmem_128k_gen64k/claude.stream.jsonl:88482
[clack-base]: /home/chaidi/qw3/benchmark/claude_deepswe_ab/recordings/kvmem3m_128gen64_full113_claude_pass4_pro_reuse_passmajor_s1000_4000_20260829/trials/clack-async-autocomplete-options/seed-1000/runs/clack-async-autocomplete-options/claude_kvmem_128k_gen64k/claude.stream.jsonl:90709
[clack-base-result]: /home/chaidi/qw3/benchmark/claude_deepswe_ab/recordings/kvmem3m_128gen64_full113_claude_pass4_pro_reuse_passmajor_s1000_4000_20260829/trials/clack-async-autocomplete-options/seed-1000/runs/clack-async-autocomplete-options/claude_kvmem_128k_gen64k/claude.stream.jsonl:90713
[query-edit]: /home/chaidi/qw3/benchmark/claude_deepswe_ab/recordings/kvmem3m_128gen64_full113_claude_pass4_pro_reuse_passmajor_s1000_4000_20260829/trials/query-persist-restored-query-state/seed-1000/runs/query-persist-restored-query-state/claude_kvmem_128k_gen64k/claude.stream.jsonl:72623
[query-fail]: /home/chaidi/qw3/benchmark/claude_deepswe_ab/recordings/kvmem3m_128gen64_full113_claude_pass4_pro_reuse_passmajor_s1000_4000_20260829/trials/query-persist-restored-query-state/seed-1000/runs/query-persist-restored-query-state/claude_kvmem_128k_gen64k/claude.stream.jsonl:72880
[query-revert]: /home/chaidi/qw3/benchmark/claude_deepswe_ab/recordings/kvmem3m_128gen64_full113_claude_pass4_pro_reuse_passmajor_s1000_4000_20260829/trials/query-persist-restored-query-state/seed-1000/runs/query-persist-restored-query-state/claude_kvmem_128k_gen64k/claude.stream.jsonl:77746
