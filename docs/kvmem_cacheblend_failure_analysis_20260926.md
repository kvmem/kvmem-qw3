# 256K、10% 重算剩余六例失败的诊断

正式测试仍为 **54/60，90%**，剩余六例全部是多键干扰 `niah_multikey_2`。诊断发现：六例都没有把包含目标名称的关键 block 召回到 32K 窗口，紧接其后的 block 却全部被召回。重算只能更新已在窗口内的内容，不能补入漏召回的证据。

本次完成 24 次有效诊断运行，使用正式测试同一冻结二进制，SHA256 为 `48b6415c4adfad971dd4a55896d59ea029a1861a7d8055df8aa834c2214f29cd`。未修改生产执行代码；下面的证据补齐使用已知答案位置，仅作原因诊断，不能计入正式准确率。

**证据在哪里丢失**

路径为：256K 历史 → 检索出约 32K 窗口 → 在窗口内部挑选 10% 重算 → Query Replay → 回答。[重算候选构造](/home/chaidi/qw3/src/kvmem_blend.cpp:85)只遍历已入选的窗口块。

下表 block 编号为历史 token 起点除以 32，排名为所有有效历史块的检索分数排名。六个缺失块均仍保存在 CPU tier，`nt=32`、`sel=0`，不是被仓库删除，也不是重算找不到源 token。其后继块均 `sel=1`，且按既有策略重建的重算名单显示均被重算。

| 种子 | 深度 | 缺失目标名称块 | 该块分数排名 | 后继块分数排名 | 改用原问题后，该块排名／是否召回 |
| --- | ---: | ---: | ---: | ---: | --- |
| 1000 | 70% | 5734 | 1165 | 1 | 915／否 |
| 2000 | 10% | 821 | 2632 | 2 | 1530／否 |
| 2000 | 30% | 2459 | 5991 | 1 | 3695／否 |
| 2000 | 70% | 5734 | 5216 | 1 | 2602／否 |
| 3000 | 10% | 821 | 776 | 1 | 127／是 |
| 3000 | 70% | 5734 | 4481 | 1 | 2330／否 |

窗口有 1024 个块，其中 sink/recent 预留 41/328 个，因此全局排名 776 也不保证入选。重算是在窗口内另做一次选择，并跳过未变化的连续前缀和正常 Query Replay 负责的后缀。

例如 seed=2000、depth=10% 的事实恰好跨块：未召回的 block 821 以 `annoying-game is: 2` 结尾；已召回且重算的 block 822 以 `292604.` 开头，后面紧接其他名称的事实。窗口里只有数字后半截，没有把名称和完整数字绑定的文本。重算这个后继块不能恢复丢失的前半句。其他案例中，后继块甚至只包含事实的末尾换行和下一条无关事实。

当前能确认的是“目标所在块低分、后继块高分、事实被截断”的现象。尚未逐层逐头追踪分数来源，不能据此断言存在全局 block 索引错位，也不能把原因完全归结为某种神经网络状态传播。

**控制实验**

| 控制条件 | 输入／状态路径 | 六例答对数 |
| --- | --- | ---: |
| 复现原配置 | 冷启动盲 ingest 全部 256K，guided query，10% 重算 | 0/6 |
| 使用原始问题的 Query | 相同 256K 和配置，仅关闭私有 guided query 生成 | 1/6 |
| 实际召回文本完整 prefill | 从实际选择名单重建 32768-token 输入，全新状态完整 prefill | 0/6 |
| 补齐缺失证据后完整 prefill | 同样 32768 tokens，只替换一个 32-token 块，全新状态完整 prefill | 6/6 |

前两组各次都实际重算 102 块、3264 tokens，`missing_source_tokens=0`，执行了 Query Replay。使用原始问题只修复 seed=3000、depth=10% 一例，因此 guided query 不是全部失败的唯一解释。

后两组共同关闭 blend、guided query 和再次检索；都从 fresh GDN/conv/attention 状态完整 prefill。补齐组加入缺失的一个 32-token 证据块，并删除一个最低检索分数的普通入选块，排除 sink、recent、证据和 query 块；两组输入长度均为 32768 tokens，末尾问题和 assistant 前缀相同。原窗口历史文本六例均没有目标名称或完整事实，补齐后六例均包含完整事实并答对。

这个等预算的内容对照说明缺失证据是关键瓶颈。它**没有**验证“在原历史 KV 复用路径里补一个块即可六例全对”：两组都重新完整 prefill，已知证据位置也不是可部署的召回策略。只有在真实检索路径实现并重新测试后，才能报告修复后的准确率。

24 次均正常 `stop`、thinking 闭合，最多生成 778 tokens，没有触及输出/thinking 上限或发生 thinking compaction。诊断耗时不用于替代正式测试的延迟统计。

**Guided query 的问题**

原多键测试在深度 10/30/50/70% 时，私有 guided query 生成的是一个错误数字：seed=1000 为 `8808463`，seed=2000 为 `5459526`，seed=3000 为 `8552350`，而不是保留目标名称的查询文本。这些数字对应开头的干扰项。原问题的 16/18 个 Query tokens 随后被 7 个生成数字 tokens 对应的 Query 表示替换；90% 深度时生成数字才匹配目标答案。

这个现象也发生在一些答对样例，且生成数字的隐藏表示仍可能携带原问题信息，所以不能仅凭表面文本认定 Query 完全丢失了目标语义。关闭生成后六例目标块排名均改善，但仅一例进入窗口，说明还需解决块选择对完整事实的覆盖。

继续追查生成与发布路径，确认了以下机制：

- 配置是 `--kvmem-guided-reselect both`、`--kvmem-guided-thinking-tokens 0`，未启用 `query_guided_direct`。实际进入[零 thinking 分支](/home/chaidi/qw3/src/qwen_native_backend.cpp:15838)，提示词要求根据“original coding task”“recent tool results”生成查询，保留“file names, symbols, errors, tests”。这是编程场景提示，与当前名称—数字检索任务不匹配；该分支也没有像 direct 分支那样单独嵌入 `SOURCE REQUEST`。完整提示字面量在冻结二进制中存在，加上关闭 thinking 的前缀恰为 108 tokens，与全部 15 例运行日志一致。
- Guided query 在首次语义召回前生成。日志显示此前压力窗口为 41 个 sink 块加最近尾部，`first_non_sink=7208`；结合压力选择策略，可重建历史区间为 `[0,1312)` 与 `[230656,262112)`，随后处理原问题后缀。10/30/50/70% 的目标事实都不在这段可直接 attention 的历史中，90% 的事实则在其中。GDN 仍可能携带先前历史信息，因此不能说模型完全没有历史信息。观察上，前四个深度全部生成开头干扰项的数字，90% 深度才生成正确数字；尚未做只替换提示词的干预实验，不能把这种相关性当作全部机制的因果证明。
- [当前发布校验](/home/chaidi/qw3/src/qwen_native_backend.cpp:16067)只要求文本非空、自然结束，并排除部分 JSON/代码/工具等格式；没有检查是否保留查询名称，也不拒绝纯数字。因此 `5459526` 这种输出会被当作有效检索请求。冻结运行日志直接证实这类数字输出被接受，未回退到原问题。
- [发布函数](/home/chaidi/qw3/src/qwen_executor.cpp:18343)将生成 token 的 Q rows 复制到 `g_query_multi_`，重设 query span/count，替换原问题的检索 Q，而非合并两者。以 `annoying-game` 为例，原问题捕获 16 tokens，最终打分使用 7 个数字 tokens 对应的 Q。模型状态回滚不撤销这次刻意发布；之后检索、重算和 Query Replay 才依次发生。因此本次错误改写先于 blend 的 GDN 清零，不能由该清零解释。

15 例中 12 例生成错误数字，其中 6 例最终仍答对；这些是带原上下文的 Q 向量，不是对数字做字面匹配。应区分“改写文本不合任务要求”和“该向量必然无法召回”。改进时可以让简单的名称查找直接使用原问题 Q，或保留现有回退路径并补充实体保留校验；仅换提示词尚不足以保证正确。代码与日志核验记录见 [guided-query-path-audit.json](/data/chaidi/qw3-experiments/cacheblend_20260925/diagnose256k_failures_20260926/guided-query-path-audit.json)。本次追查未修改生产代码或再启动 GPU 测试。

**GDN 状态的实际维护方式**

[当前实现](/home/chaidi/qw3/src/kvmem_blend.cpp:154)先保存 Query 边界状态，再在段循环外把各层 recurrent/conv 状态清零一次。重算块按窗口位置递增执行；只有历史位置和窗口位置都相邻的块才合并成同一次 `forward_n_tokens`。每次 forward 后的临时 GDN 状态继续供下一个选中段使用，段间不清零。

例如只选 B2、B10、B30，则临时状态依次经历 `0 → B2 → B10 → B30`，跳过的 B3–B9 等不会推进状态。全部结束后恢复原 Query 边界 recurrent/conv 状态及相关执行状态，保留重算后的普通 attention KV 覆盖供 Query Replay 使用。因此这不等价于完整 prefill 全部窗口。该近似仍存在，但本次证据不支持把六例失败全部归咎于 GDN：在两组都完整 prefill 的控制中，仅补齐文本证据就使结果从 0/6 变为 6/6。

**后续优先方向**

优先在现有 32K 预算内保障高分块的事实连续性，例如为高分块携带一个前邻块、对相邻块成组取舍，再让已有重算合并逻辑处理连续段；同时避免 guided query 仅输出猜测数字而不保留查询名称。这里是待验证的实现方向，尚未作为修复落地。邻块策略可能挤掉其他有用内容，需要完整四类 60 例回归。

**2026-10-16 标记待尝试：原问题 Q＋guided Q 双路召回，增加实体保留校验**

- 状态：用户选定为后续实验候选；先记录，尚未实现或启动实验。
- 方向：保留原问题的检索 Q，与 guided Q 分别评分，在同一总 KV 预算内合并选块，避免改写直接覆盖原问题信号。
- 校验：检查改写是否保留原问题中的关键名称、日期、文件名等实体与约束；校验失败时回退原问题 Q。具体校验规则和合并策略待设计。
- 验证：固定输入、KV 预算和其他策略，对比原问题 Q、现有 guided Q、双路召回，以及双路加校验；同时记录证据块召回、答案质量和查询生成／评分开销。

单纯增加重算比例不会把窗口外的缺失块加入候选，也无法直接解决本次观察到的证据截断。正式准确率保持 90%，不把使用已知证据位置的 6/6 诊断结果外推为系统已达 100%。

原始文件位于 `/data/chaidi/qw3-experiments/cacheblend_20260925/diagnose256k_failures_20260926/`：

- [验收审计、六例块排名及哈希](/data/chaidi/qw3-experiments/cacheblend_20260925/diagnose256k_failures_20260926/acceptance-audit.json)
- [256K Query 对照原始结果](/data/chaidi/qw3-experiments/cacheblend_20260925/diagnose256k_failures_20260926/query_ablation/results.json)
- [32K 证据补齐对照原始结果](/data/chaidi/qw3-experiments/cacheblend_20260925/diagnose256k_failures_20260926/compact_evidence/results.json)
- [15 例多键 guided query 审计](/data/chaidi/qw3-experiments/cacheblend_20260925/diagnose256k_failures_20260926/guided-query-audit.json)

最初尝试通过 raw `/v1/completions` 强制指定 oracle 证据块，但冻结服务该端点不解析相关字段，选择断言及时失败。该尝试保留在根目录并由 [attempt-note.json](/data/chaidi/qw3-experiments/cacheblend_20260925/diagnose256k_failures_20260926/attempt-note.json)标记无效，未纳入上述 24 次有效运行。重算名单由 9 位有效数字的检索分数重建，实际日志另行核验重算量；缺失块 `sel=0` 则直接来自检索 dump。
