# 长桌面任务的 KVMem 查询刷新诊断

2026-10-05。共同显存配置035第一轮ON的实际日志显示：长工具任务保持同一用户目标、guided refresh关闭时，服务端把自动查询重选抑制为off。随后冷预填采用有界pressure窗口，执行器没有捕获查询，Mean-K回退window-profile。现有八次对照继续保持原配置；本诊断不改变其成绩或条件。

## 实际证据

来源为`quota1-035-kvmem-on-r1`的终止引擎日志。原引擎3486935和runner3487549的登记身份与实际退役已经独立核验；原结果仍为官方评分器不可用、辅助本地0、accepted=false。

| 范围 | 请求数 | scorer事件 | 窗口回退 | 捕获查询 |
| --- | ---: | ---: | ---: | ---: |
| 全部输入仍在131072窗口内 | 16 | 16 | 16 | 0 |
| 输入已超过131072 | 7 | 79 | 79 | 0 |

前16次事件选择了全部source blocks，不能据此声称历史丢失。后7个请求全部记录continuation gate的reselect=off、有效服务端query span与pressure-prefill；其中6次产生79条scorer事件，query_expected/query_captured均0，第7次在模型请求取消前没有scorer事件，不推断其未记录行为。故障不只是查询坐标缺失：服务端知道span，却没有授权执行器做当前查询选择。

首个超过窗口的原始请求为181074输入token，服务端query span为[3944,13218)，score query message为3；原始请求没有指定kvmem_reselect或guided参数。原请求沿服务默认auto进入，而延续相同用户查询的gate随后将其改为off。原启动配置guided_refresh=off、prefix_reuse=false保持。

080-ON随后自然终止，官方0分，1802.68秒/34工具调用，deadline、未知输入0、录像收集通过；原runner/engine已实际退役。完整104条scorer事件中29条选择全部历史，超过窗口75条全部回退、查询捕获0；六个above-budget请求中五个产生scorer事件。活动前缀`kvmem-080-query-gate-active-prefix-01.json`保留；终止证据为`kvmem-080-query-gate-terminal-01.json`及`shared-quota-first080-on-terminal-01.json`。快照16确认同组OFF已终止：官方0分、1152.71秒/27调用、未知输入0、录像收集通过，agent因持续无法操作而暂停。两侧失败且停止原因不同，耗时差不能解释为提速；本路径也不解释其他任务在短历史阶段的失败。

源码`kvmem_harness_refresh_decision`在同一用户查询且guided refresh关闭时返回不重选。服务端据此设置reselect=Off，`kvmem_query_lifecycle`不会进入select_active；prefix reuse关闭也不产生warm capture。recompute-query虽在当前EngineOptions默认true，仍被这些前置条件限制。日志中mtp标签不足以把故障归因于DFlash。

## 已完成的资格与工具

新增`kvmem_retrieval_diagnostic.py`按真实请求边界区分全历史可容纳与超过窗口、服务端span与执行器query、正常最终计划与不完整活动行；仅导出结构、计数和源哈希，不导出任务或查询文字。十项检查通过，包含真实query_expected/query_captured是行数而非布尔值的回归、部分日志、缺字段、重复序号、跨请求gate污染与成功查询反例。

新CPU策略测试复用原refresh/request-plan头文件，在上述181074、131072、65536、28672实际配置上通过14项检查。现有两个C++策略测试另通过。证明启用现有both模式可让该持续增长轨迹进入private-query刷新，并保持短回调的有界宽限、原窗口/生成/输出预算；不证明GPU查询捕获、业务正确性或延迟改善。仅boundary模式在此调用中不提供middecode增长阈值，不能与both的私有查询行为混为一谈。

第一次新C++检查因测试类型拼写错误编译失败，原源码与日志完整保留；修正测试后通过，没有修改生产策略。初次解析器按布尔值限制查询行数，源复核后增加真实行数反例，旧版及九项资格保留，当前十项通过。

证据目录均相对`apps/computer_use/.runtime/`：

- `kvmem-query-gate-terminal-diagnostic-02.json`：当前解析器的完整终止日志结构核验。
- `kvmem-query-gate-terminal-lineage-01.json`：来源进程实际退役与首份诊断哈希。
- `kvmem-query-gate-qualification-01/`、`02/`、`03/`：全部初始失败、通过、精确源码和当前资格。

## 独立真实模型资格的准备

`kvmem-query-refresh-saved-input-plan-01.json`已准备，尚未派发或登记等待器。两侧每次新引擎，顺序control/candidate/candidate/control；先发送原13225-token请求，再发送原181074-token请求，均保留原始请求字节、thinking与16384输出上限。两份档案已无损恢复并核验root task与score query的消息哈希一致；模型生成的工具调用只记录、零执行，不向桌面输入。

control保持guided-reselect=off；candidate使用现有both，其他模型、DFlash、FP8、窗口、quota1、历史图像和prefix reuse配置相同。须在现有冻结模型流水线完整终止及退役、获得原串行锁、用户应用与网关空闲、满足GPU余量后才可执行；当前计划不是已运行试次。

这一步先验收control复现known-conversation抑制、candidate真正捕获私有查询并执行非回退查询评分。单独将长请求发给冷引擎属于ColdAboveBudget，不能复现本次问题；不能用那种试验假装已修复桌面延续流程。即使机制资格通过，仍需同条件真实桌面任务证明质量/速度改善。

## 独立运行绑定与当前状态

准备计划原文件保留为prepared_not_dispatched。随后新建不可覆盖的`.runtime/query-refresh-readonly-20261005/protocol.json`，明确KVMem两侧均开启、仅guided refresh为off/both，顺序off/both/both/off；每轮先原13225-token请求，再原181074-token请求，要求完整流、同一引擎、原输入token及退役回执。工具调用仅记录，零执行；初次请求或身份失败不发送第二份请求。

HTTP九项、原启动器九项、严格串行队列十三项资格全部通过，共31项，均无QW3或桌面输入；首次沙箱TCP绑定失败、早期测试源码及队列资格分别保留。实际等待器PID3585544/UID1001/start_ticks373428016已经登记，快照16确认存活、0/4，等待原40＋新KVMem8＋SDK10＋图像12＋thinking4＋反馈8＋有限续轮8完整结束与退役。各级源哈希、结果库存、VM退役、未知输入及最后自有引擎检查保持，结束后恢复原OFF用户入口。未修改现有运行程序；当前没有模型机制改善、业务提升或延迟收益。完整绑定见[运行声明](computer_use_query_refresh_runtime_20261005.json)。

快照18更新：原KVMem队列在5/8后遇到下一次OFF启动前的端口占用检查失败，未派发任务；全部后继因前序停止而自然退出，PID3585544已退役、0/4。独立端口续跑仅保留五份结果并执行余下三次，实际已运行；查询刷新及其他后继尚未重新绑定。上段16的存活状态是保留的历史快照，不能作为当前运行依据；原协议、资格与失败仍保持。


2026-10-05第二轮035终止只读复核：ON为1801.47秒/16调用，OFF为1802.74秒/22调用，均到原总期限、官方评分不可用、本地0、未知输入0、录像收集通过。ON整体日志187条评分，其中超过窗口的五个请求共175条，全部window-profile回退、查询捕获0；四个压力请求具有明确同查询gate抑制证据。日志含启动健康探针，不把187条当作任务请求数。原八项协议字节在两个续跑根目录一致，两个原result哈希独立核对；证据`.runtime/shared-quota-second035-terminal-pair-01/{result,on-retrieval-routing}.json`。新查询等待器3653494已实际登记，原四项saved-input输入/模型/顺序不变、0/4，等待完整前序终止；旧3585544停止状态保留。新的41项路由资格不是实际查询机制改善，第二轮两侧调用数差异也不是质量或加速结论。


## 2026-10-05 最新终止状态核对（快照22）

共同配置 KVMem 对照已完成 8/8：080 四份官方评分均 0，035 四份官方不可用、本地均 0；未观察到任务完成收益。最后 080-ON 第二轮没有超过检索预算，因此不能评价长上下文查询检索收益。随后 SDK 启动新 OFF 引擎时因端口占用而失败，尚未创建引擎运行目录或派发模型任务，六组后继均停止、零试次，登记进程已退出。原端口在核对时可用，但失败时占用者未知。原 ON 引擎由已登记 SDK 安全退役，所有失败及完成结果保持。新的恢复入口尚未启动。这是基础设施中断，不是模型任务失败，也不记作流程优化收益。详见[最新八次结果](computer_use_kvmem_same_quota_results_20261005.json)和 `.runtime/shared-quota-current-progress-20261005-22.json`。
