# Action attention：先跑历史 DeepSWE16

用户调整后的顺序：原 16 个 DeepSWE 样本 → 3 次独立两轮 DSH CSGO。
本轮不运行大海捞针、LongMemEval-S、MAB，旧队列仍保持暂停。

实现与验证见 `runtime/action_pressure_20261010/validation-summary.json`：
3 项主机测试和真实 Qwen 模型状态验证通过。固定 teacher-forced 窗口中采集
开销中位数为 width1 的 0.061%、width8 的 0.200%；容量排序增加约 24.8 微秒。
这些计时不包括窗口搬运或 guided query 的生成成本。

新组使用 Qwen3.8-27B Q8_0 + DFlash2 BF16、FP8 KV；KV budget 36,864、
gen reserve 16,384、thinking 4,096、sink 1,312、recent 0。gen reserve 是生成
期间 KV 预留，不是整个回答的输出 token 上限。无重算、无语义扩展、无文件版本规则。
guided=both，私有 query cap 4,096，增长触发 8,192，每请求最多 8 次。
普通相关性重选先保护必要块，再分配 50% Query / 50% 历史 action attention。
容量回收在保护 sink/必要块后，对当前 resident 候选全部使用 action attention 排序；
不足时按最近块补齐，不从冷历史额外召回。

action attention 仅采集工具动作，排除 thinking、私有探针和拒绝的草稿。
容量回收读取已提交历史和本轮已接受动作的只读分数，不额外提交或重复衰减。

DeepSWE 的任务顺序、instruction、32 个 agent/verifier 镜像、Claude Code 2.1.148
和正式验证器复用历史第一批 16 题。每题 fresh container，seed=1000，每题一次。
客户端不 compact，effort 沿用 xhigh；新 agent 仅增加 MAX_THINKING_TOKENS=4096，
对齐用户指定的 thinking budget。

历史组为 KV128K/gen64K/thinking28K、temperature1.0；新组 temperature0.6，
代码和存储实现也有差异。因此历史比较不是 attention 的单变量消融。
历史组 16 题全部有验证器结果，5/16 通过；新结果尚未完成时不得用它代替新组成绩。

冻结二进制 SHA256：`1a924738b6e74e6bf330bb0c6155459913c711f0dfd3ec1e75e5c30401b27905`。

入口 `benchmark/action_pressure_goal_20261010.py`；串行队列状态为
`runtime/action_pressure_20261010/queue-state.json`。
DeepSWE 数据在 `benchmark/action_pressure_first16_warmfix_20261010`，完成结果保存在
`results.json`，逐任务验证器结果和完整轨迹在 `round-1/*/attempt-*`。
`tool-action-comparison.json` 去重统计已完成工具结果中的 is_error，分别记录 Edit、
old_string、其他错误与 Bash 非零退出；另记录真实 guided/action/pressure 触发次数。
只对相同已完成任务与历史配对，不过滤失败答案，不重跑已完成的 reward=0。
推理或执行基础设施故障保留 attempt 并暂停队列，避免自动挑选成功轨迹。

CSGO 根目录为 `runtime/dsh_tool_guided_actionpressure50_20261010`。
每次使用新空目录与新 DSH 会话，先运行“帮我写一个网页版的csgo”，完成第一版后
在同一会话发送“帮忙添加更多功能，例如更多地图、更多敌人、武器购买系统”。
CSGO seed 沿用此前20260722，其他上述引擎配置相同。完整、失败及中断分别报告。
未启动前不会创建会话或改动 DSH 默认 provider；其他用户服务不停止。

## Warm 复用身份修复

初次未完成轨迹保留在 `benchmark/action_pressure_first16_20261010`，不进入正式
通过率/工具率。真实模型回归证实旧实现 restore+truncate 后丢失稀疏窗口块
身份，KeepSelected 请求会把 attention 错记到连续最近历史。修复后只在 action
tracking 开启时保留/裁剪有效窗口身份，过期评分索引仍失效；不新增 QK/同步。
模型回归与3项主机测试通过。新16题使用独立目录和冻结新版本，预算/提示词/
模型/任务/seed不变；不是从多个答案里挑选成功结果。原始采集和排序计时仍
保留其原测量范围，修复本身只影响请求边界主机元数据。

## 实时进度界面

DSH 页面右下角「实验进度」打开同源的 `/experiment-progress/`。
服务端入口为 `http://localhost:3080/experiment-progress/`，独立入口为
`http://localhost:8324/`。浏览器显示的临时转发端口不应作为服务端链接；
此前提供的64601入口无法访问，已更正为实际监听端口。每5秒刷新，
显示16题状态、正式任务评分、逐题输出/报错、全部已完成工具结果的错误率、
真实 guided/action/pressure 触发次数和后续 CSGO 完成数。未完成题目不进入
正式准确率；工具率包括当前正在执行题目已产生的结果，并按唯一工具ID去重。

只读服务 `qw3-action-progress-ui-20261011.service` 监听本机8324，复用
`scripts/kvmem_eval/thinking_dashboard.py` 的增量日志读取和请求解析。
入口代码为 `benchmark/action_pressure_dashboard_20261010.py` 与同名 HTML。
DSH 通过 `benchmark/action_pressure_dsh_progress_20261010.patch.yml` 挂载只读
路由和链接，不注册工具、provider 或提示词。仅 DSH 网页服务为挂载入口重启，
实验控制器与模型没有因界面更改重启。原服务配置留存在
`runtime/action_pressure_20261010/progress-ui-operations/dsh-web.before.service`。

验证记录在 `progress-ui-validation.json` 与
`progress-ui-operations/browser-validation.json`：统计与现有正式审计一致，
重复工具结果只计一次；浏览器验证逐题切换、报错标签、跟随当前、自动刷新、
桌面/手机布局，脚本错误为0。增量读取一次约2.7毫秒，仅说明页面后端自身的
测量开销；不作为推理加速或端到端性能结论。
