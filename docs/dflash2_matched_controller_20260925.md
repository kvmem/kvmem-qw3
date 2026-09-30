# 同策略 KVMem 的 none / MTP / DFlash 控制器

本阶段补齐正式基线所需的执行路径，目标权重仍为 Q8_0，KV 仍为 E4M3 FP8。
功能和质量验收仍在进行；目标未完成，尚无正式性能或公开水平结论。

此前 DFlash 可以通过完整事务控制器运行联合验证，但原生 none/MTP 的连续
批处理测试禁用了 query-conditioned 检索。guided query 和中途 refresh 则
进入串行事务控制器。因此，这些历史计时不能组成同策略的性能对照。

新增可选 `QW3_CONTINUOUS_KVMEM_CONTROLLER=1`，让 none/MTP 也使用完整控制器。
请求保留独立的执行器、RNG、检索窗口、回放和回退状态；调度器依次恢复请求，
在安全边界联合执行目标模型。none 的单 token 过渡及 MTP bridge/tail 请求
均可进入联合目标批次，且无需分配投机 checkpoint。

原生 MTP 草稿按深度联合执行，沿用现有批量 MTP 层和窗口元数据。
剩一个请求时运行独立草稿步骤；各请求遵守自身的草稿长度、位置和输出预算，
草稿完成后恢复临时窗口页表。控制器仍承担接受、拒绝、前缀重建和刷新。
`QW3_CONTINUOUS_MTP_BATCHED_DRAFT=0` 可保留逐请求草稿对照。
新入口默认关闭；原有 DFlash 开关与普通连续批处理路径保留。

新增测试选项 `QW3_CB_PROBE_MATCHED_KVMEM=1` 让所有路线使用相同的 CPU 索引、
query-conditioned 检索、查询起止位置、强制重选和 guided/refresh 设置。
`QW3_CB_PROBE_SERIAL_CONTROLLER=1` 提供串行控制组；`QW3_CB_PROBE_GREEDY=1`
保留同一组惩罚和 token 限制并使用确定性采样。输出转录独立于计时日志保存。

当前已关闭的模型检查共 544 个响应：

| 运行清单后缀 | 路线 / 场景 | 响应数 |
| --- | --- | ---: |
| `smoke01` | none/MTP/DFlash，C4 重复输出与取消 | 48 |
| `greedy_serial01`、`greedy_batch01` | none/MTP × 串行/批量，greedy | 64 |
| `sampling01` | none/MTP，C4，CPU/GPU 随机采样 | 64 |
| `c16_01` | none/MTP，C16，GPU 采样、反向启动与取消 | 128 |
| `draft_parity01` | MTP，逐请求/联合草稿，greedy | 16 |
| `replenish01` | none/MTP，4 个活动槽、8 个请求，外部取消 | 64 |
| `guided01` | none/MTP，引导查询回退，CPU/GPU | 64 |
| `refresh01` | none/MTP，中途刷新回退，CPU/GPU | 64 |
| `greedy_dflash_serial01`、`greedy_dflash_batch01` | DFlash 串行/批量，greedy | 32 |

完整清单名为 `benchmark/dflash2/q8_matched_controller_<后缀>.json`。
各运行的源码与二进制身份相同；逐次命令、环境、日志和转录哈希均被保存。
只有零退出且状态为 passed 的已结束运行才算通过。35 项主机测试通过；
CUDA 连续测试程序、CLI 和后续串行/长历史测试程序编译通过。
重新构建后者不代表其历史测试已经重新执行。

none/MTP/DFlash 的 96 个串行/批量 greedy 响应，其完整文本和逐次回调逐字节一致。
六个转录 SHA-256 相同，回放边界、suffix 长度、冷/热状态和固定上下文块数
的记录也相同。串行 none 沿用 `generate_plain`，其日志名含 `(plain)`；
两种日志格式均被核对。串行对照保留全局 `QW3_CONTINUOUS_BATCHING=1`，
只关闭请求级批处理，因此内部数值路径相同，不代表默认串行内部 BF16 运算
与批处理的全部质量门槛已经关闭。

日志确认实际联合执行：C16 none/MTP 分别有 220/72 个多请求目标批次，
MTP 有 288 次多请求联合草稿步骤。MTP 草稿切换对照验证最终 greedy 输出，
不据此宣称所有联合草稿内部张量或提议逐项完全相同。

`guided01` 的 64 次查询全部因生成未结束而走 `fallback-original`。
`refresh01` 共有 56 次刷新走原查询回退，另外 8 个响应在第 11 次回调取消，
未到第 13 token 的刷新触发点。这些通过的是对应回退及取消路径，
完整生成查询和长时间多次刷新仍需单独验收。

已结束证据汇总为 `benchmark/dflash2/q8_matched_controller_summary01.json`。
`summarize_q8_matched_controller.py` 核对全部 544 行转录的请求号、回调数量、
同路线重复输出、CPU/GPU 分组、六路 greedy 输出与回放记录、FP8 K 重建策略、
实际联合执行和调度阶段计时加总，并校验每次运行的源码、二进制和数据哈希。
`update_q8_matched_controller_validation.py` 在备份上阶段清单后登记本阶段证据；
历史测试仍保留原始源码身份，不被改标为当前源码的全量验收。

这组输入为约 800 token 历史、256 token 工作集的短回放诊断，不等价于
4K/32K/128K 的最终矩阵。完整 prefix/session/archive/repository/compaction
并发生命周期、长历史质量、正式独立性能测量、公开实现可比性和最终集成仍未关闭。
诊断构建期间部分首轮运行与主机构建重叠，不能用于独立性能结论。

复现入口：`benchmark/dflash2/run_q8_matched_controller.py --run-id <新ID>`；
可选 `--routes none mtp dflash`、`--policy query|guided|refresh`、
`--concurrency 4|16`、`--greedy`、`--serial`、`--replenish`、
`--external-cancel`、`--draft-parity`、`--gpu-only`。
