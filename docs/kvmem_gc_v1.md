# KVMem GC v1 实现与验证

2026-09-05。本版本实现了可显式启用的 CPU 后台回收线程，并接通在线 K/V NVMe 槽位回收、检索空洞过滤和 checkpoint 生命周期。默认关闭，不训练模型。原始策略讨论见 [规则策略](kvmem_rule_gc_policy.md)，完整目标与后续容量解耦见 [设计分析](kvmem_background_gc_design.md)。

## 当前可用范围

支持普通串行请求、隐式 prefix cache、immutable raw-K NVMe、CPU Mean-K index、单 prototype、top-k、inclusive stage-out。请打开 `--kvmem-prefix-cache` 让连续请求复用同一历史；每次都 cold reset 的服务没有跨请求的 GC 积累。

MTP、named cache、archive、continuous batching、semantic groups/quota、其他 retrieval method 和 legacy CPU Mean-K scorer 不在本版本支持范围，相关组合会显式拒绝。GPU 常驻索引尚未接入任意空洞掩码。

实际运行的判断信号是最近使用、衰减使用频率、已有 Mean-K 评分的相对排名，以及安全引用。**文件重复/新版本覆盖和有预算的回滚语义锚点尚未接入 serving**：纯规则函数保留这些输入，但当前运行时不会自行设置它们。现有工具协议没有可靠的逐 block 文件版本、读取范围和验证结果，因此内容关系按 unknown 处理；不会仅凭同路径 Read→Edit→Read 降级旧历史。已有协议层的 system/root-task/当前查询等必保区间通过 lease 保护。

这是可执行的保守版本，不代表完整设计中所有内容规则和固定容量长期追加均已实现。

## 规则与默认参数

在硬保护之外，越低的保留分数越先回收：

```text
R = 2^(-(step - max(birth, last_use)) / 8)
F = min(decayed_use_frequency / 4, 1)
Q = 最近至多 3 个有效 step 的相对评分百分位，按 8 steps 半衰期衰减后取最大
retention = 20 * level + 19 * max(R, F, Q)
```

| 条件 | 当前等级/行为 |
| --- | --- |
| GPU 驻留、当前/待安装工作集、checkpoint、mandatory span、sink/recent、未封口、I/O 未完成、K/V 未完整落盘、尚无已发布索引 | 硬保护 |
| CPU V 缓存仍持有副本 | v1 硬保护；先由原有缓存策略淘汰，GC 再处理 NVMe authority |
| 产生或使用不足 4 steps；衰减频率 ≥ 2；近期高分或接近入选边界 | L3，暂不回收 |
| 近期 3 个不同 step 的有效评分连续落在后 20%，未选中且远离边界 | L1，优先回收 |
| 其他冷历史，包括无评分/陈旧评分 | L2，普通候选 |

相同分数先取更久未用者，再用稳定 block ID 打破平局。分数是启发式优先级，不能解释为概率。使用计数按实际使用 step 合并，后台采样不把同一步的多个 token 当作多次使用；过快的事件合并可能低估频率，recency 和 lease 仍独立保护当前使用。

Mean-K 只消费原有 scorer 已回到 Host 的结果。排序和百分位计算在后台进行；tied ranks 使用平均排名，全体同分、非有限值、未参与评分的块不会提供低分证据。同一步重评不增加低分次数；事件丢失、scorer epoch 变化或评分过期需要重新积累。最近高分独立保留到其过期，避免随后低分立刻推翻重要证据。

默认每 25 ms 唤醒，扫描最多 256 个目录项/退休项；完成一轮增量扫描后才统一排序，避免低 ID 抢先被删。每批最多退休 64 块。候选需要观察 2 个请求 steps，使用/保护/等级发生变化会取消或重新观察。step 在 `generate_plain` 请求边界增加，**不是 token 数，也不是墙钟 TTL**；空闲不会凭空生成低分证据或推进观察期。可释放的退休资源在读者退出后仍由后台处理。

评分排序、目录发布会随历史大小增长，当前没有严格的每 tick CPU 时间上限；`scan_blocks/batch_blocks/poll_ms` 等可通过 `KvMemGcConfig` 调整。默认参数是初始值，尚未按任务质量调优。

## 生命周期与 GPU 边界

```text
已有 CPU 元数据、Host 评分
        ↓
后台扫描 → 规则排序 → 候选观察 → 发布不可变退休目录
                                      ↓
请求持有旧目录：继续原选择；新目录：跳过空洞
                                      ↓
旧目录读者退出 → 再次检查 lease / generation / 新评分
                                      ↓
后台 try-lock 回收 K/V 的准确 allocation generation
```

一次请求持有一个目录视图，避免 GC 在一次评分/选择中途改变候选集。请求结束（含异常）释放视图；当前 GPU 工作集和 checkpoint 保留各自的 lease，不依赖请求一直持有整个目录。异步 NVMe 读写批次单独保留视图至 reaping。旧读者退出前可以转交工作集引用，后台在读者退出后再次复核，取消被重新选中的候选。

回滚/截断后重新使用逻辑 block ID 会增加 block generation；NVMe 槽位还有独立 allocation generation，旧释放回调不能误删复用后的槽位。后台不访问 executor 的可变 block vector、raw-K 数组或 DeviceBackend，只消费固定大小的原子元数据和不可变事件。

GPU 上仍然执行原来的检索/推理计算。GC 不发起 CUDA kernel、D2H/H2D、同步或设备内存释放。CPU index 的既有 metadata packing 把退休块的 `block_counts` 设为 0，通过原有传输和 one-pass reducer，在 softmax 分母之前排除空洞；不会为 GC 增加一次评分。旧 two-pass 的分母不识别 counts，因此 GC 禁止该回退；one-pass 不可用时采用已有窗口热度，且不给 GC 伪造 Mean-K 观测。

前台不等待 GC 完成，也不在 decode 循环里扫描目录/排序/释放槽位。前台仍有必需的原子标志、视图/lease 交接和 CPU 评分事件打包。后台默认降低线程优先级，可绑定 CPU；NVMe GC 元数据操作使用 try-lock，LRU 改为 O(1) 摘除。**这些约束不能证明端到端 latency 增量严格为零**：共享 CPU、内存带宽和短暂元数据锁仍可能有开销。

## 回收结果与容量边界

回收的是在线 raw-K 与 V NVMe allocator 的可复用槽位。被删块不参与 top-k、压力选择或 all-fit 快路径；显式请求已删块会报错。普通 prefix reuse 仅检查待复用前缀内的 mandatory 区间；若其中有已删块，记录 `action=checkpoint-miss reason=required-history-deleted` 并尝试较短 checkpoint，均不满足时才从完整 transcript 走现有 cold prefill。原本就要重建的后缀不会因此触发整个前缀重算。这用于正确性恢复，不承诺局部重建，也不是一般语义 miss 的自动恢复。

`--kvmem-gc-max-blocks` 是软 live 目标：年轻块、工作集、checkpoint、CPU 缓存和异步读者可能让实际占用超过目标，不为达标破坏保护条件。无空间时沿用已有 admission/背压逻辑；前台不会等待一次 GC 扫描来满足分配。

本版本不 punch hole、不截短预分配文件，也不立即清零旧字节。CPU raw-K chunk 内的旧字节随原有有界缓存淘汰而释放，读取/写回混合 chunk 时跳过已删槽位。索引行和稳定逻辑 ID 保留 tombstone；metadata、位置和 NVMe 预留容量仍受 `--ctx` 约束。因此可复用槽位增加，不等于磁盘文件大小、RSS 或 GPU 显存立即下降，也不能声称支持无限会话。

## 启用与观察

在已验证可运行的 plain KVMem 服务配置上使用：

```text
--kvmem-prefix-cache
--kvmem-query-conditioned
--kvmem-immutable-source-k
--kvmem-raw-k-nvme
--kvmem-index-placement cpu
--kvmem-retrieval-method mean-k
--kvmem-select-policy topk
--kvmem-opt-stage-out on
--mtp-chain 0
--kvmem-gc-max-blocks N
```

继续配置现有 `--kvmem`、`--ctx`、工作/生成预算和 CPU/NVMe 容量。`N` 必须大于 `ceil((max(select_budget,prefill_budget)+gen_budget)/block_tokens)`，且小于 `ceil(ctx/block_tokens)`；启动时检查。`N` 是物理 block 数，不是 token 数。

增加 `--kvmem-gc-shadow` 可只观察不删除；移除该参数才实际回收。可选 `--kvmem-gc-cpu C` 绑定到进程 affinity mask 内预留的 CPU。绑定不在允许集合内会报错，不会悄悄忽略。GC 默认关闭，不改变现有服务启动参数。

`QW3_KVMEM_GC_TRACE=1` 输出 block ID、generation、step、level、score、reason bits，以及 `retire/reclaim/shadow/stop` 事件。`KvMemGc::stats()` / `QwenExecutor::kvmem_gc_stats()` 提供扫描、候选、取消、退休、已回收、读者延期、评分事件间隙、失败及 CPU tick 耗时。worker 发生异常会明确记录并停止回收。

Shadow 对同一候选只提名一次，按模拟 live 目标停止；后续实际使用会增加 `shadow_reuses` 并取消该提名。这是原选择轨迹上的反事实复用信号，不能当作真正删除后成功率或精确 re-prefill 概率。

## 验证记录

- CPU Debug 构建成功；最终完整主机端回归 **15/15 通过**，包含 GC、store、NVMe、prefix reuse、request-plan、archive 等。最初的本地 socket 测试在沙箱里因不能 bind 失败，在允许 loopback 的环境单独及整套复验通过。
- GC 主机测试覆盖：缺评分不等于低分、同分/事件间隙、低分优先级与增量扫描顺序、真实使用/新高分取消候选、保护区、shadow 复用、旧视图与 I/O 延期、实际 K/V 槽位回收、all-fit/pressure 空洞、checkpoint 引用、截断重用与 allocator ABA。并发测试连续 50 轮通过。
- ASan/UBSan/LeakSanitizer 主机测试通过。ThreadSanitizer 无法在当前环境启动（`unexpected memory mapping`），未宣称其验证通过。
- CUDA 13 / sm_120 独立构建通过。8-block GPU 掩码单测通过，覆盖 scalar 和 GEMM reducer、包含全空 tile 的任意空洞、全空候选集，并对照排除空洞后的 CPU softmax；被删块采用极大 logit，防止“只在 softmax 后置零”的错误实现蒙混过关。
- 没有运行现有模型服务的完整任务质量/TTFT/TBT 压测，未实测删除后的 re-prefill 降幅或零延迟增量。当前共享 GPU 已有进程占用约 60 GiB 显存，只运行了小规模掩码正确性单测。

独立环境下一步对比：GC off / shadow / delete，同模型、同容量、同轨迹，记录任务成功率、后续复用、cold-prefill 原因、TTFT/TBT p50/p95/p99、CPU tick 时间与可复用槽位。内容关系需先由可靠工具包装层提供整块来源/范围/版本/验证证据，再接入替代者保护和有预算的 rollback anchor。
