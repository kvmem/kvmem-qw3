# 持续会话补位优化

2026-09-27，保持Q8_0目标权重、FP8 KV和每请求K128K/G64K。

已有controller在任何pending事件存在时都不接纳新请求。独立prefill的事件可能跨
多个调度轮次保留，因此一个请求完成腾出执行槽后，新请求仍可能等其他请求完成
长prefill才能进入。这降低了持续工具会话的执行槽利用率。既有测试按整组请求完成
后再提交下一组，无法充分覆盖这种到达模式。

新增默认关闭的`QW3_CONTINUOUS_KVMEM_PREFILL_ADMISSION=1`，要求同时开启
cooperative prefill。只剩request-local prefill事件悬挂时，允许在本轮为有限数量
的空槽接纳请求；若存在尚未服务的验证、草稿或query事件，仍先完成已有事件。
不改prefill和验证的数值路径，不允许新请求占用其他请求的执行槽。
日志`continuous_prefill_admission`记录实际发生补位时的held-prefill数量。

长历史探针增加`request_streams`，与`request_groups`互斥。各stream内部顺序提交，
仅首次到达同步，每个会话完成后立即提交自己的下一轮，不等待其他会话。
校验每个请求恰好执行一次，同一个session key只能由一个stream写入。
保留完整输出IDs/字节、每请求时间和整个workload的共同墙钟，不能用重叠请求耗时
之和计算聚合吞吐。仍然是历史回放，不是新的SWE任务成功率测量。

`concurrency_replenish_build01`已完成冻结CUDA链接构建，只重编译backend和两个
受影响探针，CUDA算术保持不变。`concurrency_replenish01`队列：

1. C4短请求补位/取消一致性检查：长短prompt混合、8个排队请求，开关前后对照，
   必须实际出现held-prefill补位；短fixture的K/G仅用于功能检查。
2. 四个真实长历史会话各两轮，共8请求，各自独立续发；相同二进制、相同配置及
   采样种子，仅切换补位开关，MMA和其他未验收数值候选关闭。
3. 比较完整输出、总完成墙钟、服务吞吐、TTFT、最大输出停顿和验证批宽分布。
   首轮若无收益不扩展效率矩阵；若有收益再作必要独立复测。

短GPU检查已通过32条响应（含取消），输出和callback pieces在开关前后完全一致，
日志确认8次held-prefill补位。长历史队列已进入`reference`，完成后自动测试candidate，
尚无本次修改的长历史吞吐结果。冻结探针的`cooperative_prefill`汇总字段遗漏了新增
parity开关，误显示0；实际设置为1，有逐次prefill/补位日志为证。工作区已修正这一
显示字段，未改动运行中的冻结二进制和采样路径。

bounded的服务入口、
跨请求资源额度和HTTP生命周期仍待完成，此次补位修改不代表已开放bounded并发。

## 长历史结果：正确性通过，无整体收益

`concurrency_replenish01`已完成。四个独立stream、8请求，两侧完整输出token IDs和
字节全部一致，共6203个输出token，两侧均有4次真正的历史复用。

| 指标 | 参考 | 提前补位 |
| --- | ---: | ---: |
| 全部请求共同墙钟 | 339.8998 s | 340.5599 s |
| 含冷prefill的服务吞吐 | 18.2495 tok/s | 18.2141 tok/s |
| 最大输出间隔 | 3.5837 s | 3.5881 s |
| held-prefill提前补位 | 0 | 1 |
| prefill计算累计墙钟 | 233.7980 s | 233.9151 s |
| 验证计算累计墙钟 | 62.3261 s | 62.5851 s |

吞吐比0.9981，单次筛选没有收益，不扩展重复效率矩阵。短fixture证明该准入限制
真实存在且修正可以工作；真实长任务本轮只触发一次，不能把它当成主要加速来源。
约69%的共同墙钟仍在prefill，验证成本仍随批宽近似线性增长。此服务吞吐包含首次
131K–197K长prefill，不能与暖组51 tok/s或单请求纯decode直接作比例对比。
结果保留在`benchmark/dflash2/concurrency_replenish01/comparison.json`及原始日志。
