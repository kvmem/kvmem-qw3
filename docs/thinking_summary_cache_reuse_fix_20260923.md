# Thinking 摘要后的缓存复用修复

## 原因与实现

旧版成功压缩会通过 `restore_state()` 调用通用 `kvmem_truncate_to()`。该路径同时清空 query span、Q 的计数及 ready 标志，导致请求结束时无法保存 warm query snapshot。下一请求只能退到 query 之前的 checkpoint；没有合适 checkpoint 时完全 miss。旧正式批次前两个样本的 57 次摘要后续请求全部 miss，而其他 288 次相邻请求均命中。

新增 `restore_state_preserving_query()`，仅供本请求内的摘要替换使用：

- 检查 query 来源范围不超过保留的 thinking 起点，私有分支已取消，checkpoint 和 selection generation 仍有效。
- 保留原 Q 的字节、布局、span、计数及 ready 状态；包括行数不同于公开问题的 guided Q。不覆盖 clean/warm stash。
- 继续按原规则截断 KV、原 thinking、仓库尾部和旧索引；从保留的仓库 handle 重新发布内容索引，重建必要的非整块尾部。
- 普通历史分叉和取消仍走通用失效路径。前缀 LCP、query 来源和 selection generation 的安全条件不变。

## 验证

证据目录：`/data/chaidi/qw3-experiments/thinking_summary_cache_validation_20260923`。

| 检查 | 结果 |
| --- | --- |
| 状态 oracle，边界 63 / 511 / 1535 | recurrent、conv、hidden、主/MTP KV、logits 与直接计算逐字节一致 |
| Q oracle | GPU / host Q、7 / 48 行 guided Q（公开 span 32 行）均保持字节及 readiness；非法来源回滚被拒绝，通用 truncate 仍使 Q 失效 |
| 同输入 4B 旧版 | 成功摘要后只复用 256 tokens，prefill 361，query_snapshot=0，回归断言失败 |
| 同输入 4B 修复版 | 复用 582 tokens，prefill 35，query_snapshot=1；后续两次工具请求也命中 |
| 4B guided query + 摘要 | 私有分支产生 53 行 guided Q；149→119 tokens；后续复用 6482 / 6591 / 6683 tokens，每次 prefill 35 |
| 27B 正式阈值 1024 | 2073→343 tokens；下一工具请求复用全部 797 tokens，prefill 43；Read 路由、API 不变、未运行测试等状态检查通过 |
| 4B HTTP 完整回归 | 摘要启用/关闭均通过；涵盖短段跳过、streaming、无效摘要、取消和工具路由 |
| CPU 检查 | 格式约束、前缀复用策略 CTest 通过；7 项 campaign 指标/调度测试通过 |

4B 定向缓存测试使用 128-token 触发阈值以隔离缓存机制，**正式实验和 27B 验证仍为 1024**。4B 摘要关闭回归不是 baseline DeepSWE。27B 历史输出回放引用上一版已完成结果，本轮重新执行实际压缩和后续工具请求。共享 GPU 下的诊断耗时不作为任务级性能结论。

失败的测试准备过程全部保留：显存比例/选择预算不足；样例 thinking 未达到阈值；短摘要未自然结束；工具策略或显式 query span 改变导致合法的缓存 miss。最终测试保持工具定义、tool_choice、query span 一致，同时仍验证另一次改变提示词的请求正常执行。没有放松缓存安全检查来让这些用例通过。

## 实验与巡检

旧目录 `thinking_summary_first16_pass4_outputfix_20260922` 已停止：保留已完成的两个旧版本结果，第三个样本作为用户要求修复时中断的尝试。停止时冻结 harness 的端口释放检查超时；随后确认本实验的 supervisor、worker、服务均退出，18973 无监听，保留原异常和后续核验记录。

修复版使用独立目录 `thinking_summary_first16_pass4_cachefix_20260923`，重新按每轮完整 16 样本、seed 1000/2000/3000/4000 的顺序执行 64 个摘要 trial。baseline 仅引用既有结果。模型二进制、源码、验证结果和 harness 冻结并记录哈希，不合并旧版本结果。

保留用户已授权的共享 GPU 运行和每 30 分钟巡检。新增逐请求 prefix-cache HIT/MISS、query snapshot capture/reuse、prefilled tokens，以及摘要后下一请求的专门汇总；巡检遇到重复 query-boundary miss 会提示检查快照和提示词变化。时间仍含其他 GPU 工作负载竞争，任务质量由原 verifier 判定。
