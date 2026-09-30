# Thinking 摘要提示词调整与重跑

## 修改范围

用户要求仅先修改摘要提示词，再重新运行实验。模型源码相对于上一批
`thinking_summary_first16_pass4_cachefix_20260923` 的冻结包，只有
`src/thinking_compaction.hpp` 的提示词变化：只替换紧邻的 thinking，保留新增关键结论
及不可省略的约束/问题，用最短的单段文字表达；省略任务背景、架构回顾、历史进展、
完整文件清单、探索过程及实现展开。工具调用禁止、空 thinking 前缀和输出格式约束保留。

触发阈值仍为 1024 tokens，不设置固定摘要 token 上限；自然结束、必须缩短、无效摘要
保留原文的行为不变。未加入连续失败暂停机制，未实现后台删除 thinking。
DeltaNet 回滚、摘要回填和 query snapshot 复用代码不变。

## 验证

目录：`/data/chaidi/qw3-experiments/thinking_summary_prompt_validation_20260923`。
新 binary SHA256：`70926896358171c0b66057c342fd29a090d087f951b6fb0d58eba1ed5395e874`。

使用上一批 koota 样本第 55、67、127 次请求记录的 thinking，在相同的重建上下文下
对比新旧提示词，输出额度与原失败请求一致：

| 原请求 | 旧提示词 | 新提示词 |
| --- | --- | --- |
| 55 | 1208 tokens，长度截停 | 680 tokens，自然结束 |
| 67 | 1189 tokens，长度截停 | 741 tokens，自然结束 |
| 127 | 1053 tokens，长度截停 | 427 tokens，自然结束 |

这是任务说明、记录的 thinking 与最小工具系统提示的重建回放，不是原始完整 Claude
transcript 或原始 KVMem 状态的逐字节重放。3/3 诊断通过不能直接代表完整任务的失败率。
诊断共享 GPU，耗时不用于正式性能结论。原始请求、输出、额度和 case 映射全部保留。

其他检查均通过：

- 27B 真实摘要事务：2073→127 retained tokens（生成摘要 125 tokens）；保留必要约束，
  正确发出 Read；下一工具请求 reuse=581、prefilled=43、query_snapshot=1；后续 API
  约束和未运行测试的状态检查通过。
- 4B HTTP 摘要启用/关闭的协议回归：streaming、工具路由、短段跳过、取消、无效摘要
  保留原文且不重放、缓存续接。摘要关闭的 HTTP 检查不是 DeepSWE baseline 重跑。
- 状态 oracle：63/511/1535 边界 recurrent/conv/main KV/MTP/logits 一致；四种 query
  保存路径及非法恢复拒绝检查通过。
- 输出格式约束检查和 7 项 campaign 调度/指标测试通过。

## 实验

新目录：`/data/chaidi/qw3-experiments/thinking_summary_first16_pass4_promptfix_20260923`。
工作区 `benchmark/thinking_summary_first16_pass4_promptfix_20260923` 指向该目录。
2026-09-23 13:05:47 CST 启动 supervisor，端口 18977；13:06:11 开始第一轮
superjson 样本。13:07 已核实 8 次真实请求完成，首次巡检无异常。

仍为原 16 个样本 × 4 轮，仅摘要组，每轮完成全部 16 个样本再进入下一轮，seed 为
1000/2000/3000/4000。配置、冻结 harness 和历史 baseline 引用均与上一批核对一致；
旧结果不合并进新批次。逐请求 KV live/ready、CPU/GPU KV bytes、最终 context、耗时、
摘要成本与缓存复用继续由原 harness 记录。

旧版已停止，保留两个已完成结果及第三个被中断样本。旧 supervisor 在清理时出现
端口释放检查超时，后续独立核验 supervisor/worker/server 均退出且端口可绑定；原错误
和核验记录保留在旧目录 `prompt-only-restart-*.json` 中。

原 30 分钟 systemd 巡检继续使用，监控目录改为新版。正常状态保持安静，仅在异常、
重要进展或全部完成时通知。新目录的 `version-transition.json`、验证 manifest 和冻结
源码记录此次仅改提示词的范围与版本来源。
巡检已重新启动，首次定时检查预计 13:37 CST。
