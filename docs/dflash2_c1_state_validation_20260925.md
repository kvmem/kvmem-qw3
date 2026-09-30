# 单请求 KVMem：当前候选状态重验与评审校准

本轮未修改运行时算法，沿用阶段 7 的 build14 CLI 和 Q8_0＋E4M3 FP8 KV。
SwiGLU reuse、FFN down streaming/int32、fanout load mode=2、七草稿用于状态探针；
continuous batching 关闭。重新链接了生命周期探针，并重新构建/链接摘要压缩探针，
构建记录为 `q8_c1_lifecycle_build15.json`，没有用旧探针替代当前构建证据。
CLI SHA-256：`0286b569927ab2f124f89f0e16d4c6f9e43e003891d6fa4334e6e7b92c8e0c98`。

## Guided 与生成中途刷新

4K guided suite 的 **18 个路线步骤**通过。比较无投机和 DFlash 的历史摄入、
首次/热查询、direct/private guided 查询、attention probe、两种中途刷新，
以及返回主分支后的回答。除最终响应文本及回调数量外，runner 另比较完整生成查询、
thinking/query token 计数、刷新位置、回退/生成策略及窗口长度，两条路线一致。

这里直接比较的是响应文本和回调数量；探针没有导出完整输出 token ID，
不扩大解释为冻结性能矩阵要求的逐 token ID 对照。该项仍需独立完成。

## 摘要压缩与回滚

摘要探针通过 **63、511、1535、4095** 四个替换边界及四类查询状态检查。
覆盖私有摘要取消、嵌套拒绝回滚、已发布 thinking 尾部替换、非法来源拒绝、
合法查询保留、warm query stash，以及普通截断时的查询失效。

源码使用字节数组比较循环状态、卷积状态、hidden、草稿位置及 K/V，
并将替换后的状态、logits、替换块 raw K 与 FP8 K/V 对照干净的 prefix+replacement
执行结果。查询捕获数据及准备状态另有检查。这里的 target KV 字节检查针对替换块，
不冒充对所有历史存储页的全量独立审计。

## 32K 有界保留

当前候选的 **16 个路线步骤**通过，DFlash 与同策略无投机响应文本一致。
逻辑历史 32768 token，固定 256 个 slot、每块 32 token，保留上限 8192，
工作窗口 1024、生成预算 512、CPU tier 4 GiB。覆盖历史增长、尾部改写、
取消后恢复和旧前缀分叉。这是有界保留场景，没有把全部 32K KV 常驻 GPU。

以下为正确性运行的请求墙钟诊断，不是正式性能测量。涉及完整前缀重放的场景仍很慢，
不能用热查询吞吐替代这些开销：

| 路线 | 场景 | 秒 |
| --- | --- | ---: |
| plain | cancel-after-5 | 24.483 |
| plain | after-cancel | 25.812 |
| plain | old-prefix-fork | 25.813 |
| dflash | cancel-after-5 | 24.727 |
| dflash | after-cancel | 24.987 |
| dflash | old-prefix-fork | 25.107 |

## 冻结本地评审器校准

新增 `run_kvmem_judge_calibration.py`，使用冻结的官方逐类型评分提示函数，
不改动 `kvmem_matrix_v2.json` 或 14 条冻结正负用例。目标 GGUF 的完整 SHA-256
实际重新计算并匹配。评审配置为本地 Q8_0、FP8 KV、无投机、无 KVMem、greedy、
thinking 关闭、最多 16 token。仅用 Transformers 渲染模板，再用序列化
`tokenizer.json` 管线编码，向原生 completions 发送整数 token ID。

按 prompt SHA-256 固定排序，**14/14 校准通过**，涵盖六类问题与拒答。
保存全部 prompt、输入 IDs、请求、原始回答和 verdict；仅接受去除首尾空白后的
yes/no（大小写不敏感，可有一个句号或感叹号），且必须正常停止，非法回答不能算通过。

这只使该固定评审配置具备执行后续评分的校准资格，没有生成 LongMemEval 任务成绩，
没有证明质量非劣，也不是官方榜单评分。尚未开展多并发。

后续继续当前候选的 128K 重验、逐 token 对照、冻结 C1 性能/质量任务与独立重复。
同机公开参考资格仍未解决，正式质量和性能门槛保持未完成，目标保持 active。
原始证据、关键源码快照和哈希索引见 `benchmark/dflash2/q8_c1_optimization_stage08.json`。
