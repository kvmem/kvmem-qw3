# CacheBlend memory benchmarks：历史协议对齐

用户确认保留本次统一 36,864 KV 预算，仅对齐历史样本、提示词及评分。

- 引擎与模型：kvmem-llama-cacheblend，Qwen3.8-27B GSQ-RCO IQ3_S、MTP3、Q8 KV。
- 保留当前 10% 重算、alpha1、GDN reset-restore、sink1312/recent10496、生成预留32768、原采样和输出参数。
- LongMemEval-S：原始 selected_12_samples.jsonl 的 102 题、六类各17，完整记录及顺序固定；历史 matched102 system/history/question 文本逐题校验。
- MemoryAgentBench：原 >256K 30 contexts / 1316 questions。
- AgentLongBench：原 long250、normal100、DeepseekMillion50。
- 总计1818问。其他测试的156个输入/清单文件与历史源 SHA256 一致。

## 评分

LongMemEval 复用历史 DeepSeek Flash 客户端及官方题型提示，非思考、温度0、16token、严格正常结束的 yes/no，执行历史12例校准。API错误/非二值/截断保持待评分；同答案哈希已成功评分可复用。MemoryAgentBench 保留原本地官方指标与 DeepSeek V4 Pro 特殊题评分；总体采用类别宏平均。AgentLongBench 沿用 canonical evaluator。

原持有凭据的 grading_queue 进程保留，在真正 inference_complete 时加载更新后的 score.py；未提取或保存凭据。此进程的旧状态字段可能仍显示 Pro，实际 LongMemEval 请求和报告明确记录 Flash。

## 结果版本与恢复

旧 S500 的39个已完成答案全部移至 revisions/matched102_20260928/s500_longmemeval_cases，保留完整请求、响应及日志，不计入新102题。并非按正确性选择重跑。4个输入/运行完全一致的非LME结果通过哈希复用。

原 identity、控制器、评分脚本及 source diff 都保存于 revisions/matched102_20260928。新 runner 正常生成 v2 identity，校验原模型、二进制、共享库及全部服务参数未变；没有伪造旧哈希。

代码：/home/chaidi/kvmem-freetoken/benchmarks/cacheblend-memory-pro6000-20260928/
结果：/data/chaidi/kvmem_eval/results/cacheblend_memory_pro6000_20260928/
新启动日志：launcher-matched102.log。

`status.json` 为实时进度，`summary.json` 为推理计数，`report.json` 为独立评分。语义复核旧36题的100%属于已归档旧S500快照，不代表本组成绩。
