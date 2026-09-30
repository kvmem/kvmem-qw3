# 9B 摘要 thinking 结束标记与服务端解析核验

## 结论与证据边界

本次未发现 llama.cpp 漏识别有效 `</think>` 或客户端丢弃正文。
证据支持原两次异常属于模型没有完成 thinking -> content 通道切换便结束。
但原正式请求未保存原始 token ID，随机 seed 也未固定，因此不将该推断
写成对原两个生成序列的逐 token 证明。新的诊断响应不能替换正式失败结果。

## 原正式请求

异常发生在前两个样本的第二轮摘要，分别生成450/1057 tokens并返回stop。
content/raw_content均为空，reasoning分别1703/3541字符，内容呈摘要形式。
未达到32768输出或10000thinking上限。服务端对应任务1096、4774都没有
`reasoning-budget: deactivated (natural end)`，其前一轮正常请求任务0、2186
均有该记录。此标记来自独立的生成token匹配器，检测到thinking结束token才输出。

## 解析器离线核验

使用运行中服务 `/props` 的真实GGUF聊天模板，以及该服务实际链接的
`libllama-common.so.0.0.1`，未加载另一份模型、未修改引擎。
生成前缀为`<|im_start|>assistant\n<think>\n`，结束标记为`</think>`。
服务器 `/tokenize` 验证：`<think>`=248068，`</think>`=248069，
`<|im_end|>`=248046。

8个CPU解析案例覆盖正常闭合、立即闭合、缺少闭合、仅末尾闭合、两条归档
reasoning文本，以及两条新捕获token序列。逐字节增量（保持UTF-8边界）与
整段解析结果一致；结束标记跨流式分段时仍正确。两条归档reasoning文本作为
无闭合输入只是行为模拟，不能视为已恢复原始生成序列。

## 两次独立GPU诊断

复用原失败请求的历史片段与上一轮摘要，重建prompt SHA均与正式审计一致。
保持模型、MTP、采样、32768输出及thinking预算；诊断固定seed=0，改为
非流式并启用n_probs=1记录完整采样token ID。每题仅一次，不按结果重试。

| 样本前缀 | 生成tokens / 捕获token IDs | 首个闭合token（1起算） | 最后token | 正文字符 |
|---|---:|---:|---|---:|
| eae5f30d | 837 / 837 | 237 | `<|im_end|>` | 2285 |
| f4ecff3b | 645 / 645 | 1 | `<|im_end|>` | 2176 |

两次都正常闭合并返回正文。将采样token解码后再送入同一解析器，其reasoning/
content与在线API一致，离线流式与非流式也一致。这是同提示下新的诊断采样，
没有复现原两次空正文，不能宣称原随机轨迹得到确定复现。

## 机制说明与后续修复方向

`common/reasoning-budget.cpp`的预算采样器在未达到上限时不修改logits，
所以并不禁止模型先输出EOS；达到预算才强制输出结束thinking序列。
`finish_reason=stop`只表示终止token/停止串，不保证已产生正文。
若进一步修复，应单独验证thinking尚未闭合时的EOS约束；这会改变生成协议，
本次未擅自启用、关闭thinking或把整个reasoning字段当作正式摘要。

## 实验保护与产物

14:35:30在GPU空闲边界暂停reader，14:38:18恢复，同一个模型服务器继续运行。
两次诊断未写入正式摘要/答案/评分，未重算或覆盖已有模型失败。
该168秒诊断暂停应从后续基于reader外部墙钟的效率汇总中剔除。

完整产物：
`benchmark/agentlongbench_9b_matched27b_20260916/diagnostics/thinking_end_20260920/`。
包含原日志证据、实际模板、源文件和共享库哈希、重建prompt哈希、采样token ID、
原始解码文本、解析器小程序与8例核验、reader暂停/恢复记录。
