# 长历史 HTTP C2 会话复用检查

2026-09-27，`http_session_concurrency01`。结论：多会话复用入口已执行，
但真实增长历史的输出与冷启动不一致，不能认定吞吐优化有效。

固定 Q8_0 目标权重、FP8 KV、每请求 K131072/G65536；两个真实 DeepSWE
历史各取相邻工具轮次（Quill r0063/r0064、Effect r0062/r0063）。同一测试
服务器、相同采样参数和种子1000，分别使用轮换 key 和驻留 key；自然 EOS，
每请求输出上限256。这是历史回放，并非完整 SWE 任务成功率评测。

| 策略/轮次 | 双请求完成时间 | 输出 token | TTFT（秒） |
|---|---:|---:|---|
| 轮换 key / 首轮 | 106.398秒 | 512 | 97.973 / 98.737 |
| 轮换 key / 次轮 | 105.367秒 | 273 | 102.413 / 97.356 |
| 驻留 key / 首轮 | 107.205秒 | 512 | 98.746 / 99.517 |
| 驻留 key / 次轮 | 11.101秒 | 274 | 8.384 / 3.453 |

首轮两请求文本及 usage 完全一致。次轮 Quill 的冷/热输出分别133/129 token，
首个文本差异在字符133；Effect 分别140/145 token，首个差异在字符52。
因此表中的时间下降只证明重复 prefill 确实被绕过，不构成正确性通过后的加速比。
SSE 分块时间也不等于逐 token 时间，不报告为原生 decode 吞吐。

服务日志确认8次 continuous DFlash 派发、215次双请求验证和2次驻留会话命中。
实际前缀复用位置分别130336和131008，仅处理3731和1555 token 的新增后缀。
资源观察无错误、无外来 GPU 进程。失败与批处理路由未执行或缓存未命中无关。

待定位的假设：复用 checkpoint 位于 query/block 边界，恢复后的 prefill 分块
与冷请求不同，可能影响 recurrent state 数值；选择窗口恢复也需要检查。
目前尚无证据在数值分块、历史窗口恢复及跨会话状态问题之间作最终归因。
不能因短场景通过而忽略长历史分叉，也不能仅据文本差异断言会话串扰。

原始证据保存在 `benchmark/dflash2/http_session_concurrency01/`：
`manifest.json`、`replay/summary.json`、`replay/responses.jsonl`、`server.log`、
`resources.json` 以及带输入证据 SHA256 的 `failure_analysis.json`。
原运行状态保持 failed；修复后应使用新运行目录，不覆盖此次失败。

另一个最早请求优先调度候选关闭会话复用，与此故障独立，继续完成长C4筛选。

## 后续：固定执行路径的顺序/交错对照

`http_session_serial_reference01` 先用旧串行后端顺序运行 A0/A1/B0/B1，
仍然实际命中两次暖前缀。但4条文本均与C2对应结果不同，连首次冷请求也不同，
所以这条对照同时改变了 verifier 数值执行路径，不能单独定位缓存问题。
该失败原样保留。

`http_session_controller_reference01` 随后固定同一个二进制、continuous controller、
稳定FP8 ragged路径、页池容量与所有采样参数，只将提交次序从双请求交错改成
A0/A1/B0/B1顺序。4条响应的文本及usage全部与原C2驻留结果一致，总输出786 token。
服务日志确认4次continuous DFlash派发、2次驻留会话命中，以及实际单请求验证；
未出现双请求验证。两次warm复用仍为130336/131008，资源观察无错误或外来GPU干扰。

这一对照支持新增会话状态归属在这两段长历史中保持一致，不能把原冷/热文本分叉
直接归为并发串扰。仓库既有 `scripts/kvmem_prefix_cache_verify.py` 也明确区分了
冷全量prefill与warm增量DeltaNet计算的浮点顺序；但该说明本身不能替代本配置的
target数值/质量验证，原冷/热差异仍保留，不修改失败状态或宣布冷/热等价。

此次仅比较HTTP文本和usage，没有直接比对输出token ID。诊断开启额外日志且期间
有CPU构建，因此不把顺序/交错耗时差认定为性能提升。独立证据与SHA256保存在
`benchmark/dflash2/http_session_controller_reference01/reference_analysis.json`。
