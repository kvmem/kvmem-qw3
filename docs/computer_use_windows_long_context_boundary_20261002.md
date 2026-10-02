# Windows 原生 Codex / QW3 长历史边界诊断

两条真实模型诊断均在约 317K 输入处报 KVMem 选择预算错误，未到自动压缩点。不能据此宣称 1M computer-use 历史可用，也不能把失败归因于显卡容量不足。两条的连续录像、完整请求、原生 Codex 事件、模型用量、转发回执和独立审计均保留，已停止本次模型与转发进程，Windows 执行器仍暂停。

这是单独的共用平台压力诊断，不计入已完成的十个自定义能力场景，更不计入原十个 OSWorld 衍生场景正式结果。真实推理使用固定 QW3 + DFlash2 和 Windows Codex 0.159.2；每次 `observe` 读取真实 Windows 截图，并附约 60K token、明确标注为合成的中性历史记录。只有只读观察工具，没有桌面动作。要求观察 17 次、保留第一与最后的标记；自动压缩阈值保持 950K。重复填充不代表普通桌面负载或日常延迟。

## 实际结果

| 配置 | 最大成功请求 | 下一次输入 | 实际必须保留块 / 预算 | 终态 |
| --- | ---: | ---: | ---: | --- |
| 历史截图 pin-all | 253,792 token | 316,743 token | 1,509 / 1,024 | 第 6 次推理失败；5 次观察 |
| 历史截图 retrievable | 253,806 token | 316,760 token | 1,493 / 1,024 | 第 6 次推理失败；5 次观察 |

两条压缩完成次数均为零，最终标记答案均未交付。原生回合分别 337.246 / 340.720 秒，录像 356.543 / 358.350 秒；录像包含管理员启动和收尾。整卡采样峰值两条均约 44.39 GiB，没有观察到 GPU OOM。关闭 KVMem 的同类长历史诊断本轮未执行，不能从此表推断它的上界。

不同运行的观察 ID、标记及时间不同，此表不是完全相同请求的延迟重放对照。成功请求达到 253K 只证明该压力诊断的请求执行到此处，不能将它作为桌面任务成功、长期记忆正确或普遍上下文上限。自定义能力场景的最长有效输入约 174K，仍单独报告。

## 原因及反证

第一条失败时，初步怀疑历史截图强制保留会连带扩大文本。第二条反证显示，仅切换截图为可检索保留仍在同阶段发生同一错误，所以截图策略不是充分解释。

两条的请求计划均报告 509 个必须保留块、493 个新后续块，按 128 token 一块约为 65K / 63K，计划数低于 131,072 token 选择预算。实际最终选择却分别要求保留约 193K / 191K，再加 sink，超过该预算。两条都复用了 127,744 的 P 检查点，实际需重新处理的后续内容约 189K，而当前新观察只有约 63K。成功请求的首 token 等待也随这段后续内容增长。

**已确认：计划与实际强制保留的范围不一致，预算检查触发了正常错误退出。** 结合 `qwen_native_backend.cpp` 在最终重选前从 replay 边界设置后续保留范围的代码，工作假设是旧 P 检查点之后重新处理的历史被当作完整 live suffix 保留。还需要在相同输入上控制前缀复用/恢复路径来确认具体修复；本次未修改引擎，也未通过截断历史或放宽预算制造通过。

## 验收与后续

宿主机独立审计绑定请求规范化哈希、VM 转发回执、引擎清单、原生事件、实际用量及连续录像。通过要求包括：压缩请求本身实际超过 900K、用量与引擎 trace 一致、压缩事件完成、恢复工具后上下文减小、第一和最后标记正确。本轮两条均不满足，原失败结果不改。四项审计控制已验证：小用量不能因普通请求声称 950K 而通过、请求变化不能复用旧回执、引擎用量不一致不能通过、丢失标记不能通过。

下一步先将这两个失败作为隔离引擎回归输入，修复或明确恢复/保留的预算合同，再重新验证实际压缩与 KVMem off 退出行为。它属于冻结正式 V0 前的共同平台正确性问题，尚未作为 agent V1 的优化收益。原十场景正式运行仍为 0，006 的原简历仍缺失；十个自定义能力配对的成功结果不受这两条诊断影响。

- [两段连续录像](http://127.0.0.1:18089/long-context/index.html)
- [pin-all 原始结果](/home/chaidi/qw3/apps/computer_use/.runtime/native-long-context-boundary-on-01/long-context-on-01/result.json)
- [pin-all 独立审计](/home/chaidi/qw3/apps/computer_use/.runtime/native-long-context-boundary-on-01/long-context-on-01/independent-grade.json)
- [retrievable 原始结果](/home/chaidi/qw3/apps/computer_use/.runtime/native-long-context-retrievable-on-01/long-context-on-02/result.json)
- [retrievable 独立审计](/home/chaidi/qw3/apps/computer_use/.runtime/native-long-context-retrievable-on-01/long-context-on-02/independent-grade.json)
- [自定义能力场景结果与零通过原因](/home/chaidi/qw3/docs/computer_use_windows_capability_boundary_20261002.md)
