# Thinking 摘要输出修复与重新启动

用户目标：修复摘要分支继续调用工具的问题，验证后启动 16 样本 × 4 轮摘要版测试；**测试实际启动且每 30 分钟巡检生效即完成本次目标**，后续由定时巡检观察实验，不再以等待全部 64 次结束作为本次目标完成条件。只使用已有 baseline 作参考，不启动 baseline DeepSWE。

## 修复

- 提示词明确编码任务暂停，仅写短记忆；区分已完成事实与待办，不执行下一步、不虚构测试结果。
- 空 thinking 后预填 `Summary: `，与 guided query 的正文前缀方式一致；该固定前缀不被重放回主历史。
- 只在私有摘要分支约束工具调用、thinking、角色标记和代码围栏。检查解码字节，包括标记跨多个普通 token 的情况；保留普通文件名、标识符和不等式。GPU argmax 合法时沿用原快速路径，非法时读取 logits 选分数最高的合法 token。
- 摘要自然终止；保留 thinking ≥1024 的阈值、无固定词数/token 摘要目标、资源上限和“必须变短”的条件。
- 无效或被截断摘要保留原 thinking；不通过删掉工具 XML 或截取半句话强行接受。
- `[thinking-summary-output]` JSON 保存原始摘要、结束原因、屏蔽次数和前 32 个被屏蔽候选 ID。指标解析器将这些内容当数据，避免文本中引用日志导致误报。

## 验证结果

验证目录：`/data/chaidi/qw3-experiments/thinking_summary_output_validation_20260922`。

- CPU 格式约束测试覆盖整 token / 跨 token 标记、普通代码标识符、UTF-8、不等式、自然终止、未闭合输出、最高分合法候选。6 项相关 CTest 通过；其中 request-plan 可执行文件首次未构建，补建后单独通过。5 项 campaign 指标/顺序测试通过。
- 4B 状态 oracle 在 63、511、1535 边界验证 recurrent、conv、hidden、主/MTP KV、logits，回滚与直接计算逐字节一致。4B HTTP baseline/summary 验证通过，覆盖 streaming、下一请求、短 thinking、无效摘要、取消恢复、工具路由。这里只运行轻量 HTTP 回归，不是 baseline DeepSWE。
- 27B 用三段历史失败 thinking 重建任务+推理+最小工具上下文：旧提示词 3/3 输出实际工具调用；新提示词+前缀 3/3 自然结束、无工具标记，摘要正文分别为 **843 / 386 / 822 tokens**。对应历史 thinking 为 **24880 / 19930 / 2936 tokens**。这些单独回放未开启格式约束，说明提示词修改本身解决了这三个实例。
- 历史回放不是原始完整 Claude Code 系统提示词和 KVMem 状态的逐 token 重演，不能把它当作新的 DeepSWE 耗时或正确率结果。
- 27B 真实生产路径：**2073→343 retained tokens**（摘要正文 341 tokens），保留 `SENTINEL_R7` / 公共 API 不变 / 尚未运行测试 / 待办 Read；随后正确调用 `Read(src/cache.py)`，下一轮正常回答 API 约束及未执行测试的状态。该次摘要耗时约11.31秒，回滚约8.56ms，重放约234.75ms；不是任务级性能结论。
- 27B 首个验证环境 CPU backing 8GiB 不满足 inclusive backing，改为与正式测试相同的 48GiB；该启动失败保留在 `output-27b-insufficient-backing*`。第一版事务测试把长分析放入可见正文，thinking 只有40 tokens，正确走短段跳过；调整测试输入要求仅在私有推理中比较后，完整事务测试通过。原证据保留在 `output-27b`，成功事务位于 `output-27b-transaction`。
- 有限样例只证明当前输出修复及流程通过，不能保证所有摘要语义无损。保留任务 verifier 和失败保护。
- 验证期间另一任务启动了 OSWorld GPU 服务，随后在正式批次启动检查中发现。上述验证耗时不满足独占 GPU 条件，只作诊断记录，不混入正式结果。用户随后明确授权使用剩余显存并行测试；正式批次按共享 GPU 运行，耗时包含竞争影响。

## 新版完整实验

正式目录：`/data/chaidi/qw3-experiments/thinking_summary_first16_pass4_outputfix_20260922`，工作区入口同名位于 `benchmark/`。

按轮次执行完整16样本，seed=1000/2000/3000/4000，只跑摘要共64个trial。阈值1024，自然终止；3M逻辑上下文，512K物理仓库，128K selection、64K generation reserve，48GiB CPU backing。每个 trial 使用独立服务与容器。

2026-09-22 23:30：用户授权并行后移除独占 GPU 和其他实验进程退出门槛；每次服务启动检查至少40GiB空闲显存，当前约57GiB空闲时已进入第一个样本启动。修改记录在 `harness-revisions/002-user-authorized-shared-gpu`，模型二进制和摘要策略不变。`manifest.json` 的 `allow_shared_gpu=true`、逐 trial 的 `shared_gpu_allowed` 和报告注明共享运行；`resources.jsonl` 每10秒保留所有 GPU 进程 PID、显存分配及本实验的资源曲线。不能把此条件下的时间直接解释为独占 GPU 性能。

冻结模型二进制、源码、harness、任务、镜像和验证结果摘要哈希。每请求记录 KV live/ready、CPU/GPU/index bytes、context、摘要/回滚/重放耗时及屏蔽次数；每10秒采集资源。最终 context/KV 在服务销毁前采样。任务 reward=0 是有效完成结果，不通过重试筛选通过样本。

旧目录 `thinking_summary_first16_pass4_20260922` 停止并保留，不混入新版结果。取消时已确认旧 supervisor/worker/service/container/listener 均退出；旧 Docker 网桥地址消失造成清理端口探测误报，已把服务退出检查改为主机通配地址以验证实际监听器释放。

## 每30分钟巡检

用户 systemd `qw3-thinking-first16-monitor.timer` 调用 `scripts/kvmem_eval/check_thinking_campaign.py`。检查进度、进程身份、心跳、服务错误、日志停滞、磁盘、KV/context和摘要状态；至少3次尝试但零成功压缩会触发诊断通知。正常检查写 `monitor/checks.jsonl` / `status.json`，保持安静；异常、重要变化或完成时通过已验证的 `codex queue --thread` 通道通知当前任务并按授权修复。64次有效结果完成后汇总并关闭巡检。

该机制为本机 systemd 定时检查；当前会话未提供原生 automation_update 工具。
