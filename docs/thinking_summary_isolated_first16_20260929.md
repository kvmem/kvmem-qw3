# DeepSWE 原 16 样本：当前 thinking 独立摘要

2026-09-29。用户选择每个样本一次，seed=1000，共 16 个 trial，仅摘要组。
正式执行于 15:48 CST 开始，第一个任务为 `superjson-error-stack-serialization`。

22:09 CST 按用户要求暂停，保留 3 个有效完成结果及第 4 个任务的未完成轨迹。
用户随后要求全部 16 个样本关闭摘要后重跑，见
[KVMem + DFlash 无摘要批次](kvmem_dflash_no_summary_first16_20260929.md)。
监控端口 8322 已切换到新批次，下文保留原摘要批次的配置与启动记录。

## 批次

实际目录：`/data/chaidi/qw3-experiments/thinking_summary_isolated_first16_seed1000_20260929`。
工作区入口：`benchmark/thinking_summary_isolated_first16_seed1000_20260929`。

任务及顺序来自 `benchmark/claude_deepswe_ab/tasks_first16_full113.txt`，与之前
`thinking_summary_first16_pass4_promptfix_20260923` 的 16 个任务一致。全部任务说明
哈希、32 个 agent/verifier 镜像 ID、Claude Code 2.1.148 二进制及 agent/verifier
脚本已核对。旧批次保持暂停，本批次不混入其结果，不重跑 baseline。

- 模型：Qwen3.8-27B Q8_0，FP8 KV，BF16 DFlash 2。
- 摘要：只输入本轮 thinking，独立 executor；阈值 1024 tokens，自然结束，无固定
  摘要 token 上限；无效摘要保留原文。
- KVMem：step，CPU Mean-K index，K128K / G64K，512K 物理仓库（16384 × 32），
  48 GiB CPU backing，delete GC，guided query cap 512。
- Agent 逻辑上下文：3M；客户端不自动 compact；沿用原工具和任务执行约束。
- 采样：temperature=1.0、top_p=0.95、top_k=20、thinking budget=28672、
  presence penalty=0、repetition penalty=1、seed=1000。
- 每个 trial 使用独立服务及容器，按原任务顺序串行执行。与已有 vLLM 共享 GPU，
  每个 trial 启动前要求至少 44 GiB 空闲；耗时包含资源竞争。

本次投机方案从旧批次的原生 MTP 改为 DFlash，二进制也含后续改动。历史数据仅作
参照，不能把得分或耗时差全部归因于独立摘要，更不能把单 seed 结果当作 pass@4。

## 启动前检查

[27B/DFlash 功能回归](thinking_summary_27b_dflash_validation_20260929.md) 已通过
状态隔离、四个边界的回滚、并发及默认 1024 阈值 HTTP。

本批次又以正式 K/G/仓库/CPU 预算启动服务。首个预检请求 thinking 为 401 tokens，
按默认阈值正常跳过。第二次使用请求级测试阈值 64，实际压缩 401 → 58 tokens；
正式服务默认仍为 1024。预检请求还将 thinking budget 临时设为 1600，正式 agent
保持 28672。原请求、响应和日志在 `preflight/`、`preflight-02/`。

预检退出时，旧 harness 的 60 秒端口释放检查因 TIME_WAIT 报错；当时服务进程及
监听器均已退出。保留原失败记录，将本批次冻结 harness 的端口等待延长为 90 秒，
并独立复核进程退出和端口可绑定。没有修改运行时或评分逻辑。审计见
`operations/preflight-audit.json`。

Campaign 调度器支持显式 seed 列表，进度及完成判断改用 manifest 的 16 次规模。
8 项调度/指标测试通过，另以本批次 manifest 核验 `0/16` 初始报告。冻结文件在每个
trial 前校验；目标及 DFlash 权重的身份也记录在 manifest 中。

## 运行与结果

独立 systemd 用户服务：`qw3-thinking-isolated-first16-s1000-20260929.service`。
它在当前对话结束后仍继续执行。旧巡检 timer 没有重新启用或改指向。

- `state.json` / `live.json`：当前样本、请求数和最近 KV/context；运行中约每 10 秒更新。
- `report.md` / `summary.json`：已完成数量及任务通过数，随每个 trial 完成更新。
- `results.csv` / `samples.csv`：verifier、耗时、context、KV 和摘要成本。
- 每个 attempt 的 `requests.jsonl`、`resources.jsonl`：请求指标及 GPU/主机资源采样。
- 原始 agent、服务、verifier 日志及 patch 保留在各 attempt 的 `runs/`。

任务 reward=0 作为有效完成结果保留，不为筛选成功而重试；推理或基础设施错误保存
attempt 并进入 `needs_attention`。当前报告只代表已经完成的任务，启动成功不代表
16 个任务已经测完。

只读状态命令：

```sh
systemctl --user status qw3-thinking-isolated-first16-s1000-20260929.service
cat benchmark/thinking_summary_isolated_first16_seed1000_20260929/summary.json
cat benchmark/thinking_summary_isolated_first16_seed1000_20260929/live.json
```

## 实时监控界面

本机访问 <http://127.0.0.1:8322>。监控由独立用户服务
`qw3-thinking-dashboard-20260929.service` 运行，每 2 秒读取现有服务和客户端日志；
不写入冻结配置，不向推理服务发送请求。关闭浏览器不会影响测试或监控服务。

界面包含 16 个样本的完成/通过进度、当前请求时长、加权吞吐、逐请求曲线、
Agent 输出、thinking 摘要、工具结果及服务日志。默认跟随当前样本，可点击任务
查看其他样本。每个样本旁标注 context、耗时和通过情况；完成后显示报告中的
`final_context_tokens`（最后一轮 prompt 加摘要后保留输出）和 `total_elapsed_sec`
（Agent 加验证总耗时）。运行中标为“当前 CTX / 已运行”，不冒充最终数据；
选择样本后还可看到 F2P/P2P 的通过数。客户端输出缓存尚未释放时，明确显示“等待缓冲输出”；当前冻结
服务没有逐 token 的旁路监控，因此不估算或伪造瞬时速度。内容一旦写入日志，
下一次轮询即可显示。摘要页保留服务实际生成的摘要文字。

吞吐使用原始目标模型输出量：`decoded + sum(original_tokens - retained_tokens)`，
仅对成功提交的 `compacted` 事件补回被删除的 token。DFlash 草稿候选、摘要子任务
token 不计入主请求输出。解码卡片扣除摘要、回滚及回放耗时；完整请求卡片包含
预填充和上述全部开销，不含 agent 工具等待。两者均为已完成请求的总 token 数除以
总时间；进行中的请求不纳入均值。图表显示所选样本，顶部卡片显示全批次。

源码：`scripts/kvmem_eval/thinking_dashboard.py`、同名 `.html`。
不依赖外部前端资源或 Python 包，单独启动命令：

```sh
python3 scripts/kvmem_eval/thinking_dashboard.py \
  --experiment /data/chaidi/qw3-experiments/thinking_summary_isolated_first16_seed1000_20260929 \
  --port 8322
```

测试：`python3 -m unittest discover -s scripts/kvmem_eval -p test_thinking_dashboard.py -v`。
浏览器检查与截图保存在实验目录的 `dashboard/`，覆盖自动刷新、任务切换、
三个输出页签、窄屏和断线重连。
