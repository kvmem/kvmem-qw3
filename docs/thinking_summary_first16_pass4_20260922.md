# DeepSWE 16 样本 × 4 轮：仅 thinking 摘要组

目标保持 active，完成全部 64 个有效 trial 并汇总后才结束。任务 reward=0 是有效结果；服务或测试环境失败要保留 attempt、定位修复后重试，不筛选成功的任务结果。

用户确认本次不跑 baseline。顺序使用 `benchmark/claude_deepswe_ab/tasks_first16_full113.txt`：每轮跑完全部 16 个样本，再进入下一轮。四轮 seed 为 1000 / 2000 / 3000 / 4000，与历史测试一致。

实际目录：`/data/chaidi/qw3-experiments/thinking_summary_first16_pass4_20260922`，工作区入口 `benchmark/thinking_summary_first16_pass4_20260922` 是指向该目录的符号链接。数据盘约有 1.9 TB 可用，原始日志全部保存。

## 配置

- Qwen3.8-27B Q8_0，Claude Code 2.1.148；冻结 binary、源码、测试 harness、任务指令、镜像身份、模型和 Claude 二进制 hash。
- thinking ≥1024 tokens 才摘要，不设置固定摘要词数/token cap，自然结束。
- 主模型 temperature=1、top_p=0.95、top_k=20、thinking_budget=28672、MTP chain=4。
- 物理 `--ctx=524288`，repository=16384×32 tokens；selection=128K，generation reserve=64K，CPU backing=48 GiB，CPU Mean-K index，delete GC，guided query cap=512。
- 客户端逻辑 context 上限 3M，禁用客户端自动压缩。bounded 后端请求上限独立于物理 `--ctx`，历史可以超过物理池。
- 第一版调度器误把 3M 传为物理 `--ctx`，在服务启动校验时失败，尚未发生推理。已修复为两个独立参数。`harness-revisions/001-physical-context` 保存旧版和 manifest；模型 binary 未变。
- 旧 4 样本 pilot 已停止，未完成的 baseline 不计入本批次。

## 指标与结果

- `state.json`：当前轮次、样本、attempt、运行/需处理状态。
- `live.json`：每 10 秒更新的已完成请求数、最近一次 KV/context 快照和当前耗时。
- `results.json` / `results.csv`：每个已完成 trial 的 verifier、耗时、最终和采样峰值 KV、context、压缩次数与成本。
- `samples.csv` / `summary.json` / `report.md`：逐样本四轮统计及完成进度。
- `round-N/TASK/attempt-NN/requests.jsonl` / `requests.csv`：请求序号、观测时刻、slots/live/ready/writing/retired/releasing、GPU KV/CPU/index/prefix bytes、逻辑 context、TTFT、prefill/decode/HTTP 时长、摘要/回滚/重放统计。
- 每个 attempt 的 `resources.jsonl`：10 秒一次的 GPU 总分配、GPU 进程 RSS/HWM、磁盘余量。GPU 总分配包含权重，不等同 KV bytes。
- 所有 agent、服务、verifier 原始日志和 patch 保存在 attempt 内，不覆盖错误尝试。

最终 context 定义为末次 generation 的 prompt tokens + 可见 decoded tokens；另存真正 forward 的 committed history 位置。retained prefix、固定物理容量与 live 占用分别记录。KV 峰值是请求边界采样峰值，不是逐 kernel 瞬时峰值。耗时保留 agent elapsed、含 verifier 的 cell elapsed、HTTP、prefill/decode 和摘要成本；这些分项存在包含关系，不机械相加。

`historical-baselines.json` 冻结 51 条有效 KVMem 历史参照和 64 条 dense 参照。固定顺序优先使用较新的 KVMem 记录，不按得分或速度挑选。缺失 seed 不补跑，明确留空。旧 KVMem 使用 110 GiB CPU backing / 4096 guided-query cap 和旧仓库实现；历史参照不是只开关摘要的严格消融。`historical-comparison.csv` 标记 `controlled_ablation=false`，同时保留两边的任务成败。

## 执行与恢复

工作树入口为 `scripts/kvmem_eval/run_thinking_compaction_campaign.py`，运行中的版本位于实验目录 `harness/`。不要修改冻结文件后直接继续：先定位问题，保存旧 revision、记录修复与 hash，再恢复。每个 trial 前检查冻结文件；模型和任务镜像身份也有校验。

```bash
python3 -u /data/chaidi/qw3-experiments/thinking_summary_first16_pass4_20260922/harness/run_thinking_compaction_campaign.py \
  --experiment /data/chaidi/qw3-experiments/thinking_summary_first16_pass4_20260922
```

调度器使用进程锁防止重复启动。恢复时跳过已有有效结果，从最早未完成 trial 开始。遇到 inference stream error、服务退出或基础设施结果异常，停止本次 worker/服务并写 `needs_attention`；修复后新建 attempt 重试。SIGINT/SIGTERM 会清理本任务拥有的容器和服务，避免把遗留 agent 接入下一次服务。

上线前状态 oracle、baseline/summary HTTP 已在同一 binary 上通过；调度和指标测试覆盖轮次顺序、逻辑/物理 context 分离、错误请求不继承上一轮 KV、任务失败不当作基础设施重试。

## 每 30 分钟定时检查

已启用用户级 systemd timer `qw3-thinking-first16-monitor.timer`，执行 `scripts/kvmem_eval/check_thinking_campaign.py`。`OnUnitActiveSec=30min`，启动后30分钟首次定时运行；已立即手动执行一次，返回正常。用户 manager 的 Linger 已开启。

检查当前轮次/样本、有效 trial 数、supervisor/worker/服务身份、心跳、日志进展、推理错误、数据盘余量和最近 KV/context；记录在实验目录 `monitor/status.json` 与 `monitor/checks.jsonl`。它只读实验状态，不向计时模型服务发请求。

正常运行保持安静；出现新异常或完成全部64次时，经 `codex queue` 投递到当前任务 `01a0c8ad-2510-76e1-a033-b2fb0d9bbb7d`，继续定位修复/汇总。相同异常去重，投递失败保留错误并在下次重试。全部完成且通知投递成功后自动 disable timer。该定时器运行在实验机，不是 Codex Scheduled 界面中的原生任务。

查询：`systemctl --user list-timers qw3-thinking-first16-monitor.timer`。
停止：`systemctl --user disable --now qw3-thinking-first16-monitor.timer`。
