# DeepSWE 原 16 样本：KVMem + DFlash，关闭 thinking 摘要

2026-09-29 22:21 CST 启动。用户要求暂停摘要实验，将原 16 个样本全部重新执行；
每个样本一次，seed=1000。每个任务从干净容器开始，按原顺序串行运行，旧任务结果
不计入新批次。原始 thinking 生成保持开启，只关闭 thinking 摘要压缩。

## 配置与身份

- 模型：Qwen3.8-27B Q8_0，DFlash 2 BF16，FP8 KV。
- KVMem budget：131072；generation budget：65536；CPU backing：48 GiB。
- 逻辑 context：3145728；物理仓库：524288 tokens（16384 × 32）；CPU Mean-K index，
  step 更新、delete GC、guided query cap 512；客户端自动 compact 关闭。
- 采样：temperature=1.0、top_p=0.95、top_k=20、thinking budget=28672、
  presence penalty=0、repetition penalty=1、seed=1000。
- 同一份已验证二进制，SHA256：
  `e14b429817f42cedee5659c786087f0265f6335a58073d884a2108782ca9a2ad`。
  冻结的运行时代码与启动前工作区代码一致。实际推理配置只删除
  `--thinking-summary --thinking-summary-min-tokens 1024`；缺省禁用摘要。
- 16 份任务说明、32 个 agent/verifier 镜像 ID、Claude Code 2.1.148 二进制已核对。
  agent、验证器和评分脚本与刚暂停的摘要实验一致。调度器仅调整报告标签、分组和
  arm 元数据，完整文件哈希记录在 manifest 中。
- 继续使用用户授权的空闲 GPU 显存；每个任务开始前要求至少 44 GiB 空闲。
  与已有服务共享 GPU，耗时包含资源竞争。

## 路径与运行

数据目录：`/data/chaidi/qw3-experiments/kvmem_dflash_no_summary_seed1000_20260929`。
工作区入口：`benchmark/kvmem_dflash_no_summary_seed1000_20260929`。

后台服务：`qw3-kvmem-dflash-no-summary-first16-s1000-20260929.service`。
监控服务：`qw3-kvmem-dflash-dashboard-20260929.service`，沿用
<http://127.0.0.1:8322>，界面标明摘要已关闭。对话结束后后台继续执行。

每个样本的最终 context、总耗时（含验证）、通过情况及 F2P/P2P 分项保存在
`results.json` / `results.csv` 并显示于界面；运行中的数值仍标为当前值。
`state.json` / `live.json` 记录当前任务，`summary.json` / `report.md` 记录已完成结果。
原始轨迹、patch、验证器日志以及请求/资源采样保存在各 `round-1/*/attempt-*`。
任务未通过仍是有效结果，不重试筛选成功；基础设施或推理错误则保留证据并暂停。

旧摘要实验保持暂停：`thinking_summary_isolated_first16_seed1000_20260929`。
暂停前已完成 3 个任务，均未完全通过；第 4 个 awilix 任务中断，未评分。
旧目录 `operations/latest-pause.json` 指向暂停审计，包含轨迹和未完成工作区备份。
新目录 `summary-reference.json` 仅保存其 3 个有效完成结果供参照。

## 比较口径

`historical-comparison.csv` 随任务完成自动生成同任务、同 seed 的历史比较。
本轮可检查当前代码下这 16 个样本的通过情况。历史原生 MTP 批次的二进制、
仓库、CPU backing 和 guided-query 等配置也有差异，因此历史得分变化不能单独
归因于 DFlash；16 × 1 也不等于此前 16 × 4 的 pass@4。

当前摘要批次与本轮使用相同二进制和配置，但前者只有 3 个有效完成结果；
后续比较应分别注明参照来源，不将中断任务当作失败评分。

## 启动核验

实际模型进程保留 DFlash 2、K128K/G64K 参数，没有摘要开关。首批已完成请求均为
成功状态；`native dflash_spec_summary` 显示 `enabled=true` 并记录实际草稿接收，
未出现 thinking 摘要输出。启动审计见 `operations/launch-audit.json`。
浏览器检查确认 16 个任务、新批次标识、摘要关闭提示、请求更新均正常，未发现
JavaScript 错误；截图和检查记录见 `dashboard/`。这些只确认新实验已正常启动，
任务通过情况以最终 verifier 结果为准。
