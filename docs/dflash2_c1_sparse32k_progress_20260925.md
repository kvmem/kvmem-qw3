# 32K 稀疏 C1：进行中的正确性诊断

目标保持 active；没有改变主模型 Q8_0、E4M3 FP8 target KV 或运行时算法。
当前工作在 codex/dflash2-qwen38 隔离工作区，未集成或部署，多并发继续后置。

## 新增执行能力

`tests/dflash_kvmem_c1_probe.cpp` 与 `run_kvmem_c1_probe.py` 现在支持冻结的
`sparse_32k_w256_c1_warm`：每条请求先只摄入 32768 个历史 token，随后从该历史
恢复，按冻结查询区间检索并重放。记录历史摄入、热查询、包含摄入的总耗时、逐
输出 ID 和 callback 时间。不传入答案或事实标签；EOS 生效，保留长度截断。
窗口 256、block 32、sink 1/recent 2、生成预留 1024、CPU tier 2 GiB、mean-K
CPU 索引、step 更新均遵循原矩阵。

第一次 `c1_frozen_sparse32k_greedy01` 部分运行保留。因为 retained 路线默认只
记录回退，成功检索的索引/选择计数不在日志中，主动终止了该测试子进程（SIGTERM，
runner 已终止并记录 exit=-15），没有删除失败记录。随后在新 run ID
`c1_frozen_sparse32k_greedy02` 开启 `QW3_KVMEM_PERF_TRACE=1` 重跑。
该 trace 会影响时间，所以所有时序仍仅为诊断，不是正式性能资格。

## 已完成的无投机基线

17 次请求（1 次预热 + 16 次正式）完成。审计每次历史摄入均没有复用上一条回答；
查询恰好从 32768 token 恢复，mean-K 无回退，query_expected/captured 一致，
index 大于 selected，并执行 warm query replay。典型查询覆盖 1025 个索引块，
选中 8 个块。不是全量可容纳路径。

正式样本严格 FINAL 标记评分为 **1/16**，**3 次长度截断**。具体回答显示部分
请求取得标记所在片段，却缺少服务名，回答 UNKNOWN；这提示记录跨 block 的完整性
或排序可能影响质量，但尚未凭此完成根因归属。不能将执行通过解释为问答质量通过。
不能通过扩大窗口、去掉失败样本或改变评分来补过这一冻结场景。

诊断解码中位数 44.73 tok/s；热请求吞吐 41.81 tok/s；包含历史构建后 8.04 tok/s。16 次历史摄入共 217.030 秒。

基线证据为 `benchmark/dflash2/c1_frozen_sparse32k_greedy02/none_progress_audit.json`，
其中保存当时 manifest 的完整快照、原始日志与响应的校验信息及逐例结果。
该文档是阶段性快照；MTP 和 DFlash 正在同一 runner 中顺序执行，完整比较尚未完成。
后续以该目录的完整 manifest/summary 为准，不追认此快照为三路线通过。

## 测量工具

评分脚本已扩展为审计 warm 生命周期与摄入成本；对上一轮两组 4K 原始结果重算，
评分、token parity、时序聚合均与已封存结果完全相同，未覆盖旧 summary。

新增独立 NVML/proc 采样辅助类，主机实测 50 个 10ms 间隔样本，最大间隔约
10.1ms，GPU UUID 匹配。该辅助类尚未接入当前推理进程；其测试只读取 host audit
自身 RSS 和整张 GPU 的 used memory，不能称为本次推理内存峰值。NVML sampled peak
也不能替代 allocator high-water，VmLck/VmPin 不代表全部 CUDA pinned bytes。
源码、NVML 头文件身份和结果保存在 `resource_sampler_host_audit01`。

剩余：完成 MTP/DFlash 的逐 ID 对照、定位质量偏低及任何跨路线差异，继续 128K
有界 C1、全量公开任务与独立重复，以及完整资源计量。公开结果同机参考资格仍未通过。
