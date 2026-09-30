# KV Memory 10% 重算：256K 完整四类测试

本次延续 [64K 实现与测试](kvmem_cacheblend_20260925.md)，将输入扩展到 262144 tokens，覆盖现有冻结数据中全部 256K 样例。这里的“完整”指该冻结集合的四类任务、五个深度、三个随机种子，不代表完整 RULER 所有任务。

## 固定配置

- 任务：`niah_single_2`、`niah_multikey_2`、`niah_multivalue`、`niah_multiquery`。
- 深度：10%、30%、50%、70%、90%；种子：1000、2000、3000。
- 60 个不同输入，本次完整运行开启 10% 重算组。按用户要求复用已完成的 60 个关闭重算基线，不再补跑关闭组。
- Qwen3.8-27B-Q8_0，FP8 KV，MTP4，32K 召回/prefill 窗口，16K generation budget；先盲 ingest，后给 query，完整 query replay。
- 开启组 `QW3_KVMEM_BLEND_RATIO=0.10`，不设额外 token 上限；关闭组为 0。256K 是输入长度，重算比例的分母仍是召回窗口。
- thinking budget=4096，max output=8192；温度 1、top-p 0.95、top-k 20。每个请求使用相应样例的种子。
- 冻结二进制 SHA256：`48b6415c4adfad971dd4a55896d59ea029a1861a7d8055df8aa834c2214f29cd`。

输入文件哈希和 60 个 `(task, depth, seed)` 组合已逐一核验。开启组必须有实际重算、重算量不超过 10% 预算、`missing_source_tokens=0`；两组都必须执行 query replay，且不能触发 thinking compaction。质量分数低于 100 的有效样例照常保留。

## 复现

```bash
.venv/bin/python -u scripts/kvmem_eval/run_cacheblend_niah.py \
  --reference /data/chaidi/qw3-experiments/ruler_niah_rerope_ablation_20260925 \
  --out /data/chaidi/qw3-experiments/cacheblend_20260925/niah256k_r10_on \
  --binary /data/chaidi/qw3-experiments/cacheblend_20260925/bin/qw3 \
  --length 262144 --depths 10 30 50 70 90 --seeds 1000 2000 3000 --arms on
```

已有目录续跑添加 `--resume`。脚本支持 `--resume --extend-cohort`，只允许加入相同冻结数据中的样例，禁止修改或删除原有样例。

最初计划重跑两组，关闭组完成 29 个样例后按用户要求停止，保存在 `niah256k_r10/`，其评分和精确数字集合匹配标志与原有基线的对应 29 个样例全部一致。本次最终复用 `ruler_niah_rerope_ablation_20260925/on/results.json` 中的全部 60 个 256K 样例；该目录的 `on` 指开启 re-RoPE，仍是关闭 blend 的基线。

历史基线的二进制、thinking/output 上限与当前不同，完整 60 对属于历史结果比较，不能宣称同一二进制严格 A/B。历史基线 thinking budget 为 28672，当前为 4096；历史 60 个样例均正常停止，最多生成 2406 tokens，未达到当前上限。另对本次保留的 29 个同配置、同二进制基线做配对分析。

## 结果

已完成全部 60 个开启样例，四类任务各 15 个；输入长度、五个深度、三个种子及文件 SHA256 均核验通过。进程正常退出，本任务启动的服务已停止。

| 任务 | 历史基线答案覆盖率 | 开启 10% 覆盖率 | 提升（百分点） | 整题全对：基线 → 开启 |
| --- | ---: | ---: | ---: | ---: |
| 单针 `niah_single_2` | 100.00% | 100.00% | 0.00 | 15/15 → 15/15 |
| 多键 `niah_multikey_2` | 46.67% | 60.00% | 13.33 | 7/15 → 9/15 |
| 多值 `niah_multivalue` | 26.67% | 100.00% | 73.33 | 0/15 → 15/15 |
| 多查询 `niah_multiquery` | 68.33% | 100.00% | 31.67 | 0/15 → 15/15 |
| 合计／均值 | 60.42% | 90.00% | 29.58 | 22/60 → 54/60 |

逐样例比较为 **32 个改善、28 个持平、0 个下降**。精确数字集合匹配也是基线 22/60、开启 54/60。所有开启样例均正常 `stop`、thinking 完整闭合，最多生成 1611 tokens，没有达到输出或 thinking 上限。

另在本次保留的 **29 个同二进制、同配置样例**上，覆盖率由 65.52% 升至 86.21%，整题全对由 14/29 升至 25/29；11 个改善、18 个持平、0 个下降。这一子集也支持改善方向，但它是停止重复基线时保留的顺序子集，不是额外随机抽样。

60 次均实际重算 **102 块、3264 tokens**，窗口为 32704 或 32736 tokens；`missing_source_tokens` 全部为 0。每次合并为 61–80 次连续 forward，平均 70.18 次。开启组不存在只设置开关却未执行重算的样例。

## 耗时与结论

- 重算阶段平均 **2.954 s**，中位数 2.940 s，范围 2.697–3.191 s。该计时包括备份及临时状态操作，不包括未来恢复覆盖，也不是完整 TTFT。
- 开启组盲 ingest 平均 81.00 s；随后回答请求平均 6.71 s；两部分合计平均 87.71 s，不含 reset。
- 历史基线回答请求平均 4.64 s。历史运行环境、二进制和输出长度不同，不据此宣称固定的减速倍数。
- 开启组各样例开始时的 GPU 进程采样均仅发现本任务服务；这些采样不是全程 GPU 独占证明。

在这批 256K 冻结样例中，10% 重算使观察到的答案覆盖率提高了 29.58 个百分点，收益集中在多值和多查询；先前 64K 样例关闭时已满分。它仍是使用检索分数挑块、对 DeltaNet 进行近似处理的实现，不能据此推断所有长上下文任务均有同样收益。本次未修改模型执行代码或将开关改为默认开启。

剩余 6 个失败均为多键任务，正常结束但未返回目标答案：

| 种子 | 失败深度 |
| --- | --- |
| 1000 | 70% |
| 2000 | 10%、30%、70% |
| 3000 | 10%、70% |

开启组原始记录位于 `/data/chaidi/qw3-experiments/cacheblend_20260925/niah256k_r10_on/`，包含冻结 cohort、运行脚本快照、配置、逐样本响应和日志、GPU 进程记录，以及复用基线快照和审计说明。

- [验收审计与汇总](/data/chaidi/qw3-experiments/cacheblend_20260925/niah256k_r10_on/acceptance-audit.json)
- [逐样例配对结果 CSV](/data/chaidi/qw3-experiments/cacheblend_20260925/niah256k_r10_on/paired-results.csv)
- [开启组原始结果](/data/chaidi/qw3-experiments/cacheblend_20260925/niah256k_r10_on/results.json)
- [历史基线复用说明](/data/chaidi/qw3-experiments/cacheblend_20260925/niah256k_r10_on/baseline-reuse-audit.json)

开启组 `results.json` SHA256 为 `5ac750dc361114a5f94863631a9062a00fc0d9695e713b7900d7732606aa8426`。

剩余六例的原因及 24 次控制实验见[失败诊断](/home/chaidi/qw3/docs/kvmem_cacheblend_failure_analysis_20260926.md)。诊断不改变本报告的正式准确率。

2026-09-27 完成[结果复核及跨实现对照](/home/chaidi/qw3/docs/kvmem_cacheblend_reaudit_20260927.md)：独立复核原始 60 例，另做六例、12 次全新进程开关对照；抽查收益复现。两边历史 256K 测试的数据与配置不同，不能直接用增幅判断同一个重算算法的效果。
