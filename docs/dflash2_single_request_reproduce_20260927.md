# DFlash 2.0单请求最终组合复现

本轮10阶段GPU正确性检查、四阶段正式回放、独立审计及开发源码接入均已完成。
结果见[dflash2_single_request_results_20260927.md](dflash2_single_request_results_20260927.md)。
130 tok/s只是参考，最终两种子99.2733/102.5191 tok/s；最高整卡采样峰值
47,989,981,184B，距离48GB仅约10MB，不是allocator硬上限。

## 冻结设置与构建

|项目|设置|
|---|---|
|GPU|单张RTX PRO 6000 Blackwell Server，96GB，SM120a|
|目标|`models/Qwen3.8-27B-Q8_0.gguf`，Q8_0权重|
|目标KV|FP8，page 16；KVMem block 32|
|草稿|`models/Qwen3.8-27B-DFlash2-BF16`，主请求7草稿、私有检索查询5草稿|
|KVMem|K131072 / G65536，prefill预算131072，GPU索引，非bounded路径|
|其他|ctx1048576、prefill chunk2048、C1、T1/top-k20/top-p0.95|
|口径|总输出token/总native decode秒，包含decode内刷新；另报HTTP请求总耗时|

已验证CLI：`benchmark/dflash2/lazy_attention_scores_build02/qw3`。
SHA256：`cddff7aab02908237fb43040522f02b2b21f69361d447f189d73a7feccaf880f`。
完整编译命令、源码和artifact哈希位于该目录`manifest.json`。
开发源码与实测构建对应记录位于
`benchmark/dflash2/lazy_attention_scores_integration_review02/application.json`；
所有6个相关源码匹配，executor SHA256为
`fa77e8f7fb4ff693f2f2f0278b69d1cc1c0ca7b91b5277f128f1cf55c7b3ccb8`。
旧`build-dflash-flashinfer/qw3`及core保留为基线，未覆盖。

以下命令均在`/home/chaidi/qw3-dflash2`执行。运行/输出ID必须是新名称，不能覆盖原始证据。

## 使用已验证binary重放

```bash
/home/chaidi/qw3/.venv/bin/python benchmark/dflash2/run_lazy_attention_scores_replay.py --run-id lazy_attention_scores_replay_reproduce01 --build-id lazy_attention_scores_build02 --validation-id lazy_attention_scores_validation02
```

运行器复核10阶段正确性与全部冻结依赖，获取共享GPU锁，每阶段等待连续60秒
空闲，再独立启动服务：seed1000/1001各两个真实刷新请求，随后seed1000/1001
各完整45请求。四阶段均使用新延迟分配；控制组为之前封存的
`transient_memory_replay03`，不是新鲜的同binary单项A/B。

完整环境与请求在`benchmark/dflash2/lazy_attention_scores_replay01/plan.json`，
每阶段manifest保存实际CLI；请求体还包含刷新和采样参数，不可只复制CLI。
主要新增开关为：

```text
QW3_DFLASH_KVMEM_DEFERRED_REFRESH_SPECULATE=1
QW3_FLASHINFER_VERIFY_PLAN_CACHE=1
QW3_EXPERIMENTAL_FP8_STABLE_PREFILL=1
QW3_EXPERIMENTAL_DFLASH_GUIDED_QUERY=1
QW3_EXPERIMENTAL_SWIGLU_FIXED5120=1
QW3_EXPERIMENTAL_GUIDED_DFLASH_FIVE=1
QW3_EXPERIMENTAL_DFLASH_TRIM_TRANSIENTS=1
QW3_EXPERIMENTAL_LAZY_ATTN_SCORES=1
```

此列表不是完整环境的替代品。主请求长度仍由
`QW3_DFLASH_PRIVATE_MAX_DRAFT_TOKENS=7`设置，五草稿开关只作用于私有查询。
运行器先清理继承的`QW3_`变量，再使用冻结环境。开发源码实验开关默认关闭，
本轮没有替换现有服务。

四阶段正常关闭后，独立报告命令：

```bash
/home/chaidi/qw3/.venv/bin/python benchmark/dflash2/render_lazy_attention_scores_replay.py benchmark/dflash2/lazy_attention_scores_replay_reproduce01 --output benchmark/dflash2/lazy_attention_scores_report_reproduce01
```

报告重核哈希、原始计时/资源、配对输入/输出/查询/KV/边界，给出完整分桶、接受率、
阶段耗时及显存，保留所有短输出和截断请求。输出文件必须不存在。

## 可选：从冻结来源重新构建与验证

依赖是本工作树已保存的源码、base archive、安装的CUDA 13和FlashInfer环境；
不是独立于本工作树的安装器。完整依赖哈希在build manifest。顺序执行：

```bash
/home/chaidi/qw3/.venv/bin/python benchmark/dflash2/build_lazy_attention_scores02.py --run-id lazy_attention_scores_build_reproduce01
/home/chaidi/qw3/.venv/bin/python benchmark/dflash2/validate_lazy_attention_scores.py --run-id lazy_attention_scores_validation_reproduce01 --build-id lazy_attention_scores_build_reproduce01
/home/chaidi/qw3/.venv/bin/python benchmark/dflash2/audit_lazy_attention_scores_validation.py benchmark/dflash2/lazy_attention_scores_validation_reproduce01 --output benchmark/dflash2/lazy_attention_scores_validation_audit_reproduce01.json
```

新的重放使用对应的新`--build-id`和`--validation-id`。正确性审计不运行GPU，
只接受10阶段均关闭的验证链。它重读完整logits、两组4位置×8宽度矩阵及4K/128K
状态和Q行记账。128K合成状态测试使用K64K/G32K，不替代正式K128K/G64K吞吐。

## 比较和资源边界

- 45请求来自冻结的DeepSWE历史轨迹；不运行工具，不提供SWE任务成功率。
- 两个真实刷新请求独立报告，不混进45请求加权指标。
- 正式测量关闭allocation/sync profiling，不与编译、其他GPU作业并行。
- NVML请求10ms采样，实际最大间隔保存于JSON。峰值低于48GB只证明该轮采样，
  不证明allocator硬上限或任意1M历史保证；余量约10MB，不能把它看成宽裕容量。
- 超48GB不会中断测量，会保存完整吞吐与超限结果。校验失败也保留证据，
  不删除断言、不拼接不同轮次为完整测量。
- 本轮五分钟监测在测试链完成后已停止；复现runner本身负责锁、准入和资源归属检查。
