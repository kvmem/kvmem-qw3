# DFlash单请求目标验收记录

2026-09-27，单请求优化、测试、源码接入已完成。以用户最新指示为准：130 tok/s
是参考，按实测空间排序完成尝试、记录不足即可，不作为完成门槛。
显存按本轮正式负载的整卡采样峰值判定；不增加未要求的allocator硬上限承诺。

|要求|当前证据|结论与范围|
|---|---|---|
|Q8_0目标权重、FP8 KV、K128K/G64K、C1|最终replay plan、native预算审计、冻结模型哈希、四阶段实际命令|完成；正式配置一致，合成128K状态测试另用K64K/G32K|
|等待工具/XML/围栏闭合时继续投机|`dflash_deferred_refresh.hpp`、native backend；最新4K/128K各28请求及CPU策略日志|完成；闭合候选进入p/q验证，保留512 token紧急余量、紧急刷新和取消恢复|
|降低target验证开销及长attention数值一致性|稳定attention、CPU plan测试、边界/memcheck记录；最新1K/128K各32组逐层/全logits精确检查|完成；缓存仅复用shape元数据，当前源码仍每次展开物理KV页索引|
|四组原始优化开关对照|`deepswe_two_recovery01/diagnostic_reference,refresh_only,plan_only,both`|四组各3请求，00/10/01/11，响应相同且资源无异常；不是四组完整45请求|
|完整长历史吞吐及指标|`lazy_attention_scores_report01.md/json`，独立重读四阶段全部计时、资源、输入/输出/策略|完成；两种子各45请求99.2733/102.5191 tok/s，分桶、接受率、各阶段耗时、短输出/截断均报告|
|真实刷新、私有查询与恢复|两个最终Scriggo刷新场景；私有查询pilot和最新状态回归|完成；最终长请求46.7630/52.8140 tok/s，输入/输出/查询/KV/边界匹配对照；未声称完整Q张量位级证明|
|按剩余空间逐项优化|GPU时间线、FFN算子/模型检查、私有五草稿配对、临时回收和延迟分配|完成；收益、无明显收益、负收益及未尝试的主请求宽度均记录；130未达不阻止本轮关闭|
|普通QW3兼容|最新paged/main/原生MTP三模式off/on，共每侧19组完整词表logits|完成所测兼容检查；普通main/MTP均在使用前补分配，paged路径不分配；不外推普通QW3性能|
|整卡显存低于48,000,000,000B|四阶段原始NVML资源记录，无外部GPU作业；最终独立报告|本轮采样要求通过；最高47,989,981,184B，余量10.019MB；不是硬上限或任意1M历史保证|
|不新增BF16目标权重支持|Q8模型及环境、当前`qwen_weights.cpp`无本轮diff|完成；草稿和既有激活精度不等于新增BF16目标权重支持|
|代码对应实测版本|最终application.json、5个archive对象来源核对、6个源码SHA256|完成；最后executor补丁已应用，源码字节与已编译验证候选一致；`git diff --check`通过|
|隔离GPU、五分钟检查、不抢占作业|共享锁、每阶段连续60秒空闲准入、资源归属记录、systemd monitor|完成；四阶段服务退出0，验证/回放监测器完成后已停止，无服务替换|
|可复现交付|最终报告、源码/构建/模型/输入/输出哈希、复现页、失败及负结果文件|完成；旧基线binary保留，实验开关默认关闭，未进入并发开发|

## 权威证据

- `benchmark/dflash2/lazy_attention_scores_validation02/manifest.json`：10阶段完成。
- `benchmark/dflash2/lazy_attention_scores_validation_audit02.json`：重新解析全部验证；
  4K/128K各28请求，私有Q步骤370/374；128K的query_cancel释放未触发，4K覆盖6次。
- `benchmark/dflash2/lazy_attention_scores_replay01/manifest.json`：两个刷新＋两个完整45请求完成。
- `benchmark/dflash2/lazy_attention_scores_report01.md/json`：最终四阶段独立报告，
  每个输入/输出/查询/KV/边界匹配对应已封存控制组；控制组不是新鲜同binary A/B。
- `benchmark/dflash2/lazy_attention_scores_source_provenance02.json`与
  `lazy_attention_scores_integration_review02/application.json`：构建来源及最终源码接入。
- `benchmark/dflash2/single_request_integration_build01/build_12.log`、`build_14.log`：
  CPU刷新策略和plan的split/rollback/invalidation/overflow检查通过。
- `stable_prefill_full_report02`、`guided_dflash_pilots_report01`、
  `guided_dflash_regression_audit01`、`single_request_integration_audit01`、
  `swiglu_fixed5120_report01`、`swiglu_fixed5120_model_report01`、
  `guided_query_width_report01`、`transient_memory_report03`保留各项收益的独立来源。

## 保留失败及比较边界

临时回收validation01的features分支未覆盖、validation02复制执行文件丢失x位，
以及延迟分配初版探针选择FP8不支持的native路径，均保留原文件和错误/撤回记录。
修正后使用新的build/validation ID；没有删除断言或把未执行分支算通过。
最新通过链为lazy build02 → validation02 → replay01 → 独立报告 → 开发源码接入。

旧attention与新attention的生成轨迹有差别，旧/新整体吞吐差额不是固定输出因果对照。
同一份最终分配候选与上一临时回收组合逐请求匹配。所有负桶、短输出、截断及
小幅负变化保留；冻结回放不执行工具，不证明SWE任务成功或质量不下降。
无硬件计数器数据，不称已到达硬件极限。48GB只在此负载的采样意义上达标，余量很小。

[结果](dflash2_single_request_results_20260927.md) ·
[复现](dflash2_single_request_reproduce_20260927.md)

此前阶段审计文本另存于`benchmark/dflash2/single_request_completion01/previous_docs/`，
原始实验文件均保留。
