# 单请求吞吐筛选后的临时显存优化候选

**最终状态：已完成。** 临时回收和普通分数缓冲区延迟分配已接入开发源码，
10阶段正确性检查和四阶段正式回放均通过独立审计。最终完整两种子
99.2733/102.5191 tok/s，最高整卡采样峰值47,989,981,184B，余量约10MB。
结果见[dflash2_single_request_results_20260927.md](dflash2_single_request_results_20260927.md)。
以下保留实验过程与失败原因；其中带时间的“待执行/未接入”是当时状态。

01:26更新：私有查询七/五草稿对照已关闭，五草稿两种子局部decode+1.94%/+1.42%，
保留为候选。`transient_memory_build01`CPU构建已通过，仅在独立源码实施下列候选，
**没有接入开发executor，没有新的显存收益结果**。Q8_0/FP8、K128K/G64K、ctx1M保持不变。
初次`transient_memory_validation01`数值通过，但probe在prefill后才开特征采集，
features缩小未覆盖，按要求失败保留。build02仅把探针采集前移，不改变模型
binary/core或断言；validation02的1K已通过并实际覆盖两个缩小分支，128K执行中，
随后4K/128K状态回归。replay02已接续排队，所有检查通过才做真实刷新A/B和
两个完整45请求复测。
01:43补充：validation02的1K/128K均已通过，接续state时因build02复制文件丢失
执行位而未启动。build03只恢复新副本权限，四个artifact字节哈希全部不变。
validation03重核并复用通过的parity，只接续状态回归；replay03等待它全部通过。
旧失败记录保留。均设置五分钟只读监测。继承的取消用例在主请求延期输出阶段取消，
不保证命中私有查询中途取消；报告单独记录`query_cancel`实际覆盖，不能混称。
01:49实测：4K回归28请求通过，`query_cancel`释放实际命中6次，`query_end`10次，
batch/features各28次；后续输出与查询/KV策略均匹配旧参考。这证明此次取消分支
释放后恢复路径通过，不把它扩大为任意时刻异步用户取消的穷尽证明。
02:17更新：validation03四阶段全部关闭，独立审计
`transient_memory_validation_audit03.json`复核封存文件和原始资源。128K状态组
28请求/14配对输出及查询/KV策略一致，Q行记账374步通过，batch/features各28次、
query_end 16次；此组未触发query_cancel，4K已有6次覆盖。128K整卡采样峰值
43,615,322,112B，无外部GPU进程。合成正确性不替代K128K/G64K正式显存结果。
replay03已启动真实刷新对照，随后自动执行两个45请求组合回放；开发executor
接入及最终取舍仍待这些吞吐/显存测量。
02:20首个真实刷新种子配对已完成并独立复核，见
`transient_memory_first_pair_audit03.json`。同一binary只切换回收开关：两请求
加权native decode 49.4693→50.2408 tok/s；整卡采样峰值
47,736,225,792→46,937,210,880B，减少799,014,912B。输入、输出、查询、KV
选择和刷新边界一致；实际刷新1次，batch/features各回收2次、query_end 1次。
这是一个种子的小样本，不作为显著提速或完整45请求低于48GB的结论；第二种子
及两个完整回放仍由同一队列顺序执行。
02:26第二种子配对也已关闭：两请求55.3824→55.4211 tok/s，变化+0.07%，
基本持平；整卡峰值47,729,934,336→46,930,919,424B，同样减少799,014,912B。
独立审计`transient_memory_refresh_pairs_audit03.json`重新核对四臂输入、输出、
查询、KV选择、边界及原始资源，两组一致，均无外部GPU作业。当前支持的结论
是刷新小样本减少约0.80GB且未观察到decode退化，不是稳定的吞吐提速保证。
完整45请求仍待执行，显存目标尚未关闭。
130 tok/s不是门槛；不为达到吞吐数字继续无依据搜索。

## 优先尝试：阶段结束后缩小临时分配

已有`deepswe_memory_diagnostic01/allocation_summary.json`只覆盖大于等于1MiB的
owned CudaTensor，不能冒充完整分配账本，也不能替代NVML整卡峰值。它记录：

|分配|既有观测|生命周期检查|拟验证方向|
|---|---:|---|---|
|`kvmem_guided_query`|805,306,368B|`kvmem_end_guided_query_probe`已复制接受的Q行到评分副本，原采集区仍保留；cancel也只重置标志|评分副本复制完成后释放采集区；取消路径同样释放，不改变查询容量上限|
|`prefill_batch_scratch_arena`|353,107,968B|`ensure_batch_scratch`只增长；开始下一次forward时可替换整个arena，现有代码已先销毁views再销毁storage|从大prefill切入至多8行验证时缩到8行；下一轮prefill按原2048 chunk重新增长|
|`dflash-target-features`|曾达334,950,400B，依实际rows而变|`prepare_dflash_features`先清空feature view，但底层只增长|开始新的小批forward时缩到8行；必须确认上一批特征已提交到draft上下文|

前两项只是既有观测大小，不代表一定同时位于最终组合峰值。不得直接将其
相加后从49.05GB中减去并宣称达标。缩小arena会更换views及stride；需要实际
检查逐层/全logits、私有查询Q行、主分支恢复，以及prefill→decode→refresh→
prefill重复切换。释放和重新分配可能引入同步与重分配成本，须同时测TTFT、
含刷新decode和整卡峰值。

拟用一个默认关闭的实验开关，先在独立executor/core实现。保留原路径可做
同一binary开关对照；不在每个小批次反复收缩，统一保留8行。释放Q采集区
必须在异步D2D完成后，且取消/异常路径不能因新增异常破坏RAII恢复。不能
释放仍由StateSnapshot、checkpoint或DFlash feature view引用的张量。

## 次优先候选，当前不改

`g_kbar_multi`在ctx1M按固定层stride预留1,073,741,824B；固定stride用于
续接与历史切片复用。直接改成当前history大小会破坏增长时的旧层偏移，不能
作为一行缩容修改。动态增长需迁移并保留各层完整索引，还要覆盖decode追加。

已有`--kvmem-index-placement cpu`可移走GPU索引，但使用分块GPU评分与额外
staging/statistics，不是纯粹减掉1GiB；也需验证分数、KV选择和吞吐。先试
不改变索引布局、检索算法和K/G预算的临时缓冲区生命周期，再决定是否需要它。

## 完整首种子超限后的下一项：普通attention分数缓冲区

02:43首个完整45请求通过输出/策略一致性审计，99.2919 tok/s，但峰值
48,086,450,176B仍超48GB。当前第二种子继续完整测量，不能用局部刷新峰值
46.94GB替代全量结论。

源码显示`scores_`在`ensure_scratch`按`n_heads * kv_ctx_size * sizeof(float)`
无条件分配；当前24头/ctx1M为100,663,296B。仅两个普通attention调用消费它，
分页路径无需此张量。`lazy_attention_scores_review01`保留三处独立改动：
默认关闭的开关延迟初始化；主请求普通路径在backend.begin和CUDA Graph
capture之前补齐分配；原生MTP普通路径使用前补齐分配。普通路径仍分配原容量，
不缩ctx、K/G或索引，也不改变attention计算。

`prepare_lazy_attention_scores.py`及其产物已记录SHA256，只完成源码准备，
尚未编译或GPU执行。原prepared临时回收补丁和开发源码均未覆盖。后续需要
检查分页路径确实无materialized记录、普通main/MTP确实进入补充分配且与开关
关闭时logits一致，再做相关状态/刷新检查和完整吞吐复测。100.7MB是代码中的
分配量，不是已测得的峰值下降，不能据减法提前宣布低于48GB。

02:57更新：此前六阶段回放已全部关闭并独立重解析，完整45请求两种子
99.2919/102.4750 tok/s，峰值48,086,450,176/48,090,644,480B。报告
`transient_memory_report03.md/json`保留两个未达48GB的结果。

延迟分配候选`lazy_attention_scores_build02`已独立CPU构建成功，executor
源码仍为review01的`fa77e8f7fb4ff693f2f2f0278b69d1cc1c0ca7b91b5277f128f1cf55c7b3ccb8`。
初版probe误用FP8不支持的native decode，已在编译/GPU执行前撤下01队列并保留
`withdrawal.json`；02仅改探针为非分页FlashInfer，未改候选executor。
CLI SHA256为`cddff7aab02908237fb43040522f02b2b21f69361d447f189d73a7feccaf880f`，
core为`71d036ee530c10e9dbb54beb7c63592a71465d6c8eb41f51c08125b6e78723ab`。
`lazy_attention_scores_validation02`运行中，首项分页eager参考已通过，仍待
分页lazy配对、main/MTP实际补充分配、1K/128K数值及4K/128K状态，共10阶段。
五分钟监测已启动；正式回放尚未用此新binary，开发源码及运行服务未替换。

03:17更新：三模式off/on、1K/128K parity、4K状态已通过，共9/10阶段。
独立审计重新解析每侧19个完整词表logits，有限且逐字节相同；main/MTP各在
所需路径补充分配一次，分页路径从未补分配。4K状态28请求、370私有查询步骤，
输出、查询和KV策略与临时回收参考相同，batch/features各回收28次，正常Q结束
10次、取消6次；整卡采样峰值42,856,153,088B，无外部作业。此合成检查不替代
正式K128K/G64K负载显存。128K状态接续中，四阶段正式回放已排队等待全验证完成。

开发executor的组合接入补丁位于`lazy_attention_scores_integration_review02`，
包括私有五草稿、临时回收及延迟分配；`git apply --check`通过，尚未应用。
候选源码SHA256与build02一致，最终接入须再检查当前开发源码仍等于封存before，
并以完整正式复测结果决定，不覆盖其他改动。

## 最小闭环

03:42验证10阶段全部完成，并经`lazy_attention_scores_validation_audit02.json`
独立重解析。128K的28请求/14配对及374私有步骤均通过；零/部分/全部接受
138/202/34，正常Q结束释放16次，未命中query_cancel释放（4K命中6次）。
整卡采样峰值43,514,658,816B，原始资源无外部作业；这是K64K/G32K合成验证，
不代表正式负载显存。03:43正式回放已开始首个真实刷新场景，开发补丁尚未应用。

1. 已完成：10阶段正确性检查、独立审计；吞吐宽度选择与临时回收组合已完成。
2. 运行已排队的两种子真实刷新和完整45请求，独立复核输入/输出/查询/KV选择，
   保持正式吞吐关闭allocation/sync profiling。
3. 根据实测决定组合接入，补齐最终复现与统一报告。
4. 观察48,000,000,000B目标、短历史负桶与吞吐变化。
   采样峰值通过只对实测工作负载有效，不声称任意1M历史的allocator硬上限。
