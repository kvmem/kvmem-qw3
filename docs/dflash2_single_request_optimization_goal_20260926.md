# DFlash 2.0 单请求性能测试与优化目标

2026-09-26 根据用户要求建立持续目标，接续现有验证，不重新启动已在运行的 GPU 作业。

**2026-09-27最终状态：本轮单请求优化、测试及开发源码接入完成。**
两种子各45请求加权native decode为99.2733/102.5191 tok/s；最高整卡采样峰值
47,989,981,184B，低于48GB约10MB。10阶段正确性、四阶段正式回放及独立审计
均通过，开发源码与实测版本匹配。130 tok/s按用户最新指示不作为门槛；
收益、负结果、短输出/截断和显存余量限制均保留，未开展并发开发。
权威结论见[最终结果](dflash2_single_request_results_20260927.md)、
[逐项验收](dflash2_completion_audit_20260927.md)及
[复现入口](dflash2_single_request_reproduce_20260927.md)。以下执行条目保留阶段历史。

## 范围与验收

主线是 Qwen3.8-27B **Q8_0 目标权重、FP8 目标 KV、KVMem K128K/G64K**。
优先完成以下两项优化，再根据真实耗时决定进一步修改：

1. 达到刷新阈值、等待工具/XML/代码围栏闭合时继续投机，保持原安全边界、
   查询与 KV 选择策略、取消恢复和采样正确性。
2. 降低 target 验证开销，包括 plan-cache，以及已发现的长历史 attention
   分块边界数值一致性修复；若调度缓存收益不足，继续定位验证 kernel 的主要耗时。

完整记录未优化、只开第一项、只开第二项、两项均开的对照，保留负结果与失败。
主要指标为同配置的 `总输出 token / 总 native decode 秒数`，同时按历史长度
分桶，报告接受率、每批接受长度、draft/verify、零草稿串行、刷新耗时及显存。
**用户于22:32调整验收方向：130 tok/s仅作参考，不是必须达到的最终门槛。**
先根据实测耗时与可改善空间排序，再从高到低逐项尝试；以同配置A/B与正确性
证据决定保留或撤回，记录无收益、负收益和未解决的限制。合理的高优先级空间
处理完后，如继续优化的预期收益很小，允许以实测吞吐结束本轮单请求调优，
无需为了追到130持续追加试验。不能用算子微基准或短合成提示代替DeepSWE吞吐。

整卡最终目标低于 **48,000,000,000 字节**。**用户后续明确要求先测吞吐，暂缓
显存优化；当前性能测试不因超过 48GB 停止，显存只做观测。** 核对实际 K/G 窗口，不能以缩窗或改变
权重、采样设置制造提速。保留短输出、截断与失败；NVML 采样峰值不等于瞬时
allocator 峰值。冻结请求重放不执行工具，不算真实 SWE 任务通过。

本目标不增加 BF16 target 权重支持，不先行开展多并发开发。

## 执行过程记录（历史状态）

### 2026-09-27 02:17：最终组合正确性关闭，正式回放已启动

`transient_memory_validation03`四阶段通过；1K/128K数值检查沿用经字节哈希确认
相同的已通过probe，4K/128K状态回归各28请求/14组对照为新执行。独立审计
`transient_memory_validation_audit03.json`重新解析输出、查询、KV策略和原始
资源记录。128K私有查询374步Q行记账全部通过，接受类别覆盖齐全；主请求
greedy、CPU/GPU sampling、紧急刷新及取消恢复通过。4K实际query_cancel
回收6次，128K该分支未触发，保留覆盖边界。

`transient_memory_replay03`已自动转入四组真实刷新开关A/B，随后两种子各45
请求完整回放。首个刷新参考臂已结束。五分钟监测保持运行，不与编译或其他
GPU实验重叠。临时回收和私有五草稿仅在隔离构建中；准备好的46行开发源码
补丁待正式吞吐、输出/策略一致性及显存结论后决定接入。

### 最新优化顺序与停止条件（22:32用户指示优先于后面的历史记录）

| 暂定顺序 | 优化空间 | 现有证据与适用范围 | 下一步依据 |
| --- | --- | --- | --- |
| 1 | 剩余target验证计算 | 新路径冷前缀verify约31.7ms/批，占decode约82%；每批都支付 | 完成新路径profiling，按各算子可消除的实际成本重新排序；不沿用旧attention占比 |
| 2 | KVMem私有检索查询投机 | 真实刷新中generated_query约8.4–14.7秒；仅影响发生查询/刷新的阶段 | 先复算当前新attention事件对照，确认查询占比，再试独立DFlash查询状态与恢复 |
| 3 | 草稿长度与调度 | 固定7草稿，新路径冷前缀每批接受约2.2–3.2；缩短可能同时损失产出 | 依据实际接受长度和验证成本试少量候选，包含完整吞吐，不仅计算理论节省 |

上述是按当前证据的暂定顺序，不是承诺三项都能提速；新profiling若改变成本排序，
按证据调整。每项记录假设、覆盖范围、正确性、同配置吞吐、接受长度、分阶段
耗时及显存观察，并明确保留/撤回/后续处理。已证明变慢的SwiGLU对齐加载、
interleave/pipeline等试验不在没有新依据时重复。不能仅因最终不足130而否定
有效优化，也不能仅因单条超过130而跳过正确性与完整工作负载检查。

正在运行的full02/refresh02/profile02冻结依赖保持不变。profile02原先的130
条件仅是旧队列的调度条件，不再是任务验收条件；若它因达到130跳过，本轮
仍需另行执行无该门槛的profiling，以完成用户要求的优化空间检查。显存优化
继续后置，尚未用该调整替代单请求正确性检查或提前开始并发。

- 已完成：build10 独立构建与 CPU 策略检查；长 attention 八项诊断；
  长历史六请求策略诊断；完整回归的六项基础 GPU 检查。
- 已完成：4K 刷新回归 **28 请求、14 组 token 与策略对照**，包括延期投机、
  接受安全边界、紧急刷新和取消路径。整卡采样峰值 **42,969,399,296 字节**。
- 已完成：128K 历史刷新回归 **28 请求、14 组 token 与策略对照**；CPU/GPU
  sampling、紧急刷新和取消恢复覆盖齐全，输出文件哈希与资源观察已复核。
- 已完成：三项真实 >=128K 前缀的六配置诊断，包括用于隔离 attention 修复的
  prepartition 对照；同一 build10 四种优化开关的输出相同。
- 已完成：吞吐优先队列的三组完整 45 请求回放。双关与双开 seed1000
  为 78.63 → 78.92 tok/s（+0.37%），尚无显著收益；双开 seed1001
  为 79.71 tok/s，不同生成轨迹不作为第二组配对收益。真实刷新事件四组亦已完成，见下方19:58记录。
- 根据结果：如未观察到真实延期刷新，补测相关事件；冻结输出上限 4096，
  不能默认覆盖历史 Quill 输出 5114 token、等待闭合 2580 token 的问题轮次。
- 后续：对主要剩余验证耗时进行 profiling，修改后重复相关正确性检查及相同
  工作负载 A/B；必要时接真实工具循环与 verifier 验证任务行为。

首轮吞吐优先队列 `deepswe_throughput_priority01` 已完成：重新完整执行 45 请求的双关、
双开 seed1000、双开 seed1001，再执行单独冻结的 6 个真实刷新前缀四组对照。
显存诊断 `deepswe_memory_diagnostic02` 已按用户要求停止；9/25 请求已完成，
停止原因是用户调整优先级，并非显存失败或正确性失败。保留该次原始中断记录。
新队列使用已验证 build10，关闭 allocation/sync profiling，不修改模型执行代码。
该轮每五分钟检查使用 `qw3-dflash-throughput-priority01-monitor.timer`，状态文件为
`benchmark/dflash2/deepswe_throughput_priority01_monitor/latest.md`。
原完整重放在 197K 请求触及显存限制的失败记录仍保留，详见下面的 17:57 记录。
现有测试执行器及冻结依赖不得在运行中修改；新分析脚本独立于执行器。

### 用户最新优先级：先完成吞吐，再优化显存

`run_deepswe_throughput_priority.py` 沿用已冻结的 build10、输入顺序、采样设置与
K128K/G64K，复核并复用之前通过的八项 GPU 正确性检查与六组诊断。
此前中断的 24/45 数据不拼接为完整基线；每组从新服务重新执行全部 45 请求。
`throughput_observation.py` 继续记录 NVML 显存、首次越过 48GB 的时间与数值，
但显存超目标只作为观测，不触发终止。外部 GPU 作业、监测错误、服务异常、
输入/输出 token 数或 artifact 校验失败仍会使吞吐测量无效。
报告分别保存 `usable_completed_throughput_measurement` 与原来的显存合格状态，
不把显存超过 48GB 的测量标为最终显存目标通过。

显存诊断01的缩减三请求序列峰值 **47,723,642,880 字节**，未复现完整序列的
48,566,697,984 字节。发现私有查询采集缓冲区约 **768 MiB** 在评分副本建立后
仍被保留，但尚未修改其生命周期；该方向待吞吐测试结束后处理。

吞吐优先队列已实际启动 GPU 服务。前 **24/45** 个基线请求复跑时，逐请求的
序列化输入、生成结果（仅忽略服务分配的响应/工具调用 ID）、KV 选择哈希均与
旧中断基线一致，环境也一致。证据为
`benchmark/dflash2/deepswe_throughput_restart_audit01.json`；这只证明前序复现，
不能当作完整吞吐或显存验收。

已有三个固定 128K 前缀的接受长度分析保存为
`benchmark/dflash2/deepswe_verify_width_hypothesis01.json`。例如 Effect 每批
平均接受约 2.0 个草稿；回顾性截断至 3 个草稿，要求每批总耗时下降超过
约 15.6% 才可能获益。该计算只给出实验筛选条件，不是短草稿的实测吞吐；
短块可能改变草稿候选与采样轨迹。过去短历史宽度 6 的试验收益不一致，不能
沿用为长历史优化结论。当前 45 请求与真实刷新对照仍固定 7 个草稿。

### 9月27日01:38 临时显存候选补齐探针覆盖，1K通过，128K执行中

01:49补充：validation03的4K状态回归已通过28请求、14组token与策略对照。
独立重新解析日志并核对封存SHA，所有响应/查询/KV策略匹配旧串行查询参考。
greedy延期270批，CPU/GPU sampling各284批，均命中安全边界、紧急open-XML刷新
和延期期间取消。私有五草稿370步，零接受92、部分204、全接受74；batch/features
各缩小28次，query_end释放10次，query_cancel释放6次。整卡采样峰值
42,956,816,384B，无observer/foreign错误。这是合成状态检查，不是K128/G64正式
吞吐或完整长历史显存验收。128K状态检查接续，最终六组性能队列尚未开始。

01:43补充：128K也已通过，两个缩小分支实际覆盖，逐层/全logits均一致。
之后state_4096未能启动，原因是build02使用copyfile复制CLI和状态probe，
权限变为0664（原为0775），报`PermissionError(13)`；空worker.log与失败
manifest保留。build03新副本恢复执行位，CLI/core/target probe/state probe
四个内容哈希全部保持不变，无模型代码修改、无重编译。
`transient_memory_validation03`明确记录复用来源，重核两项parity的命令、环境、
exit code、分支覆盖、资源归属与artifact哈希，接续4K/128K状态回归；不重复
已通过且二进制完全相同的数值检查。`transient_memory_replay03`等待接续验证
全部通过，后续仍是四组刷新A/B及两个完整45请求。02失败队列未执行GPU重放。

`transient_memory_validation01`保留为失败：1K所有逐层残差和logits检查实际
通过，但探针在history prefill之后才调用`set_dflash_layers`，因此只命中
batch arena缩小，没有创建过大特征缓存，无法覆盖features缩小分支。验证器
按设计拒绝通过；依赖队列`transient_memory_replay01`随之停止，未执行GPU重放。

`transient_memory_build02`仅修改独立探针，将原有全层特征采集设置移动到history
prefill前；输入、位置、四个位置×宽度1–8检查与全部断言不变。CLI、core、
状态回归probe与build01哈希一致，没有更改模型执行实现、没有放松验证器。
`transient_memory_validation02`已完成1K：batch/features缩小各一次，64层残差
和全词表logits串行/批量位级一致。128K检查正在执行，之后是4K/128K状态回归。

`transient_memory_replay02`已排队，要求上述四项检查全部通过并重新核对原始
证据才开始。先执行两个种子的回收关/开真实刷新对照（第二种子反转），再执行
最终组合的两个完整45请求队列。所有阶段验证输入、输出、查询、KV选择和边界
与各自冻结对照一致，保持短输出与截断请求；显存完整观察，单独判定48GB。
完整45请求对照使用此前已封存的新attention两组，不能冒充新鲜的单项回收A/B。
两个新服务均设五分钟检查，不抢占其他GPU作业；开发executor仍未接入该候选。

### 9月27日01:26 私有查询五草稿两种子通过，结束有限吞吐筛选

`guided_query_width_pilot01/02`全部结束，独立报告`guided_query_width_report01`
重新校验封存哈希、wire输入、输出、查询、KV选择、边界和Q行计数。两臂均开启
FFN固定5120；仅私有查询七/五草稿不同，主请求仍七草稿，seed1001反转顺序。

|种子|历史/输出|七草稿native tok/s|五草稿native tok/s|查询生成秒：七→五|
|---|---|---:|---:|---:|
|1000|192451 / 648|45.8716|46.7596（+1.94%）|7.545959→7.220220|
|1001|192451 / 978|51.8319|52.5680（+1.42%）|7.394015→7.132252|

私有查询五草稿保留为候选，仍待4K/128K状态回归；不把局部收益加到无中途
刷新的完整45请求98–102 tok/s上。两种子五草稿事件整卡峰值47.7362/47.7299GB，
七草稿48.0718/48.0487GB；完整工作负载49.05GB仍未解决。两种子结果都来自
相同真实刷新前缀，不是多个独立任务的统计置信结论，也不是SWE执行成功率。

按有限候选筛选停止条件，当前不继续搜索主请求宽度或重复已失败kernel。
转入后置显存阶段：`transient_memory_build01`已启动CPU独立构建，实验开关默认
关闭。尝试在接受Q复制完后释放采集缓冲、在下一次小批forward将大prefill
arena及DFlash特征存储缩到8行；下一轮prefill仍使用原2048 chunk。
不改变K/G、ctx1M、权重/KV精度、索引布局与检索策略。构建与验证脚本保留
同binary开关对照，GPU正确性、吞吐和显存仍待验证；未接入开发executor。
设计与限制见`dflash2_transient_memory_review_20260927.md`。

### 9月27日01:10 FFN模型对照完成，保留小幅收益；接续私有查询宽度

`swiglu_fixed5120_model01`已完成，独立报告重新校验构建、阶段封存与原始日志。
1K/128K四个位置、宽度1–8的64层残差和全词表logits串行/批量一致；三个
131–132K真实前缀两种子对照的输入、输出、查询、KV选择与边界均一致。

|种子|参考native tok/s|候选native tok/s|观察收益|verify ms/批|
|---|---:|---:|---:|---:|
|1000|107.9601|109.5928|+1.51%|32.1348→31.5674|
|1001|82.9095|83.8532|+1.14%|31.7553→31.3248|

第二种子反转开关顺序。四组整卡采样峰值约46.68GB；这三个冷前缀不发生
middecode刷新，也不替代完整45请求及其49.05GB峰值。GPU时钟/热状态未独立
控制，不宣称统计显著性或SWE成功率。报告`swiglu_fixed5120_model_report01`。
固定5120候选已接入`src/mmvq_q8.cu`，开关默认关闭，保留原通用kernel；
接入前后SHA和补丁保存在`swiglu_fixed5120_integration01`，没有替换运行服务。

已启动`guided_query_width_build01`隔离构建，基于通过模型检查的FFN core。
下一步仅比较私有查询七/五草稿，两个种子反转顺序，FFN设置两臂相同，主请求
仍为七草稿。`stable_prefill_main_width_hypothesis01.json`重核完整两组45请求
接受直方图：主请求缩到五草稿的回顾性产出损失约8–10%，比私有查询3–4%更大；
故先试私有查询，不能把回顾性计算当作实际吞吐。完成有限吞吐筛选后处理显存。

### 9月27日固定5120 SwiGLU有3.63%算子收益，已接续模型对照

`swiglu_fixed5120_experiment01`完成，91项linear检查、边界保护及memcheck均
通过（零错误）。六轮正反交替、378,798,080字节的轮换权重工作集下，中位
0.1458665→0.1405675 ms，耗时减少3.63%；寄存器108→157，stack/local为零。
原始日志、stage哈希及计时矩阵已独立复核，报告为
`benchmark/dflash2/swiglu_fixed5120_report01.md/json`。
算子结果不代表整请求也提升3.63%，按当前成本占比预期影响较小，须实际测量。

`swiglu_fixed5120_model01`已启动并链接独立CLI/core/target probe。runner PID
1793470，复用已检查的算子对象，未改`src/mmvq_q8.cu`或基线archive。接下来先
做1K/128K逐层及完整logits一致性，再做Quill/Effect/Dynamo三个约131–132K
冻结真实前缀的两种子四组对照，第二种子反转顺序。保持Q8_0/FP8、K128K/G64K、
C1、七草稿、T1，关闭同步profiling，要求配对响应/查询/KV策略及wire输入一致。
不包含工具执行或SWE评分；根据完整模型结果决定纳入或放弃，不以达到130
作为追加试验的理由。每五分钟检查使用`qw3-dflash-swiglu-fixed5120-model01-monitor.timer`。

私有查询五草稿的builder与pilot脚本均已准备，但未启动，等待较高优先级的
FFN模型结果。当前工作树的已集成优化仍只有先前验证通过的attention/私有查询
等改动，固定5120 kernel未纳入开发源码，更未替换运行服务。

### 9月27日集成验证完成：真实刷新收益复现

`single_request_integration_validation01`已完成，1K/128K各四个位置×宽度1–8的
逐层残差和完整logits一致；真实刷新pilot两个arm也通过。独立审计重新检查
全部stage artifact哈希，并确认两组响应分别复现隔离原型的相同响应，证据为
`benchmark/dflash2/single_request_integration_audit01.json`。

192451历史的长请求保持648个输出token、527个私有查询token，输入、输出、
查询文本、KV选择及刷新边界一致。串行查询生成13.423605秒，投机7.645489秒；
长请求native decode **32.2597→45.5887 tok/s（+41.32%）**。
包含首个短请求时为34.7915→48.5074 tok/s，不能与长请求单独数据混用。
此结果复现原型的约+42%，不能加到无刷新的45请求98–102 tok/s上。
整卡采样峰值47,614,590,976→48,054,992,896字节；最长样本间隔约238ms，
是观测峰值，不是allocator硬上限。显存优化仍按用户要求后置。

开发代码已接入并通过上述集成验证，两个实验开关仍默认关闭、运行服务未替换。
固定5120的SwiGLU算子试验已接续CPU编译，尚未产生新的算子或模型性能结果。

### 9月27日00:24 kernel结果已复核，集成构建通过，下一项试固定K的SwiGLU

`stable_prefill_kernel_profile01`成功结束；128K逐层残差与完整logits一致，
资源观察无foreign或错误。独立检查所有artifact哈希后，报告保存为
`benchmark/dflash2/stable_prefill_kernel_report01.md/json`。
44批八行验证（含4批warmup）中，FFN gate/up+SwiGLU为9.9833 GPU ms/批，
FFN down为5.3802，输入融合投影合计4.7040，输出投影2.1949，LM head 0.9862。
gate/up每线程108个寄存器、无local memory；未采集硬件计数器，不能据此声称
已达到DRAM或SM上限。CUDA API与kernel耗时重叠，不相加为decode耗时。

据此排队`swiglu_fixed5120_experiment01`，在集成验证结束后，单独复制算子源码，
只为8行/reuse=1/cols=5120增加五次K步循环展开，保留DP4A、FP32累加与归约
次序。既有generic kernel保留。运行91项linear检查、memcheck，再做六轮正反
顺序、权重工作集超过两倍L2的配对计时。尚未编译或运行这个候选，不能声称
新收益；只有通过且有收益才考虑模型级验证。不在没有新依据时重试旧负结果。

`single_request_integration_build01`已完成四个translation unit的重编译、独立
CLI/probe链接和两项CPU策略检查；1K target parity已通过，128K正在执行，
随后自动做真实刷新配对pilot。构建存在NVCC生成代码/未用参数等warning，
记录原始build日志，未扩展修复无关代码。运行服务未替换。

已准备私有查询五草稿的独立构建脚本`build_guided_query_width.py`，但尚未启动；
先看优先级更高的SwiGLU试验，再决定是否测试。主请求始终保持七草稿。

### 9月27日00:17 私有查询4K/128K回归全部通过

`guided_dflash_regression01`已结束，两种历史长度各28请求、14组token/策略
配对，全部复现对应的串行查询对照。独立重解析与artifact哈希检查保存在
`benchmark/dflash2/guided_dflash_regression_audit01.json`。
128K组372次私有查询step的接受Q行计数正确，greedy/CPU sampling/GPU sampling
分别记录269/235/235个延期批次、1/3/3个接受安全边界，均覆盖紧急开放XML刷新
和取消。该组整卡采样峰值44,745,687,040字节，无foreign或资源观察错误。
4K组358次step中6次零草稿回退；两组均未覆盖uncaptured前导分隔符step，
未验证完整Q张量位级哈希，不扩大正确性结论。128K合成回归使用K64K/G32K，
不能替代K128K/G64K真实任务吞吐。

kernel profiling已获得准入并接续。Nsight Compute实际安装于
`/opt/nvidia/nsight-compute/2025.3.1/ncu`，但当前账号无相应capability，驱动
`RmProfilingAdminOnly=1`，非交互sudo不可用；未运行硬件计数器采集、未改变
权限设置。先用已排队的Nsight Systems kernel时间，不据此声称已确定SM或
DRAM饱和率。集成验证队列`single_request_integration_validation01`也已排队，
等待构建后做1K/128K target parity与同配置真实刷新pilot，每五分钟检查。

### 9月27日00:15 开发源码已接入，集成构建等待kernel采集结束

前述组合补丁已经应用到开发工作树的五个源码文件，记录见
`benchmark/dflash2/single_request_integration_review01/application.json`。
attention、私有查询投机开关仍默认关闭；保留已有text-scan与allocation trace
改动，没有替换运行服务或修改冻结测试构建。原有23:55的“尚未应用”是当时状态。

新增`build_single_request_integration.py`，冻结当前源码后排队为
`single_request_integration_build01`，等待kernel profile成功结束，持共享测试锁
重编译executor、backend及两个CUDA单元，再链接独立CLI和两个GPU probe。
使用未包含独立guided helper对象的build04 archive，避免与已并入executor的
成员定义重复。同时重跑延期刷新与plan-cache的既有CPU策略检查。构建通过
后仍需集成版本的GPU对照，不能把孤立原型的通过直接标为集成版本通过。

128K状态回归仍在运行，已经进入最后的GPU sampling组；4K已通过。新Nsight
kernel采集尚在等待，本节不声称新增FFN kernel或吞吐成绩。
130 tok/s仍仅作参考；显存继续观测，暂不优化，多并发尚未开始。

### 23:55 私有查询4K回归通过，排队kernel级剖析并准备组合补丁

`guided_dflash_regression01/history_4096`的28请求和14组token/策略配对已通过，
输出与query/selection策略均复现旧新attention、串行查询对照。358次私有查询
step包含6次零草稿串行回退，逐step接受Q计数无错误；未覆盖uncaptured前导
分隔符step，不能扩大覆盖说明。greedy及CPU/GPU sampling均覆盖延期投机、
接受安全边界、紧急开放XML刷新和取消。整卡采样峰值44,085,084,160字节，
资源错误和foreign为0。128K尚在执行（当时4/28），worker1766827存活。

`stable_prefill_kernel_profile01`已排队，runner1767564等待上述状态回归全部
通过。新脚本`run_stable_prefill_kernel_profile.py`只给复制的target probe加入
CUDA profiler API范围与PID握手；保持build04推理core不变。它在固定128K
teacher、5个draft feature层、2/4/6/8行的既有验证宽度试验中采集GPU kernel，
检查完整serial/batch数值与44批八行FFN调用覆盖，再导出Nsight SQLite。
这些是kernel诊断，包含范围内的prefix restore，不是DeepSWE吞吐。GPU采集
尚未开始。共享锁、空闲准入、实际target PID的NVML观察与五分钟监测已接入。

组合审阅补丁保存在`benchmark/dflash2/single_request_integration_review01/`，
包括attention两个CUDA文件与私有查询三个executor/backend文件，保留已有
工作树修改。`git apply --check`通过，补丁尚未应用、尚未构建集成后的程序。
patch SHA256：`3e1a05a13a9f3060bc32fbcf8bc2309165ee8852df39a291efe70970b365ecef`。
推理服务未替换，没有新增BF16 target支持，也没有开始并发开发。

### 23:47 两种子私有查询对照均通过，状态回归接续

`guided_dflash_pilot02`完成，同build、反转顺序的第二种子长请求保持978输出、
473查询token及相同查询/KV选择/刷新边界。串行查询12.060371→投机7.360858秒，
长请求decode23.478554→18.915623秒，**41.6550→51.7033 tok/s（+24.12%）**。
首个种子的+41.85%与第二个种子的+24.12%均已独立核对计划、构建、阶段及
artifact哈希；汇总见`benchmark/dflash2/guided_dflash_pilots_report01.md/json`。
不能将这两个真实刷新请求的收益直接加到无刷新45请求的98–102 tok/s上。

第二组候选194次私有验证，76次零接受、109次部分接受、9次全部接受；提交
473行Q。整卡峰值串行47,597,813,760、候选48,048,701,440字节，无foreign
GPU样本。现有证据检查查询文本/token计数、选择和状态行为，尚未独立测量
完整Q张量位级哈希，不扩大正确性结论。

状态回归`guided_dflash_regression01`已接续并完成独立probe的CPU编译/链接，
继续4K和128K检查；吞吐与显存仍不作为这组合成检查的验收成绩。
新宽度筛选记录`guided_dflash_width_hypothesis01.json`：回顾性截短到5草稿，
两个查询的产出约损失2.85%/3.59%，既有128K固定teacher验证成本降低7.34%。
这只支持后续候选筛选，短块会改变proposal，尚未进行短块查询吞吐实验。
没有启动新的Nsight采集或Q8 FFN kernel变体，主请求仍固定7草稿。

### 23:42 私有查询投机首组同输出提速，第二种子与状态回归接续

`guided_dflash_build01`完成隔离CPU编译和链接，默认关闭
`QW3_EXPERIMENTAL_DFLASH_GUIDED_QUERY`；主工作树推理源码和运行服务未替换。
`guided_dflash_pilot01`的串行/投机两组已完整结束，重新核对冻结输入、构建、
manifest、原始资源和输出。192451历史长请求的主输出648 token、查询527 token、
KV选择及刷新边界完全一致；1191历史的无查询请求也保持输出一致。

| seed1000长请求 | 串行查询 | 查询投机 |
| --- | ---: | ---: |
| 查询生成秒 | 13.426224 | 7.631391 |
| native decode秒 | 20.129235 | 14.190095 |
| decode tok/s | 32.19198 | 45.66566 |

同输出长请求观察收益**+41.85%**，查询生成1.7593倍。两请求加权decode为
34.7081→48.5859 tok/s，与长请求单独成绩分开报告。198次私有验证包括69次
零接受、123次部分接受、6次全部接受；提交527行Q，拒绝行计数回退检查通过。
整卡采样峰值47,612,493,824→48,054,992,896字节，foreign样本为0；按用户要求
仅记录显存。该pilot不替代更广泛状态回归，也不作SWE质量结论。

`guided_dflash_pilot02`使用同一build、seed1001，反转为投机/串行顺序。投机组
已完成并复现对应历史输出、473 token查询、KV选择和刷新边界，长请求为
51.7033 tok/s；同build串行组尚在等待空闲，不提前给出第二组配对倍率。

`guided_dflash_regression01`已排队，须第二种子pilot完成后才独立编译测试probe，
再做4K/128K各28请求的greedy、CPU/GPU采样、延期闭合、紧急刷新、取消及
取消后恢复检查，并与旧新attention/串行查询的对应输出和策略逐项比较。
这是合成正确性集合，128K用K64K/G32K，不作为K128K/G64K真实吞吐。
该队列尚未执行GPU。所有新队列延续共享锁、空闲检查和五分钟只读监测。
regression监测使用`qw3-dflash-guided-query-regression01-monitor02.timer`；首次
监测入口路径拼写已修正，未影响测试服务或GPU测量。

### 23:29 新路径profiling完成，按可改善空间调整下一轮顺序

`stable_prefill_verify_profile02`六请求完成，两种模式的三个插桩输出各自与
未插桩输出一致。独立复算报告为`stable_prefill_verify_profile_report02.md/json`。
以下为同步操作组墙钟时间（含主机与同步），不是纯GPU kernel时间或正式吞吐：

| 主要操作组 | 原路径 ms/批 | 新路径 ms/批 | 新路径占已测操作组 |
| --- | ---: | ---: | ---: |
| FFN | 16.219 | 16.248 | 47.37% |
| attention norm及其融合投影 | 5.712 | 5.742 | 16.74% |
| 分页attention | 17.430 | 5.197 | 15.15% |
| recurrent output | 1.986 | 1.972 | 5.75% |
| DeltaNet | 1.779 | 1.818 | 5.30% |

无刷新时下一步kernel检查重点是Q8 FFN与融合投影，而非继续沿用旧attention
热点排序。已试过的SwiGLU加载、occupancy和保留旧归约的INT8 MMA均是负结果，
需更细GPU时间线找到新的实现依据，不能把全部16.25ms当作可消除的开销。

结合真实刷新结果，先执行有明确串行成本和可复用批量验证机制的私有查询
投机pilot，再做Q8 FFN的kernel级诊断与有依据的候选，之后评估草稿长度。
这个顺序按可改善空间与证据调整，不按是否跨过130决定。查询原型首轮仅
启用中途刷新私有查询；初始检索保持原路径，待有需要且单独验证后再扩展。

`guided_dflash_build01`已开始隔离CPU编译。pilot已准备同build串行/投机两组、
seed1000、原两条Scriggo冻结输入；要求输出、查询文本和token数、KV选择、
刷新边界全部一致，并检查实际草稿验证与接受Q计数。尚未开始该pilot的GPU
测试，原型尚未通过模型正确性或取得吞吐结果。

### 23:22 真实刷新四组完成：查询成本抵消验证提速

refresh02四组全部结束，独立复核见`stable_prefill_refresh_audit02.json`、
`stable_prefill_refresh_report02.md`。四组均真实发生一次刷新、延期投机及接受
安全边界；默认关闭新attention的两组完整复现历史deferred-enabled控制组的
输出、查询、KV选择和边界。profiling已接续启动，未改变冻结运行依赖。

| 192451历史的长请求 | 原路径 | 新attention |
| --- | ---: | ---: |
| seed1000 decode tok/s | 35.62 | 32.54 |
| seed1001 decode tok/s | 43.77 | 41.55 |
| seed1000 verify ms/批 | 43.940 | 32.015 |
| seed1001 verify ms/批 | 44.168 | 31.837 |
| seed1000 查询token / 生成秒 | 339 / 8.441 | 527 / 13.416 |
| seed1001 查询token / 生成秒 | 340 / 8.407 | 473 / 12.050 |

这是保留的负结果：验证更快，但新轨迹的检索查询更长。单个查询token约
24.7–25.5ms，不能把总查询时间增加全归因于执行变慢。候选查询生成分别占
长请求decode的67.4%、51.2%，总刷新占73.4%、56.4%；因此私有查询是
发生刷新时优先级很高的可改善路径，无刷新验证仍另看正在执行的操作组profile。
不能把两种请求集合混合成一个成绩，不能用缩小查询预算改变检索策略。

新增隔离原型`guided_dflash_step.cpp`与`build_guided_dflash_experiment.py`：
在私有查询中提出greedy草稿、批量target验证、裁掉拒绝的Q行，只提交接受的
target features，并沿用外层target/draft snapshot恢复。开关默认关闭，
主请求RNG不传入私有查询；尚未编译或运行GPU，不声称正确或提速。
构建器要求当前profile结束后才能编译，避免影响正式测量；只改新目录中的
源文件与archive，不改运行中的build04、工作树推理源码或服务。

### 23:12 完整两种子对照结束，进入真实刷新测试

`stable_prefill_full_replay02`四组45请求全部完成，独立报告与复算见
`benchmark/dflash2/stable_prefill_full_report02.md`、同名JSON以及
`stable_prefill_full_controls_audit02.json`。默认关闭新attention的两组各45项
输入、输出、私有查询及KV选择均完整复现对应历史控制组。

| seed | 原DFlash路径 tok/s | 新attention tok/s | 观察吞吐变化 | verify ms/批：原→新 |
| --- | ---: | ---: | ---: | ---: |
| 1000 | 78.91 | 98.37 | +24.65% | 37.4087→29.7768 |
| 1001 | 79.69 | 101.54 | +27.42% | 39.3085→30.3947 |

两组128–256K历史的verify分别下降28.32%和28.39%，到约31.79ms/批。
全部45项加权decode保留短输出与截断输出；不是无MTP基线，不是端到端
工具任务速度。候选verify仍占decode约81%，需要新路径profiling决定下一项。
短历史小于32K的吞吐分别下降2.00%和3.93%，verify成本增加约2%，保留负结果。

两组新旧输出仅5/45、6/45相同；私有查询各43/45、KV选择各42/45相同。
因此观察收益包含生成轨迹与权重变化，不能全部解释成固定输出的执行加速，
也不能声称SWE质量不变。两个候选均无中途刷新，须单独完成事件测试。
整卡采样峰值分别49,046,945,792和49,053,237,248字节；显存优化按用户要求后置。

full02服务正常退出，五分钟监测已自动停止。refresh02已接续执行，
reference_s1000完成、candidate_s1000实际使用GPU；profile02仍等待事件队列。
两候选均触发原队列的profiling调度条件，130无需作为验收门槛或修改冻结脚本。

### 22:58 新attention真实刷新复核工具

新增独立`benchmark/dflash2/audit_stable_prefill_refresh.py`，要求refresh02
四阶段全部结束，再复核冻结输入、构建、原始资源、计时、实际dispatch，并
比较逐请求生成、私有查询、KV选择与刷新边界。默认关闭新attention的两组
还必须复现旧deferred-enabled事件的输出及查询/选择/边界策略。

事件次数需与native计数一致；检查保留预算和紧急刷新保留完整开放片段。
同时明确报告各组是否真正发生刷新、延期投机及接受安全边界，缺失覆盖不能
作为通过。该事件集合保持独立，不混入45请求成绩，也不作固定输出因果结论。

CPU检查使用原四组真实事件的8请求（其中4请求发生刷新），并检查缺失边界、
紧急保留不足及未结束队列均被拒绝。没有启动额外GPU进程，现有冻结脚本未改。
待refresh02结束后执行该复核器；当前仍不能声称新attention真实刷新已通过。

### 22:51 第二个完整候选完成：101.54 tok/s，对应基线待完成

candidate_s1001的45请求已结束，plan/build/stage/artifact、输入wire、计数、
计时、实际dispatch及原始资源记录复核通过，见
`benchmark/dflash2/stable_prefill_full_candidate_s1001_audit02.json`。
29296输出token/288.522268秒，**101.5381 tok/s**。两个完整候选目前分别
为98.3678和101.5381 tok/s；第二个种子的同构建reference尚待完成，暂不
把旧历史成绩当作本次配对基线。

| seed1001候选历史桶 | decode tok/s | verify ms/批 |
| --- | ---: | ---: |
| <32K | 122.55 | 28.1775 |
| 32–64K | 91.53 | 28.1512 |
| 64–128K | 115.57 | 29.2438 |
| 128–256K | 98.34 | 31.7910 |
| >=256K | 91.57 | 32.0602 |

128–256K的verify与seed1000候选31.7945ms非常接近。全组平均verify
30.3947ms/批、占decode81.43%，draft6.0860ms/批；平均接受草稿2.7940、
输出3.7899/批。14个短输出和1个达到4096上限的输出均保留，无中途刷新。
76221条资源样本、foreign为0，整卡峰值49,053,237,248字节，最大样本间隔
253.15ms。该记录不构成第二组优化倍率或任务质量结论，刷新事件仍需后续测试。

### 22:50 剩余空间的源码核对

新增`docs/dflash2_remaining_optimization_review_20260926.md`，把首个完整
种子的verify81.14%、draft16.59%与源码可行性对应起来。源码确认现有
`QW3_GRAPH`不能直接用于分页KVMem批量验证；普通生成的CUDA profiler API
范围也不覆盖`generate_mtp`。这些是现有入口的适用范围检查，不是CUDA Graph
或Nsight新试验的结果。

私有查询可复用现有MTP的target checkpoint、查询Q批量采集/回退，以及
DFlash feature裁剪、外层draft snapshot恢复；还需接入private prompt与
greedy draft提交。本轮未新增这条实现，也未启动额外GPU采集。新路径
profiling完成后再按实际可节省成本确定试验顺序。第二个种子候选目前44/45，
同一worker1726358仍存活，对应的reference_s1001尚未开始。

### 22:38 首个完整45请求对照：78.91→98.37 tok/s

full02的seed1000新旧两个阶段均已完整结束。报告工具独立复核plan/build/
stage/artifact、实际attention dispatch、逐请求输入与计数、原始资源记录，
生成`stable_prefill_full_pair_s1000_report02.md`及JSON；仅包含这一个种子，
第二个种子尚未完成。没有从截断的运行拼接数据。

| 历史范围 | 旧decode tok/s | 新decode tok/s | 观察比值 |
| --- | ---: | ---: | ---: |
| 全部45请求 | 78.91 | 98.37 | 1.2465 |
| <32K | 130.30 | 127.69 | 0.9800 |
| 32–64K | 76.73 | 89.38 | 1.1648 |
| 64–128K | 84.14 | 119.38 | 1.4188 |
| 128–256K | 76.01 | 95.73 | 1.2593 |
| >=256K | 65.04 | 89.65 | 1.3784 |

旧组40588token/514.339秒；新组36131token/367.305秒。平均verify从
37.4087降到29.7768ms/批（约20.40%），128–256K从44.3576降到31.7945ms
（约28.32%）。draft约6.073→6.089ms/批。全组平均decode批次成本
44.3396→36.6975ms、产出3.4990→3.6099token/批；对应代数分解为成本
倍率1.20825×产出倍率1.03169=观察吞吐倍率1.24654。该分解不是固定输出
的因果归因，详细记录在`stable_prefill_full_pair_s1000_attribution02.json`。

45项输入wire相同，5项生成输出相同、43项私有查询相同、42项KV选择相同。
策略轨迹不同的三项分别是Effect197110历史、Dynamo198492和262280历史。
因此不能将全部+24.65%声称为严格相同生成轨迹下的kernel加速。15项短于
128token的输出、2项达到4096上限的输出全部保留（旧组11/5），无中途刷新。

本组最快单条是Quill68058历史、输出4096达到上限，156.05 tok/s；>=128K
且未截断的最快单条为Dynamo201026历史，3081token/20.5928秒，149.62 tok/s。
两者均不代替全组98.37。完整45请求重放不执行SWE工具，不是任务成功率。

整卡采样峰值旧48,931,602,432、新49,046,945,792字节，增加115,343,360字节。
新组84071条资源样本、foreign为0、最大间隔251.34ms；超过48GB继续只记录。
新的短历史桶约慢2%，说明该attention路径并非所有长度都获益。下一步仍是
第二个种子、真实刷新及profiling；按22:32用户指示，不将130作为最终门槛。
已完成首组低于130也意味着旧profile02调度条件会触发，不需要为该门槛修改
运行中的冻结脚本。当前全组verify仍占decode约81%，继续优先检查其剩余空间。

### 22:31 完整单种子对照报告入口

独立报告工具新增`--seed 1000/1001`，仅在对应新旧两个阶段都完整结束、
封存哈希和原始记录通过复核后，允许报告一个种子的全部45请求。默认仍要求
四阶段完成。保存读取时的root manifest快照，避免正在运行的第二个种子更新
状态后丢失报告来源；明确标记`all_four_arms_reported=false`。

已用完成的冷前缀四阶段验证原报告内容不变、单种子只包含对应完整两阶段，
并确认当前未完成的full02在默认、seed1000和seed1001模式下均被拒绝。
该修改只涉及独立CPU复算器，没有更改在跑队列的任何冻结依赖。

### 22:23 首组完整基线复现：45/45输入、输出与检索策略相同

full02的reference_s1000已完成，独立复算为40588输出token/514.339066秒，
**78.9129 tok/s**，与此前full_both的78.9197基本一致。
全部45项输入wire、生成响应（忽略服务分配ID）、私有查询及KV选择均相同。
plan/build/stage/artifact哈希、计时及原始98383条NVML记录已复核，见
`benchmark/dflash2/stable_prefill_full_reference_s1000_audit02.json`。

平均verify37.4087ms/批，占decode84.37%；128–256K桶76.0143 tok/s、
verify44.3576ms/批，>=256K桶65.0402 tok/s。11项短输出和5项达到4096上限
均保留，无中途刷新。整卡观测峰值48,931,602,432字节、无foreign进程；按
用户优先级不因超过48GB停止。该结果验证基线复现，不是新attention的完整
吞吐；队列已自动转向candidate_s1000。

### 22:22 后续优化空间：验证成本、刷新查询和草稿长度

根据完成的两组冷前缀数据重新计算，新attention每批总decode成本为
38.5846/38.5889ms，verify占82.60%/82.11%；每批输出为4.1986/3.2054，
其中接受草稿为3.2128/2.2085。下一步profiling应基于新路径定位attention、
Q8矩阵计算与状态操作的剩余成本，不能照搬旧路径的算子占比。

按当前每批成本，130 tok/s需要约5.02个输出/批；若保持目前两个seed各自的
产出，则批次成本分别需要降到32.30/24.66ms。这只是吞吐公式给出的条件，
不是新优化的实测结果或收益承诺。固定7草稿仍有缩短或自适应长度的试验空间，
但必须把减少验证工作与减少每批产出同时计入；不改正在执行的正式四组配置。

另一个独立瓶颈是KVMem的私有检索查询：源码中`prime_guided_mtp`仍在
`dflash`为真时返回false，中途查询循环因此使用`forward_one_token`。
旧真实刷新四组的generated_query为14.700/8.457/8.412/10.747秒。
这条路径尚未适配DFlash；需要私有提示feature、独立草稿状态、逐行查询采集
及外层状态恢复，不能仅删除条件判断。其收益主要体现在发生刷新的请求，
无刷新样本不能用于验证此方向。

当前先完成完整45请求、真实刷新及条件触发的profiling。新attention改变了
teacher数值及自由生成轨迹；在质量和完整吞吐证据补齐前保持实验开关默认关闭。
显存优化继续暂缓，单请求资格未完成前不进入多并发。

### 22:11 完整基线推进到25/45；报告复算工具已准备

full02的reference_s1000已完成25/45，worker1708428继续执行。22:10核验时，
已结束的24个请求的完整输入wire与生成响应均复现原
`deepswe_throughput_priority01/full_both`。这些只是完整基线的进度检查，
不据部分请求计算正式整体成绩；新attention的完整45请求尚未开始。

新增只读 `benchmark/dflash2/render_stable_prefill_throughput.py`，完成阶段
必须通过plan/build/stage哈希、请求/计时/预算、实际dispatch及原始NVML
复核后才输出Markdown和详细JSON。报告包含两个seed、历史分桶、草稿接受
比例与接受长度、draft/verify成本、零草稿串行、刷新和显存。短输出、达到
上限的输出全部保留，并逐请求比较输出/私有查询/KV选择；不声明SWE通过。

已用完成的冷启动四组生成 `stable_prefill_cold_report05_v2.md` 与JSON，全部
聚合值和资源统计与独立 `stable_prefill_cold_pairs_audit05.json` 一致。对仍
运行的full02调用收集器确认会拒绝，未生成部分请求的正式报告。该工具不
启动GPU进程，也未修改运行中队列的冻结代码。

### 22:04 三档数值诊断完成，完整45请求已通过前置检查

`stable_prefill_precision03` 已正常结束，三档原始日志重新解析、全artifact
哈希及NVML记录复核通过，见 `stable_prefill_precision_audit03.json`。
最终manifest SHA256为
`d654f9e5d0c004970e0c7864f354eb3e5392ab32b7e1633d4c1cdf9407368fe4`。

| 历史tokens | 固定teacher行数 | argmax不同 | 最大logit绝对差 | 平均KL(p‖q) | 最大total variation |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1024 | 32 | 0 | 0.316676 | 0.000279357 | 0.0257813 |
| 65536 | 32 | 0 | 1.026223 | 0.000414303 | 0.0450757 |
| 131072 | 32 | 0 | 0.434246 | 0.000184981 | 0.0310560 |

三个历史长度都实际命中page16、FP8 KV、width1新attention；切回旧路径后
同一teacher轨迹的全部logits精确恢复。三档资源样本2458/5100/8623条，均
无foreign GPU或observer错误。上述是相同输入下的数值变化与状态恢复诊断，
新旧分布并不相同；只用一个合成coding前缀、每档32个旧greedy teacher token，
不能推出SWE正确率、采样轨迹相同或质量非劣。尤其不能只凭argmax全相同
忽略已记录的自由生成长度/截断变化。

`stable_prefill_full_replay02` 已完成前置artifact复核与三个数值日志重解析，
128K八行target成本比0.699172也满足预先设置的<0.95筛选条件，目前进入
第一组reference_s1000的空闲观察/启动阶段。正式45请求四组、真实刷新对照、
条件触发的profiling继续顺序执行；仍未产生新完整45请求吞吐。precision03
的五分钟timer已正常结束，剩余测试继续监控，显存优化仍暂缓。

22:05复查：full02第一组reference_s1000已实际进入GPU执行，worker PID1708428
在NVML与/proc均确认存活，已完成3/45请求。refresh02、profile02进程也已
确认存活、继续等待前置；没有并发启动其他GPU性能任务。

### 21:58 两个seed的冷启动对照均完成：验证成本稳定下降约28%

`stable_prefill_tile_recovery05` 已正常结束，最终manifest SHA256为
`760260a3f6b18f43e55a638e0e1c19ef44875594b3ee64258801e365e996d8ca`。
四个重放阶段重新解析、输入wire/输出计数/所有artifact与原始资源记录复核
完成，证据为 `stable_prefill_cold_pairs_audit05.json`。

| seed | 旧→新加权decode tok/s | 旧→新verify ms/批 | 旧→新总decode ms/批 | 旧→新输出token/批 |
| --- | --- | --- | --- | --- |
| 1000 | 63.34→108.82 | 44.19→31.87 | 50.94→38.58 | 3.226→4.199 |
| 1001 | 57.06→83.07 | 43.91→31.68 | 50.80→38.59 | 2.899→3.205 |

seed1001旧组3293输出/57.7142秒、新组2122输出/25.5458秒。三项配对
分别为Quill53.04→72.74、Effect69.15→117.45、Dynamo59.98→86.33 tok/s。
所有六项配对的原始输入、私有查询与初始KV选择相同，生成结果均不同；
无中途刷新，不能用本测试替代真实刷新效果或SWE任务成功。

两seed的验证成本比为0.72126/0.72150，总decode成本约38.59ms/批，说明
执行成本下降在该范围内重复出现。不同seed每批产出仍差异明显，因此当前
83–109 tok/s的三前缀结果尚未稳定接近130，不能用Effect的单条142 tok/s
代表整体。下一步仍是完整45请求、多历史长度的正式统计，再依据新的同步
profiling决定后续优化，暂不进入并发。

四阶段均无foreign GPU；新seed1001整卡采样峰值46,677,164,032字节，
原始样本19893条。各阶段最大采样间隔约244–250ms，峰值不当作allocator
硬上限；当前显存仍只记录。恢复队列的五分钟timer已正常结束。
`stable_prefill_precision03` 已自动完成CPU编译，正在等待空闲观察后执行
1K/64K/128K数值比较；full02、refresh02、profile02进程均继续等待前置。

### 21:49 首组三个真实冷前缀：63.34→108.82 tok/s；修复收尾覆盖检查

build04、Q8_0+FP8 KV、K128K/G64K、T1、seed1000的三个固定冷启动长前缀
已完成。输入wire相同；私有查询及初始KV选择相同；三个生成输出均不同。
结果审计为 `stable_prefill_cold_pair_s1000_audit04.json`：

| 前缀 | 旧decode tok/s | 新decode tok/s | 旧→新输出token | 旧→新verify ms/批 |
| --- | ---: | ---: | --- | --- |
| Quill，131310历史 | 72.12 | 107.74 | 146→164 | 44.00→31.45 |
| Effect，132075历史 | 58.86 | 142.31 | 539→233 | 44.17→31.67 |
| Dynamo，132233历史 | 71.68 | 85.49 | 199→195 | 44.39→32.31 |

加权decode为 **884/13.9571=63.34 → 592/5.44043=108.82 tok/s**。
每批验证44.1898→31.8723ms，减少27.87%；每批总decode50.9384→38.5846ms。
每批输出同时由3.2263变为4.1986，尤其Effect接受草稿2.0→4.4419/批，故整体
倍率1.718包含生成轨迹和接受长度变化，不能全部归为固定输出的执行加速。
两个阶段均无实际中途刷新，也没有短于128token或耗尽4096预算的输出。
新路径原始资源17911条、无foreign GPU，整卡采样峰值46,672,969,728字节；
最大采样间隔247.48ms，仍只是观测峰值。不是完整45请求或任务质量分数。

`stable_prefill_tile_validation04` 在这组请求全部结束后收尾失败：旧
`check_dispatch`错误要求真实重放必须同时出现width1和width8，而三个请求
实际只有width8和终止width6，零草稿串行步、bridge步均为0。141次验证共
1126行；没有走单行路径，并非新attention未命中或CUDA执行失败。原失败
manifest、其余被前置失败停止的队列及原始数据全部保留。

修复仅在测试脚本 `run_stable_prefill_model.py`：真实重放要求观测到width8；
完整target及恢复审计仍严格要求width1–8，独立precision仍要求width1。
真实日志[6,8]正例、缺width8/错误page/关闭却命中/全宽度缺width1等反例均已
执行检查。没有更改GPU二进制、数值通过条件或生成请求。

新增 `resume_stable_prefill_tile_model.py`，`stable_prefill_recovery_audit05`
已重新解析并核验11个已完成阶段、artifact及资源记录，通过CPU复用审计。
正式恢复目录为 **`stable_prefill_tile_recovery05`**：复制保留已验证数据，仅
在新目录补齐candidate_s1000的posthoc summary，继续candidate/reference
seed1001两组。plan SHA256为
`57043677d10eed91781c7c61a691f670ffa8a84d53bb8ad9dea6fda80108773b`。
之后的实际新队列依次为 **precision03 → full_replay02 → refresh_replay02 →
verify_profile02**，均已启动等待并建立五分钟检查timer。原precision02、
full01、refresh01、profile01已失败退出，不再称为正在排队。

### 21:35 新attention的128K刷新回归完成，开始真实长前缀对照

`stable_prefill_tile_validation04/deferred_131072` 已完成 **28请求、14组token
与策略对照**。greedy延期投机关/开、CPU/GPU sampling的输出、查询、刷新
位置与KV选择均通过配对检查。greedy_on共269个延期批次、1次接受刷新边界；
两种sampling各235个延期批次、3次接受刷新边界。紧急刷新、延期期间取消及
后续请求恢复覆盖通过；实际page16、验证行宽1–8均命中新attention。

独立重跑原分析器、核验全部artifact和 **146,964条原始NVML记录** 后，审计
保存为 `stable_prefill_tile_deferred128k_audit04.json`。无其他GPU作业或监测
错误，采样最大间隔46.56ms；整卡峰值 **43,577,573,376字节**。本机制回归
使用128K合成历史、K64K/G32K和缩短的刷新阈值，不替代K128K/G64K正式吞吐。
其stage manifest SHA256为
`24ab93ee5457665343915e84cdedc02acb2c51f36dc0c7d7c1523aeb87012036`。

同输出合成用例可确认延期投机消除了大量串行尾段：两段代码围栏686 token，
零草稿步611→1，native decode 16.203→4.451s；XML562 token，零草稿步
550→1，decode 14.635→3.985s。上述仅说明该执行路径的效果，不算DeepSWE
吞吐，也不等于两种attention的加速对照。

与旧build10比较只有20/28输出完全相同，见
`stable_prefill_tile_output_drift128k04.json`。采样two-fences从532增长到768
上限，CPU/GPU两条记录均截断；取消后请求的输出从663缩短到161 token。
后者旧、新版本都未遵守“只返回XML”的提示，新版本改成更短的解释与建议。
因此恢复对照通过只支持执行状态检查，不支持模型任务质量通过的结论。

回归进程正常结束后，测试链已自动启动 `reference_s1000`，server PID1692294、
端口40619，执行三个原冻结真实长前缀的第一组冷启动对照；之后仍为candidate
seed1000、candidate seed1001、reference seed1001，再接precision与完整45
请求。尚无该候选的完整DeepSWE吞吐结论。

另已生成 `stable_prefill_integration_review04/candidate.patch`（两个CUDA文件、
214行diff），`git apply --check`通过，保留当前allocation trace和已有text-scan
改动。补丁仅供审阅，未应用到正式源码；其SHA256为
`75ebb7d095d3ea662b507a8a750692a23b22e62c38246e758fe7d51f3375e985`。

### 21:26 为吞吐未达标后的瓶颈定位准备独立profiling

128K刷新回归已完成17/28请求，worker1677372仍在执行，已完成记录的输出
与各自参考一致，尚未结束。新增
`benchmark/dflash2/run_stable_prefill_verify_profile.py`，使用相同build04，在
完整45请求及真实刷新四组均结束后才运行；当前已由
`qw3-dflash-stable-prefill-profile01.service` 启动等待，未使用GPU。
其plan SHA256为
`382434d960603b1114c6063d8bbd8da66486035433f05f848b88d25c7ad12b83`。

若两个seed的candidate整体及128–256K、>=256K各桶都达到130 tok/s，则跳过
profiling；否则在原三个冻结冷启动长前缀上比较attention关/开两组的同步操作
耗时。该诊断必须逐组复现对应未插桩测试的完整输出，并核对trace行数、验证
batch数及LM-head调用数。报告每个操作的ms/验证批，不把同步插桩耗时合并
进正式吞吐，也不把不同生成轨迹间的操作差额声明为固定输出的因果加速。

CPU语法、前置构建/请求结构及旧完整profile的318验证行、40批计数校验通过。
18个冻结输入哈希已复核，显存仍仅观测；新队列也有独立五分钟检查timer。
这是准备完成并实际排队的诊断，不是已有新的GPU profiling结果。

### 21:20 吞吐队列正常运行；记录新旧输出长度与截断变化

当前GPU仍由 `stable_prefill_tile_validation04/deferred_131072` 独占，已完成
11/28请求，近期请求及日志持续推进，尚未发现错误。该长历史机制回归结束后，
依次执行三个冷启动真实前缀四组对照、precision02数值诊断、full01完整45请求
四组吞吐和refresh01真实刷新事件四组对照。四个独立五分钟检查timer均已启用；
其余三个GPU队列等待前置，不抢占当前作业。所有新队列仍只观测显存。

新增只读分析器 `benchmark/dflash2/analyze_attention_output_drift.py`，对已经
结束的4K回归重新核验artifact哈希、原始策略分析及candidate dispatch后，输出
`stable_prefill_tile_output_drift4k04.json`。与build10相比，28请求中18个输出
完全相同；greedy prose前113 token相同，之后末句措辞变化。更值得记录的是
采样two-fences从634增长到768 token上限，围栏标记从4个变为3个。CPU/GPU
sampling都出现这一结果，因此是两个机制对照记录、一个采样轨迹，不能计为
两个独立质量样本。旧回归0个输出耗尽预算，新回归2个记录耗尽预算。

这些变化不否定新候选内部的投机/串行与CPU/GPU采样一致性，但也不能用内部
一致性宣称新旧teacher的任务质量相同。完整45请求必须保留短输出和达到上限
的输出，并同时查看接受率及每批verify成本；生成轨迹变化后的总体tok/s只能
称为同工作负载吞吐比较。当前没有新候选的完整DeepSWE吞吐结论，最新完整
旧路径结果仍为约79 tok/s；128K验证组件耗时下降30.1%不直接替代该指标。

### 21:10 新attention下4K刷新回归通过；真实刷新对照已排队

`stable_prefill_tile_validation04/deferred_4096` 全部 **28请求、14组token与策略
对照**通过。greedy延期投机关/开、CPU/GPU sampling配对的文本、私有查询、
刷新位置及selection策略均一致；greedy_on与两种sampling均覆盖延期批次、
接受的刷新边界、emergency-open-fragment、延期期间取消和后续恢复。
其延期批数分别为270/284/284。实际新attention page16、宽度1–8命中。
重跑原始分析器、校验全部artifact并复核25381条资源记录后，证据保存于
`stable_prefill_tile_deferred4k_audit04.json`；无其他GPU作业或observer错误，
该合成回归整卡采样峰值 **42,967,302,144字节**。

仅说明执行机制的同输出合成用例（不是DeepSWE吞吐）：

| 用例 | 输出tokens | 零草稿串行步：关→开 | 串行秒：关→开 | native decode秒：关→开 |
| --- | ---: | --- | --- | --- |
| 两段代码围栏 | 722 | 647→1 | 15.088→0.023 | 16.489→4.402 |
| XML工具调用 | 562 | 550→1 | 12.759→0.024 | 15.035→4.777 |

与旧build10同名28请求比较，新attention只有 **18/28** 的token/text完全相同；
其中greedy的prose和after-private-query也有分歧。因此候选内部精确一致
不能代表新旧teacher生成完全一致，不能宣称任务质量已保持。后续precision
诊断仍只测数值变化，任务结果与质量还需独立审核。当前已进入128K刷新回归。

新增并冻结 `run_stable_prefill_refresh_replay.py`，Python语法和冻结输入配置
已检查；`qw3-dflash-stable-prefill-refresh01.service` /
`stable_prefill_refresh_replay01` 等待完整45请求测试成功结束后运行。
使用原Scriggo两个前缀（1191及192451token）、max8192、T1、原K128/G64，
两seed的attention关/开/开/关四组，**延期投机在两臂均开启**。默认attention
组必须复现旧 `deepswe_refresh_near_boundary_replay02/optimized_s1000,s1001`
的完整输出；这些旧artifact哈希已固定。它单独观察新attention在真实刷新时
的decode、私有查询、串行尾段和事件覆盖，不混入原45请求指标，不执行SWE
工具。若事件未自然发生则记为未覆盖；不同文本/查询的倍率仍需逐事件审核。
新队列有独立五分钟timer，当前未执行GPU；显存仍只记录。

### 21:04 完整target长历史验证耗时下降30.1%，进入刷新回归

`stable_prefill_tile_validation04` 的1K/128K、默认关闭/开启四项完整模型检查
全部通过。每项4个位置×行宽1–8的逐层残差、全vocab logits均精确一致。
128K候选确认实际page16及所有宽度命中，position131103的32-token边界拆分
也已触发。四项stage manifest、全部artifact、32项logit检查、计时解析与原始
NVML样本/峰值/最大采样间隔均已复核：无其他GPU进程、observer错误或显存
中止。完整证据为 `stable_prefill_tile_full_target_audit04.json`。

四轮轮换、每轮10次、五层实际DFlash feature采集、GPU logits与行checkpoint，
restore不计时，128K target的结果如下：

| 验证行数（草稿数） | 旧路径 ms | 候选 ms | 耗时下降 |
| --- | ---: | ---: | ---: |
| 2（1） | 26.9214 | 25.7118 | 4.49% |
| 4（3） | 31.8730 | 26.6322 | 16.44% |
| 6（5） | 37.4777 | 28.2631 | 24.59% |
| 8（7） | 43.6253 | 30.5016 | **30.08%** |

1K八行26.5381→27.2132ms，慢2.54%。上述是完整target组件计时，尚无drafter、
采样、真实KVMem重放吞吐。两模式各自构建串行teacher，模式内固定teacher
比较宽度；跨模式teacher token身份未记录，不能称为同一完整生成轨迹的纯
执行A/B，也不能直接据此声称130 tok/s。新旧attention的数值漂移仍由后续
独立precision诊断观察。128K候选整卡采样峰值38,395,510,784字节，只代表
此dense probe，不替代完整45请求显存峰值。

新成本也改变了短草稿筛选条件。`stable_prefill_width_screening04.json` 用旧
接受长度直方图和不变nonverify成本作回顾性估算：128–256K的5草稿需要总
批次成本比低于0.9009，但新算子估算约0.9402，暂不优先；>=256K边界优势
只有约0.3个百分点，不能作实测收益。该分析显式保留“旧teacher接受率”、
“dense128K成本迁移”和“非验证成本不变”的限制，新候选接受率尚未测出。
因此先保持7草稿完成真实长历史测试。

当前GPU worker已启动 `deferred_4096`，之后执行128K刷新回归，再做三项
冷启动长前缀四组对照。precision02、完整45请求full01仍等待前置，未抢占
GPU。吞吐优先与每五分钟检查继续，显存优化暂缓。

### 20:58 1K完整模型边界修复已通过，继续128K

`stable_prefill_tile_validation04` 已完成115项边界、cache关/开各11项memcheck
（均零错误），及1K完整模型默认关闭/开启两组检查。开启组确认实际page16
及行宽1–8命中；position1055的实际split日志为batch2、first_rows1、tile32。
四个位置1047/1055/1063/1071、行宽1–8的逐层残差与全vocab logits全部精确
一致，原先0.385712的差异已在该回归消除。开启与关闭组分别建立自己的串行
teacher，不表示新旧attention互相数值相同。

证据为 `stable_prefill_tile_operator_audit04.json` 和
`stable_prefill_tile_target1k_audit04.json`，已复核阶段hash、完整32项logit检查、
width计时解析与资源记录。1K八行target计时26.5381→27.2132ms，约慢2.5%，
并非优化目标的长历史收益，也不是任务吞吐。当前接续128K候选/对照；
随后仍需刷新状态回归、三项冷前缀、数值诊断及45请求完整重放。
五分钟检查第2次正常，服务持续运行；显存只观测，不因48GB目标中止。

### 20:53 修复候选115项边界通过；完整重放后续队列已冻结

`stable_prefill_tile_build04` 构建成功，manifest SHA256为
`3a9cbce118cdcc8bf3733c3e221ff5de2310c8fa26b6ac36e6d4edc227245523`。
`stable_prefill_tile_validation04/tile_boundary` 的115项全部通过，实际tile split
日志已命中，artifact和15479条资源样本已复核，独立记录为
`stable_prefill_tile_boundary_audit04.json`。目前继续cache关/开的memcheck，
尚未获得修复后的完整模型一致性或吞吐结论。

`stable_prefill_precision02` 已排队，在该模型检查链成功后比较同一teacher输入
的旧/新logits、KL、TV与argmax，并要求恢复后旧路径精确复现。它只诊断数值
变化，不能代表任务质量不劣。

新增 `run_stable_prefill_full_replay.py` 已通过Python语法检查，并启动
`qw3-dflash-stable-prefill-full01.service` / `stable_prefill_full_replay01` 等待前置。
冻结原45请求、两seed与off/on/on/off四组顺序；每组新服务完整重放，不拼接
旧的部分结果。旧build10两seed的完整默认输出和artifact哈希也已冻结。
只有模型/刷新回归与精度诊断均完成、128K八行target成本下降超过5%，才启动
GPU重放；候选默认关闭组必须复现旧完整输出。不同候选采样输出分别保留，
不将工作负载吞吐比冒充相同输出的纯执行加速；不执行SWE工具或评分。
这两条后续队列也有独立五分钟timer，当前均未执行GPU。
所有队列无48GB中止条件，后续显存优化仍暂缓。

### 20:47 完整模型发现新边界差异，修复实验已构建中

`stable_prefill_model_validation02` 已真正执行新 attention：日志确认 page16、
24 Q/4 KV heads、head_dim256、FP8，行宽1–8全部命中。默认关闭组通过，
但开启组在 position1055、宽度2–8出现串行/批量差异：首个不同残差层为39，
全vocab logit最大绝对差 **0.385712**，本组argmax暂未改变。其余三个位置
1047/1063/1071及宽度1精确一致。这是候选的正确性失败，不能以argmax相同
放宽通过标准，也不能使用失败前输出的部分width计时作有效模型性能结果。
失败artifact、32项logit检查与资源记录已复核于
`stable_prefill_model_failure_audit02.json`。整卡峰值34.06GB，无资源错误；
停止原因是数值检查失败，并非48GB限制。依赖队列 `stable_prefill_precision01`
因此正常拒绝启动，尚未编译或执行GPU精度比较。

初步定位为32-token KV tile边界：首行有效KV长度1056，第二行1057。
固定1024-token split-KV和Q tile大小还不足以固定尾部masked循环次数。
新增隔离候选 `build_stable_prefill_tile_model.py`，仅在跨32-token边界时
把attention分两段执行；使用借用tensor view及原backend独立pinned metadata
槽位，不修改权重、KV精度、上下文窗口或因果范围。该修复目前仍是假设，
需要完整模型检查确认，未修改生产源码或运行服务。

`qw3-dflash-stable-prefill-tile04.service` / `stable_prefill_tile_validation04`
已启动CPU构建 `stable_prefill_tile_build04`。顺序为新增115项边界矩阵，
cache关/开各11项memcheck，再做1K/128K完整target检查、4K/128K刷新回归及
原三项真实长前缀四组重放。新增1055及长历史对应32-token边界，并要求实际
split trace；原来的严苛精确一致标准不变。独立五分钟timer已启动。
此前tile03初始化检测出补丁锚点拼写错误，已由本任务停止，未编译或执行GPU；
其失败目录保留，修正并通过CPU锚点/拆分覆盖检查后才启动tile04。

实际page16的算子四阶段已全部通过，证据为 `prefill_page16_operator_audit01.json`：
75项边界、cache关/开各8项memcheck且零错误，六轮轮换计时。八行attention
在64K为 **0.5943→0.1270 ms**、128K为 **1.0641→0.2604 ms**；单行反而
慢10.6%/26.4%，1K八行慢约156%。这仅说明长历史批量attention有优化空间；
上述完整模型失败说明算子矩阵不足以代替模型验证，真实任务吞吐仍以原45请求
**78.63→78.92 tok/s** 为最近有效配对结果。显存仍只观测，暂不优化。

### 20:31 实际16-token页的75项边界已通过

`verify_operator_followup02/prefill_boundary` 全部75项通过，包括新增的首个
16-token页边界；冻结source、artifact与原始NVML已复核于
`prefill_page16_boundary_audit01.json`。当前继续cache关/开的memcheck与计时。

`stable_prefill_model_validation02` / `qw3-dflash-stable-prefill-model02.service`
已排队，使用build04算子源码和新增实际dispatch检查，等待本轮算子全部结束
后再构建 `stable_prefill_model_build02`。两条队列各有五分钟timer；没有恢复
显存上限或重跑已确认更慢的Q8候选。完整模型候选有效性与吞吐仍未验证。

### 20:27 修正完整模型测试未命中新路径的问题

**撤销把 `stable_prefill_model_validation01` 的1K/128K通过计作新候选验证。**
它们虽然设置了开关，但实际KV页大小是16，而build01入口只匹配32，执行了
原decode；128K八行仍43.60255ms与原值接近，促使追查执行路径。
已经主动停止这条测试链，保留原manifest/log以及
`stable_prefill_model_validation01/coverage_invalidation.json`。
停止原因是确认的覆盖缺陷，不是CUDA/数值失败或用户取消；后续刷新回归和
真实请求对照尚未运行，不能引用为新候选通过或无性能收益。

直接读取GGUF元数据并核对executor配置：target为24 Q heads/4 KV heads、
head_dim256、hidden5120；`KvPageTable::configure` 默认KV页16。
**KVMem的block_tokens32不等于物理KV页大小。** 证据为
`target_attention_shape_audit01.json`。之前page32的算子正确性和计时仍有效，
但不能据此推断默认page16的完整模型表现。

新增build04固定prefill调度支持页16/32，以 `1024 / page_size` 定义分块
页数，保持1024-token分块不变。`verify_operator_followup02` /
`qw3-dflash-verify-page16-02.service` 正在构建，随后验证实际page16的75项
边界、cache关/开memcheck各8项和同一计时矩阵。已有负收益Q8候选不重复跑。

完整模型重试的builder/runner已补上实际执行证据：候选backend首次命中
每个行数时记录page/head/dtype；target probe输出实际KV页大小；候选检查
必须看到页16的1–8行实际命中，刷新和真实重放必须看到1/8行命中，关闭组
不得命中新入口。两组显式固定 `QW3_PAGED_KV_PAGE_SIZE=16`，与原默认一致。
如果只设置环境变量而没进入候选，不再允许通过。

### 20:16 attention 算子有长历史收益，Q8 FFN 候选更慢

`verify_operator_followup01` 十项阶段全部完成；artifact、解析结果与原始
NVML 计数/峰值/采样间隔已复核于 `verify_operator_followup_audit01.json`。
attention 的cache关/开 memcheck均为0错误。memcheck的CUDA进程是sanitizer
子进程，由runner实时检查所属进程组；manifest中的launcher显存0不代表
子进程未用显存，资源比较使用整卡峰值。

六轮交替、两对KV轮换，主机enqueue加最终同步的单次attention中位耗时：

| 历史 / 行数 | 原decode ms | 固定prefill ms | 候选/原耗时 |
| --- | ---: | ---: | ---: |
| 1K / 1 | 0.02688 | 0.07393 | 2.751 |
| 1K / 8 | 0.02888 | 0.07405 | 2.564 |
| 64K / 1 | 0.11349 | 0.12529 | 1.104 |
| 64K / 8 | 0.58650 | 0.12654 | 0.216 |
| 128K / 1 | 0.20312 | 0.25765 | 1.268 |
| 128K / 8 | 1.04948 | 0.26017 | 0.248 |

长历史批量验证有明显算子收益，单token和短历史路径更慢，不能直接宣称
完整模型提速四倍或达到130 tok/s。两对KV在64K/128K超过L2，1K工作集仍可
驻留缓存；这不是完整模型逐层轮换权重/KV的测量。

Q8 SwiGLU mode0/1/2各91项linear检查通过，mode1/2 memcheck均0错误。
378,798,080-byte权重工作集超过两倍L2，六轮轮换中位为0.146003、0.156064、
0.154377 ms。两个新载入方式分别**慢6.89%/5.74%**，不采用；生产
`src/mmvq_q8.cu`未因此修改。较少静态load指令/寄存器不是性能收益。

已构建 `stable_prefill_model_build01`，只在隔离源码中添加默认关闭的
`QW3_EXPERIMENTAL_FP8_STABLE_PREFILL`，让FP8、GQA6、256维、页32、1–8行的
串行/验证统一使用已测固定prefill数学；同一个开关控制tile/chunk/workspace
规划，并加入plan-cache key，拒绝未验证的CUDA graph capture。
它改变attention内部Q/O舍入（FP16），没有修改Q8_0权重或FP8 KV；与旧的
FP32-Q decode不承诺逐位相同，需做模型一致性及输出质量检查后才能采用。

`qw3-dflash-stable-prefill-model01.service` / `stable_prefill_model_validation01`
已在前序全部结束后CPU编译成功，接续1K/128K target逐层残差和全vocab logits
串行/批量检查与width计时，再做4K/128K刷新/取消/CPU-GPU sampling回归，
通过后执行原来三项冷启动长前缀的两seed off/on/on/off对照。默认关闭组必须
复现build10原始输出。该队列仍不是原45请求暖前缀正式测量，显存无48GB
终止限制，有独立五分钟检查timer。服务运行配置未替换。

另已准备 `target_attention_precision_probe.cpp`，用于固定teacher输入测量
旧/新attention的logit、KL、TV及argmax差异，并在恢复原路径后复查原输出；
尚未编译或GPU执行。该诊断不等于SWE或公开任务质量评分。

### 20:01 固定 prefill 扩展边界检查 65/65 通过

build03 编译成功。`verify_operator_followup01/prefill_boundary` 已通过全部
65 个形状：包括实际行数1/2/4/6/8、短历史CPU oracle、页边界、1024-token
固定分块边界、64K/128K历史及原来的decode partition变化附近。
每种形状三次重复，全部 finite、输出guard完好且串行/批量精确一致。
artifact 和11478条NVML记录已复核，无其他GPU进程或observer错误。
目前继续执行 cache关/开的 memcheck，之后才进行attention计时及Q8 FFN候选
检查；尚未进行完整模型的新算子集成，不把这项正确性通过算作吞吐改善。

### 19:58 四组真实刷新完成；收益仍需区分执行路径与采样变化

`deepswe_refresh_near_boundary_replay02` 四组全部正常完成，输入/环境/输出计数、
artifact 和全部原始 NVML 样本已在 `deepswe_refresh_near_boundary_audit02.json`
复核。该审计包含 middecode 私有查询，替代 audit01 不完整的查询比较。

仅列同一 **192451-token 输入**的长请求（初始1191-token请求未混入表）：

| seed | 延期投机 | 输出 token | native decode tok/s | 零草稿步 / 秒 | 刷新秒 | 延期批数 |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| 1000 | 关 | 568 | 23.980 | 16 / 0.421 | 16.044 | 0 |
| 1000 | 开 | 792 | 35.727 | 2 / 0.053 | 9.340 | 2 |
| 1001 | 关 | 796 | 31.244 | 7 / 0.184 | 11.933 | 0 |
| 1001 | 开 | 1207 | 43.799 | 0 / 0 | 9.294 | 9 |

两组都命中真实延期投机与接受边界，零草稿执行减少；全部在 `boundary=safe`
刷新，recent pin 保持16384，K/G及采样设置未改。两组均有生成文本、私有查询、
刷新位置与KV选择分歧，不能将上表倍率称为同一输出的纯执行加速；也未通过
真实工具循环验证任务成功。全部四组无其他GPU进程、监测错误或显存终止，
整卡采样峰值最大 **47,581,036,544 字节**。这不覆盖原45请求的48.93GB峰值。

新增明确的框架限制：`prime_guided_mtp` 在 `dflash` 非空时直接返回false，
因此私有检索查询仍逐token执行；相关主循环调用 `forward_one_token`。
本组私有查询生成约8.4–14.7秒，查询与重选总计占长请求native decode的
**33.7%–67.7%**。这些不是草稿接受率造成的损失；单独优化普通 verifier
也不会自动消除这部分开销。尚未修改私有查询执行和状态恢复路径，先完成
当前 attention/FFN 候选测试，再根据收益决定后续工作。

`verify_operator_followup01` 已在上述四组结束后开始 CPU 构建 build03，
当前正式主模型配置仍为 build10，不宣称新增内核已有真实请求提速。

### 19:54 固定 prefill 首轮通过；真实刷新已获得覆盖

`prefill_fixed_schedule_validation02` 已正常结束。新编译 control 复现原来的
15 项中 10 项不一致；固定八行规划、1024-token KV 分块的候选 **15/15 全部
bit-exact**。原始日志、artifact 和 NVML 已独立复核，见
`prefill_fixed_first_matrix_audit01.json`。这只能证明已测形状的 prefill
自身串行/批量一致，不等于与 FP32-query decode 数值相同或全模型通过。

已准备 `prefill_fixed_schedule_build03`：修正单个 KV 分块无需 float scratch
时被提前拒绝的问题，并增加 `prefill_fixed_boundary_probe.cu`，覆盖短历史、
页边界、1024-token 分块边界及 64K/128K 历史，共 65 个形状。短历史另与
FP64 CPU oracle 对照；之后才运行 cache 开/关的 memcheck 和两对 KV 轮换
的 attention 计时。计时包含主机 enqueue 和最终同步，不是纯 kernel 计时，
更不是 DeepSWE tok/s。

`qw3-dflash-verify-followup01.service` / `verify_operator_followup01` 已排队，
等待 Scriggo 四组对照结束后，取得同一实验锁，再进行 CPU 编译与 GPU 检查，
避免编译干扰当前真实请求计时。该队列也会检查独立的 Q8 SwiGLU 对齐载入候选：
mode0/1/2 各 91 个 linear case、mode1/2 memcheck、六轮交替的超 L2 权重工作集
计时。尚无模型路径或算子性能收益结论；两项候选均未进入生产 dispatch。
队列有独立的五分钟只读检查 timer，显存仍只观测。

Scriggo seed1000 已完成 off/on 两组；192451-token 输入确实在生成中刷新。
off/on 分别生成 568/792 token，native decode 为 23.980/35.727 tok/s，
零草稿步数 **16 → 2**，其耗时 **0.421 → 0.053 秒**；开启组观察到两批
延期投机、一个接受边界。这证明第一项优化已在真实输入上执行到，不能把
两次吞吐之比直接当作同一生成轨迹的加速比：采样继续文本不同，安全刷新点
分别为 decoded76/68，私有查询长596/339 token，刷新耗时16.044/9.340秒，
后续 KV 选择也不同。seed1001 的反向顺序对照仍在执行。

`deepswe_refresh_near_boundary_audit01.json` 已复核首两组输入、artifact 和原始
资源记录。该版的 `same_private_queries` 只采集了原 prefill 查询日志，
没有包含 middecode 查询，空列表相等不能用于判定查询相同；审计器已补齐
middecode 查询采集，待四组结束后写新审计文件，保留旧文件说明缺陷。
该 Scriggo 集合仅用于刷新事件覆盖，不并入原 45 请求分数，也不算 SWE 成功。

### 19:39 128K 验证成本已实测，prefill 原路径不能直接接入

`target_verify_width_diagnostic01` 正常结束，1K/128K 两组每组四轮交替测量，
全部 serial/verifier residual 与完整 logits 一致。artifact 和原始资源已复核，
见 `deepswe_verify_width_audit01.json`。

| 验证行数（草稿数） | 1K 验证 ms | 128K 验证 ms | 128K 相对八行成本 |
| --- | ---: | ---: | ---: |
| 2（1） | 22.007 | 26.920 | 0.6168 |
| 4（3） | 22.881 | 31.901 | 0.7310 |
| 6（5） | 24.375 | 37.477 | 0.8587 |
| 8（7） | 26.528 | 43.642 | 1.0000 |

真实 128K–256K 七草稿基线每批平均 51.41 ms，其中非验证约 6.92 ms。
若仅暂作筛选、假设 dense 成本比可迁移且其余开销不变，三草稿的整批成本比
约 0.767，高于接受长度要求的 0.741，暂不值得默认缩至三草稿；五草稿约
0.878，低于要求的 0.901，可作后续小规模长历史对照。>=256K 区间三/五草稿
均有条件进入初筛。**这只是基于旧接受直方图的实验筛选，不是新配置吞吐预测**；
较短块会改变候选与采样轨迹，最终需重新测真实请求。

原 `prefill_verify_feasibility01` 的 15 形状中 10 项 self-parity 不通过，
恰好是全部 4/8 行形状；2 行五项通过。不是 CUDA 崩溃，无外部 GPU 干扰，
原始 4,562 条 NVML 记录与 artifact 已复核。因此现有 prefill 开关不能直接
用来替换当前 verifier。固定分块仍是待检验假设。

复核固定分块实验时发现：实际执行已计划 8 行/1024-token chunk，但容量查询
仍用旧的动态计划，可能导致部分形状静默回退到不分块算法。**尚未执行 GPU**
的 `prefill_fixed_schedule_validation01` 和其后继 Scriggo 队列已主动停止，
保留原始 KeyboardInterrupt 与 `interruption_reason.json`，不记作 GPU 数值
失败。build02 同步计算新计划的缓冲需求，保留原计划需求以兼容其他调用，
固定实验路径遇到不足不再静默回退。正在重新 CPU 构建，完成后用新 run ID
接续队列。运行中的 build14 文本扫描回归没有中断。

19:41 更新：build02 已 CPU 构建成功，GPU 队列改为
`prefill_fixed_schedule_validation02` / `qw3-dflash-prefill-fixed02.service`，
后继为 `deepswe_refresh_near_boundary_replay02` / `qw3-dflash-refresh-near02.service`。
两者均已确认进程存活，并安装各自五分钟监测。旧 01 没有执行 GPU，保留取消
原因，不替换原始记录。
当前 fixed 实验检查从 1K 历史开始，并会拒绝无 float scratch 的调用；集成到
完整模型前还必须处理单 KV chunk、无需 scratch 的短前缀，并补测跨 1024-token
分块边界和 memcheck。即使这 15 项通过，也不能直接打开模型 dispatch。

build14 的 `deferred_scan_shortcut_validation01` 已完成并独立审计：28 请求、
14 组内部 token/策略对照，以及全部 28 个请求与旧 build10 的 token 和
refresh/query/selection 对照均通过。三个开启变体均覆盖延期投机、接受安全
边界、紧急刷新和取消恢复。证据为 `deferred_scan_shortcut_audit01.json`；
这是文本扫描 shortcut 的正确性结论，尚没有该改动的真实任务提速数据。

### 19:29 原四组事件全部复核；验证宽度测试开始执行

`deepswe_throughput_priority01` 已完成三组 45 请求及四组 6 请求事件回放，
服务正常退出。全部事件结果的配置、输出 artifact 及原始 NVML 数据已复核于
`deepswe_refresh_event_audit02.json`。seed1000 为 93.6000 → 93.4544 tok/s，
seed1001 为 91.8612 → 92.2746 tok/s；每个 seed 的六对输出、私有查询和 KV
选择全部相同。四组均无 middecode refresh / deferred batches，第一项真实
收益仍未验收。部分算子 CPU 编译与事件末段重叠，这些微小差异不作因果结论。
补充 Scriggo 队列的执行条件因此已满足，但仍按既定顺序等待前面的 GPU 检查。

验证宽度诊断已在 GPU 执行完 1K 历史，serial/verifier 的 residual/logits
检查通过。2/4/6/8 行的验证中位耗时分别 **22.007 / 22.881 / 24.375 /
26.528 ms**。验证行数减半仅减少约 13.75% 的 target 时间；这不是完整
DFlash 批次耗时，也不能推广到尚在等待执行的 128K 诊断。

同时完成 Q8 SwiGLU 独立候选的 CPU 构建，尚未排入 GPU：
`q8_swiglu_load_experiment01` 使用相同 build10 core，只对宽度 8、reuse=1、
cols 可被 64 整除的 gate/up 权重读取增加 regular/streaming int32 模式，
保持原八元素点积与 FP32 归约顺序。工作树生产 `src/mmvq_q8.cu` 没有修改。
编译后 SASS 中 U16 load 静态指令 18 → 10，全部 global load 34 → 30；
该变化仅是静态证据，不证明实际加速。regular 版本报告 96 寄存器、零 stack/local，
同样不能把寄存器数视为性能结论。后续需要算子精确/guard 检查和大于 L2 的
轮换权重测量，再决定是否继续完整模型验证。

### 19:18 补充准备更接近刷新边界的真实请求

原事件组 seed1001 的 Effect 输出也未达到阈值：对应剩余 1285/312/150 token
的三个请求分别输出 1203/155/143 token。Dynamo 仍在运行，不提前判定整组
是否覆盖。为避免后续仍缺少第一项的真实事件，已对原有 32 个 trajectory
inventory 仅按历史输入长度筛选 `[192384,192512)` 的请求，未按新输出或吞吐
选取。唯一符合者为 `scriggo-method-declarations` 的历史 request 180。

`deepswe_refresh_near_boundary_requests01` 已完成 CPU 冻结与当前原生模板计数：
初始请求加 request 180 共 2 请求，后者仍为 192,451 token，距离原 K128/G64
刷新阈值 61 token。源为 `requestplan10_pier_mini_20260823_c`，不同于原 45
请求所用的 run；因此只作额外事件覆盖，不并入原吞吐分数。未合成 padding、
未缩小预算，仍自然 EOS / max8192。此准备没有排入 GPU，先审核现有四组事件。

19:21 补充：两个 seed 的优化臂均已结束，deferred_batches 都为 0；最后的
seed1001 参考臂仍运行。已把这两个 Scriggo 前缀的四组对照条件排队，服务
`qw3-dflash-refresh-near01.service` 等待当前验证宽度、prefill、scan-shortcut
及固定分块实验结束；还要求原四组事件完成且依然没有延期刷新覆盖，否则跳过。
使用原 build10、相同采样种子 1000/1001 和已固定的开关次序，显存只观察。
独立 `deepswe_refresh_near_boundary_replay01` 保留全部结果，每五分钟检查。

### 19:14 第一组真实刷新前缀对照完成，尚无延期刷新覆盖

`event_reference_s1000` / `event_optimized_s1000` 各完成 6 请求，均输出
2,857 token，加权 decode **93.6000 → 93.4544 tok/s**。逐请求 wire、输出、
私有查询、KV 选择及各自 artifact/原始 NVML 记录已复核，见
`benchmark/dflash2/deepswe_refresh_event_audit01.json`。

全部自由生成在各自刷新阈值前结束，middecode refresh 和 deferred_batches
均为 0。两边各有一个 zero_draft_step，不能把这个计数当成等待工具闭合的
串行损失。该六请求集合只用于事件覆盖，不合并到原 45 请求分数；**未覆盖
事件不算第一项真实收益验收通过**。seed1001 对照仍在运行。
固定分块的 CPU 编译与 seed1000 优化臂末段重叠，且本身没有事件；这里微小的
计时差异不作因果结论。

现有 FlashInfer prefill scheduler 在 GQA=6 时，为串行／2 行选择 CTA Q=16，
4／8 行选择 CTA Q=64，并动态选择 KV chunk。它们是待验证的数值差异来源，
尚未拿到原 prefill 的 GPU self-parity 结果。

已建立独立 `prefill_fixed_schedule_build01`，control 与 fixed 均重新编译
`flashinfer_prefill_adapter.cu`，以独立 object 链接相同 build10 core/probe。
fixed 仅针对 FP8、24 Q heads/4 KV heads、head_dim=256、page=32、实际
1–8 行，使用 8 行计划与 1024-token KV 分块。设备 Q 指针及实际输出行数
保持不变；多出的 merge 描述不参与输出。没有改生产源文件或 dispatch。
两份二进制已 CPU 编译成功，但尚无 GPU 数值或性能结论。

`prefill_fixed_schedule_validation01` 已排在 scan-shortcut 回归之后，用相同
15 形状检查新 control 与原 core 的诊断一致，再检查 fixed。纯数值失败会保留
并否决直接接入；即使通过，仍需额外分块边界、memcheck、完整模型一致性及
性能测量。服务 `qw3-dflash-prefill-fixed01.service` 与每五分钟监测已启动。

### 19:03 三组完整回放结束，开始真实刷新事件对照

双开 seed1001 完成 45/45：30,813 输出 token / 386.5412 秒 native decode
= **79.7147 tok/s**。输入顺序、wire 内容、实际 K/G 预算、artifact、token
记账及资源原始记录已独立复核，保存于
`benchmark/dflash2/deepswe_full_seed1001_audit01.json`。13 个短输出与 2 个达到
4096 上限的请求全部保留。该组没有同 seed 的双关参考，不能把它与 seed1000
相减报告优化收益；`full_throughput_summary.json` 的跨 seed 比值也仅是观察值。

| 历史长度 | 双开 seed1000 tok/s | 双开 seed1001 tok/s |
| --- | ---: | ---: |
| <32K | 130.07 | 127.51 |
| 32K–64K | 76.78 | 83.06 |
| 64K–128K | 84.07 | 96.15 |
| 128K–256K | 76.02 | 70.23 |
| >=256K | 65.35 | 71.16 |

seed1001 的 128K–256K 每批接受草稿约 2.607，verify 44.39 ms/批；同一
优化配置 seed1000 约 2.901、44.3 ms/批。接受长度随生成内容变化，而单批
验证仍昂贵。现有数据不支持“长历史已接近 130+”，也不足以将差距全部归于
草稿对任务适配不好。三组完整回放均没有 decode 中刷新，第一项优化仍需事件覆盖。

seed1001 整卡采样峰值 48,914,825,216 字节，85,793 个样本，最大采样间隙
258.5 ms，无外部 GPU 作业。显存只记录，没有因超过 48GB 中止。

### 19:02 文本扫描 shortcut 的 GPU 回归已排队

`deferred_scan_shortcut_validation01` 使用已完成 CPU 构建的 build14，
在当前完整回放、真实刷新事件、验证宽度及 prefill 可行性检查后执行。
仅补测受这次代码修改影响的 4K 生成文本回归：28 请求、14 组内部 token/策略
对照，并将全部 28 个输出及刷新、私有查询、KV 选择策略与旧 build10 逐项比较。
该测试是正确性回归，不计为真实任务吞吐；不重复未受影响的 128K 全部回归。
prefill 候选的纯数值不一致属于该独立候选的否决，不阻断文本扫描回归；
GPU 运行错误、外部作业干扰及监测失败仍需先检查。

服务为 `qw3-dflash-scan-shortcut01.service`，每五分钟监测为
`qw3-dflash-scan-shortcut01-monitor.timer`，已确认处于等待前序测试状态。
显存继续只观测。运行中的正式吞吐执行器、其依赖和模型二进制均未改动。

### 18:32 完整吞吐基线通过，优化组合正在重放

`deepswe_throughput_priority01/full_reference` 完成 **45/45 请求**，
40,588 输出 token / 516.2023668 秒 native decode = **78.6281 tok/s**。
输入序列化内容、实际 K128K/G64K 请求预算、native/HTTP token 计数及全部
artifact 哈希已独立复核，证据为
`benchmark/dflash2/deepswe_full_reference_throughput_audit01.json`。

| 历史长度 | 请求 | 输出 token | 加权 decode tok/s | 平均接受草稿 / 批 | 验证 ms / 批 |
| --- | ---: | ---: | ---: | ---: | ---: |
| <32K | 9 | 855 | 129.82 | 3.518 | 27.83 |
| 32K–64K | 9 | 12,457 | 76.42 | 2.007 | 32.37 |
| 64K–128K | 9 | 12,550 | 83.73 | 2.653 | 36.67 |
| 128K–256K | 15 | 13,579 | 75.81 | 2.901 | 44.49 |
| >=256K | 3 | 1,147 | 65.06 | 2.337 | 44.29 |

整体草稿接受比例 **35.77%**，每批输出 **3.499 token**，验证日志耗时占 decode
约 **84.41%**；每批草稿约 **6.08 ms**。45 请求均无 middecode refresh，
因此该完整基线不覆盖第一项优化，后续独立真实刷新事件集合仍然必需。
所有 11 个短输出与 5 个达到输出上限的请求均保留。没有执行工具，不宣称 SWE
任务通过；这些结果也不能直接与先前三个冷前缀的 63.3 tok/s 相减当作优化收益。

采样整卡峰值 **48,931,602,432 字节**，98,681 个采样，最大间隙约 251.8 ms，
无外部 GPU 作业或观察错误。按用户最新优先级，该数值不阻断吞吐对照，亦不
标为原 48GB 目标通过。两项优化同时开启的 seed1000 组已实际启动 GPU 重放。

为进一步定位验证开销，`target_batch_parity_probe.cpp` 新增可选
`--benchmark-widths --feature-layers 5,19,33,47,61`。默认正确性路径不变；新路径
先要求所有层 residual/logits 与串行逐值相同，再在同一 prefix 与 teacher token
上交替测量 2/4/6/8 行验证。使用实际五层 drafter feature 捕获、GPU logits 与
逐行 checkpoint，预热后重复，明确排除 prefix restore 耗时。
这是 **dense target 的固定 token 成本诊断，不是 KVMem/DeepSWE 吞吐**。

`target_verify_width_build01` 仅编译新 probe，链接完全相同的 build10 core，CPU
编译已通过。`qw3-dflash-verify-width01.service` 正在等待当前三组完整重放及四组
刷新事件全部结束，再持有同一 GPU 锁执行 1K/128K 诊断；尚未执行 GPU 测量。
其每五分钟检查为 `qw3-dflash-verify-width01-monitor.timer`，不抢占当前测试。

### 18:44 补充验证 kernel 的可行性检查

除了测 2/4/6/8 行的固定 token 验证成本，还准备了现有 FP8 Tensor Core
BatchPrefill 的逐 token／批量调用一致性检查。当前 BatchDecode 对各验证行分别
扫描前缀；复用已有 BatchPrefill 是候选方向，但它使用 FP16 Q 与不同分块计划，
不能仅凭批量调用更快就改变 verifier。必须先检查串行与批量使用该路径时是否
产生相同结果，再考虑完整模型检查。

`tests/flashinfer_decode_gqa.cu --prefill-serial-parity` 覆盖 FP8 KV、GQA=6、
head_dim=256、2/4/8 行，以及 1K、64K、128K 附近共 **15 个形状**，使用乱序
物理页和输出 guard。它比较 BatchPrefill 自己的串行与批量结果，不将 FP16 Q
的结果假称为旧 FP32 Q decode 的逐值等价；不一致则拒绝直接接入，保留失败。
尚未修改生产 dispatch，也没有新增 BF16 target 支持。

`prefill_verify_operator_build01` 仅编译该检查，仍链接原 build10 core，CPU 编译
已通过。`qw3-dflash-prefill-feasibility01.service` 等待 width 诊断完成之后执行，
串行依赖顺序为：完整吞吐三组 → 真实刷新事件四组 → 固定 token width 诊断 →
BatchPrefill 自一致性检查。每五分钟检查使用相应 `-monitor.timer`。
这项检查尚无 GPU 结果，不是性能收益或任务正确性通过证明。

已冻结的六请求刷新集合不保证自然生成跨过阈值：只读复查原轨迹发现，Effect
117/118/119 与 Dynamo 98 的历史输出分别为 541/128/101/2171 token，小于当轮
距初次刷新的 1285/312/150/2294 token。集合选择先前仅依据历史位置、未依据
输出长度或成绩；此后不改动其输入。新 seed1000/1001 是否实际跨越并延期刷新，
必须读取新运行计数；若仍无覆盖，不能宣称第一项优化已完成真实事件验收。

### 18:48 首个完整同 seed 对照：仅观察到约 0.37% 差异

双关与双开 seed1000 各完成 **45/45** 请求，输出均为 **40,588 token**。
逐请求输入、响应内容、KV 选择哈希、私有检索查询均一致；两组的 token 计数、
实际请求预算、artifact 哈希已复核，原始 NVML 日志也逐行重算，无外部 GPU
作业。对照文件为 `benchmark/dflash2/deepswe_first_full_pair_audit01.json`。

| 历史长度 | 双关 tok/s | 双开 tok/s | 差异 |
| --- | ---: | ---: | ---: |
| 全部 | 78.6281 | 78.9197 | +0.371% |
| <32K | 129.8191 | 130.0695 | +0.193% |
| 32K–64K | 76.4241 | 76.7766 | +0.461% |
| 64K–128K | 83.7333 | 84.0651 | +0.396% |
| 128K–256K | 75.8150 | 76.0166 | +0.266% |
| >=256K | 65.0586 | 65.3535 | +0.453% |

两组均没有延期投机或中途刷新，故不能据此判定第一项优化无效。整体每批验证
耗时 **37.5614 → 37.3959 ms**，这一个完整配对只显示很小的变化，不能声称
已建立稳定显著收益。整卡采样峰值均为 **48,931,602,432 字节**，按用户要求
继续只记录。双开 seed1001 已启动，用于观察另一个生成轨迹；它没有配套的
双关 seed1001，不能冒充第二个相同生成内容的配对。

代码复核另发现：延期刷新开关开启后，原代码每个投机周期都调用文本安全扫描，
即使尚未达到刷新阈值、helper 的结果与文本安全性无关。`qwen_native_backend.cpp`
现增加相同 `refresh_due` 判断，只有达到阈值后才扫描；刷新阈值、边界处理、
紧急刷新及概率接受逻辑均不变。该修改尚未纳入当前冻结吞吐队列。

`deepswe_two_optimizations_build14` 已完成 CPU 构建和两项策略检查。构建工具新增
经过哈希核对的 archive overlay：以 build13 为基础，确认所有头文件相同，仅
重新编译修改的 `qwen_native_backend.cpp`，重新链接 CLI 与 probes；build13 中
默认关闭的显存诊断代码保持关闭。新文本扫描优化仍待 GPU 验证及性能测量，
未宣称它已产生吞吐收益，亦未替换运行中的服务。

## 已获得的局部性能证据

build10 同一 attention probe、320 次暖调用、三次交替测量的中位耗时：

| 历史 | cache 关闭 | cache 开启 | 耗时下降 |
| --- | ---: | ---: | ---: |
| 1K | 9.83532 ms | 9.17916 ms | 6.67% |
| 64K | 183.948 ms | 179.275 ms | 2.54% |
| 128K | 341.022 ms | 337.161 ms | 1.13% |

上述数字仅说明调度缓存的局部收益。新的 DeepSWE 基线见后面的 17:28 记录。
长历史下收益有限，不能据此认定验证瓶颈已解决。

汇总入口 `benchmark/dflash2/summarize_two_optimizations.py` 支持当前验证目录，
统计各配置与历史分桶、接受率及耗时，同时保留未知旧计数器为 null。
`zero_draft_s`、`bridge_s` 与 `prefix_s` 有重叠，不能相加当作独立阶段。
未完成或资源/计数/哈希不通过的运行不参与已完成 A/B 比值；没有真实延期投机
覆盖时，报告明确要求补充事件测试。

## 16:55 后续候选与队列

128K 刷新回归已完成 16/28 请求；七个 greedy off/on 场景的生成输出全部
一致。已检查的代码围栏、XML、紧急刷新、取消恢复等内部策略对照也一致；
完整采样对照尚未结束，不能提前标记该阶段通过。

针对验证成本，独立 `build11` 已编译并通过 CPU 策略检查。新增默认关闭的
`QW3_FLASHINFER_VERIFY_TILE_INTERLEAVE=1`：只在 FP8、head_dim=256、
GQA=6、2–8 行及均匀分块布局时，交错调度各草稿行的同一个 KV tile。
FlashInfer device 函数接受显式逻辑 block 索引，候选只改变物理 CTA 的排列，
继续按原索引读入和写出 partial 结果；partition、每行归约和最终 merge 不变。
其目的是改善 KV 缓存复用，实际收益仍需实测，不能把此假设当成已确认瓶颈。

扩展的 operator probe 比较 cache/interleave 的四种开关组合，在可变页表、
其他 API 覆盖 metadata、变宽、回退位置及输出 guard 场景下要求逐值相同。
缓存固定开启后，再做三次交替的 interleave 开关耗时对照。

`qw3-dflash-interleave-diagnostic01.service`（PID 1536021）已确认存活，
正在等待现有实验锁；不会抢占 build10 的回归或 DeepSWE 重放。
它依次运行 1K/64K/128K operator、长 attention 边界和完整 target 一致性检查，
保留独立目录 `benchmark/dflash2/verify_interleave_diagnostic01`。
该队列也有独立的五分钟 timer，16:55 已成功触发。
此候选尚未执行 GPU 检查，也尚未接入真实任务的性能对照。

性能汇总新增命令行与每请求日志中的 K/G 预算核对；缺失或不一致则不计入
有效已完成 A/B。实际窗口 trace 仍需审核。已用历史真实日志校验秒数单位、
旧版本缺失计数器、加权吞吐，并检查预算缺失及变化能被拒绝。

## 17:07 长历史回归通过，进入真实前缀对照

`partition_fix_validation02_full/deferred_131072` 已全部通过：28 请求、
14 组 token 与内部策略对照。已重算该阶段 artifact 清单哈希，资源记录无
错误或越界。148,416 次采样的整卡峰值为 **43,546,116,096 字节**，
最大采样间隙约 47.38 ms。该合成回归使用 K64K/G32K、128K 输入历史，
不能当作真实 K128K/G64K 长任务的显存验收。

128K 围栏场景的零草稿步数 **635→1**，零草稿耗时 **15.117→0.024 秒**；
XML 场景 **550→1**、**13.178→0.024 秒**，生成 token 与内部策略一致。
4K 对应围栏场景为 647→1、14.569→0.023 秒。以上证明所测边界场景的
串行段被消除，不是 DeepSWE 加速比。分析记录保留在
`build10_refresh_regression_analysis01.json`、`build10_refresh_128k_analysis01.json`。

原执行器已自动进入 `diagnostic_baseline17`，server PID 1545464，已核对实际
HTTP 请求计划 `selection_budget=131072 generation_budget=65536`；首个 Quill
请求输入 131,310 token。尚无本轮新 DeepSWE 吞吐，不提前判定优化收益。

汇总额外报告 middecode 的查询、私有 prompt、生成查询、状态处理与重选
耗时；总刷新时间只取 query+reselect，内部子项不再次相加。已用真实完成的
回归记录校验单位与包含关系；发生刷新却缺少计时的情况保留为未知。

## 17:28 真实前缀基线、验证 profiling 与恢复

三项固定 >=128K 前缀均已完成两组基线，配置为 Q8_0/FP8、K128K/G64K、
T=1、k=20、p=.95、seed1000、自然停止/max4096。全部没有中途刷新。

| 前缀 | 旧 build17 输出 / decode tok/s | 修复 build10、两开关关闭 输出 / decode tok/s |
| --- | ---: | ---: |
| Quill，131310 输入 | 146 / 71.16 | 146 / 71.21 |
| Effect，132075 输入 | 449 / 76.52 | 539 / 58.84 |
| DynamoDB，132233 输入 | 199 / 72.01 | 199 / 72.24 |

旧组共 794 token / 10.68346 秒，加权 **74.32 tok/s**；整卡采样峰值
**46,651,998,208 字节**。修复组峰值 **46,649,901,056 字节**。
两组的文件完整性、HTTP/native 计数、逐请求 K/G 计划与资源记录均已核对。
这些是三个前缀的诊断，不是完整 45 请求或整场 SWE 成功率。

Effect 的请求字节、查询 span、初始 selected_hash 一致，但生成输出分岔。
不能把它的吞吐差异归因于两个开关。build09 与 build10 的冻结源码清单仅
`kernels_cuda.cu` 不同，diff 仅包含 causal-row partition 边界修复；编译参数
与基础库一致。恢复队列补测 build09 的同顺序三个请求：要求其输出与旧 build17
一致，以隔离数值修复影响；修复版本内 reference/plan_only、refresh_only/both
仍要求精确输出一致。旧版本与修复版本不等价的事实保留，任务质量需另行检查。
CPU 预审记录：`historical_baseline_drift_audit01.json`。

原 `partition_fix_validation02_full` 在启动 `diagnostic_refresh_only` 前的
socket bind 检查报 `OSError(98, 'Address already in use')`，没有启动该组 GPU
worker。之后检查端口已经空闲；不能据此认定当时是 TIME_WAIT 或某个外部服务。
原失败目录不改写。新恢复脚本 `resume_deepswe_two_optimizations.py`：

- 重核原 manifest/plan、二进制、权重、脚本、所有输出文件、参数和资源，
  复用八项通过的 GPU 检查及两组已完成基线；重算 4K/128K 的 28 请求分析。
- 每个后续服务分配独立可绑定端口，记录实际启动参数；绑定竞争仍视为失败。
- 在原共享 GPU 锁中继续三个开关诊断及 45 请求三组重放，不抢占其他作业。

无 GPU 预审 `deepswe_two_recovery_audit01` 已通过；人工构造的输出文件改动、
资源违规均被拒绝；已测试已占用端口与十个独立端口的分配。恢复服务 PID1556168
已运行，独立五分钟 timer 已建立。失败实验和新实验的证据分别保留。

另一个默认关闭的 CTA interleave 候选 build11 已完成全部六项 GPU 检查，
精确输出通过，但三次交替 operator 中位耗时无有效改善：

| 历史 | interleave 关闭 | 开启 | 结论 |
| --- | ---: | ---: | --- |
| 1K | 9.20559 ms | 9.84058 ms | 慢约 6.9% |
| 64K | 178.841 ms | 178.819 ms | 基本持平 |
| 128K | 337.274 ms | 337.402 ms | 基本持平 |

该候选不进入正式配置；保留负结果 `verify_interleave_diagnostic01`。

`deepswe_verify_profile01` 已完成相同 Quill 131310 输入的 build10 双开插桩
运行，输出仍为 146 token，资源记录正常，峰值 **46,398,242,816 字节**。
同步 operation-group 总时间 1.850 秒：长 attention **37.379%**、FFN
**34.813%**、attn_norm_batch **12.183%**。最后一项包含融合投影，不能称为
纯归一化耗时；recurrent/QKV projection 后紧邻的空 record 占比很小也不意味着
投影免费。这是同步墙钟归因，包含同步/主机开销，不作为吞吐测量。
结果说明 plan-cache 单独收益有限，后续内核候选需要针对 attention/FFN 并
通过数值一致性和真实工作负载测量，不能仅凭调度假设采用。

## 17:33 分岔隔离完成；下一候选已排队

`deepswe_two_recovery01/diagnostic_prepartition` 已完成三个请求，输出、usage
和停止原因与旧 build17 全部一致；两者仍与修复版的 Effect 输出不同。
`partition_drift_audit.json` 已记录该结果。修复前后冻结源码仅有上述 partition
修复差异，因此此次生成分岔已隔离到这项数值修复；这不等于修复版已通过真实
SWE 任务质量验收。同一修复版本的开关诊断目前继续运行。

已从工作源码撤回无收益的 interleave 分支及对应 probe 选项，build11 冻结
源码和全部结果继续保留。新的默认关闭候选
`QW3_FLASHINFER_VERIFY_PIPELINE_STAGES=1/3` 只改变 FP8、256 维、GQA6、
2–8 行验证时的异步载入流水线深度；未设置或设置 2 仍用原路径。planner 继续
使用原两级 kernel 的 occupancy 和 chunk 划分，避免再引入 partition 数值变化。
该假设尚未 GPU 验证，不计为已完成的优化收益。

probe 将比较三种深度 × cache 开关的六种组合，包含页表变化、metadata
覆盖、宽度变化、回退、guard 与 repeated-layer 检查。三轮轮换计时后，仅当
64K 和 128K 两处都超过预先规定的 2% operator 改善时，才进一步运行长
attention 和 target 精确一致性检查；之后仍需真实请求对照。

`qw3-dflash-pipeline-diagnostic01.service` / `verify_pipeline_diagnostic01`
已经排队，等待恢复 A/B 完成后取得同一个实验锁，再编译 build12 并执行诊断。
这样编译也不与正式吞吐测试重叠。源文件哈希已冻结；计时分析已检查正常输入
及截断拒绝，C++/CUDA 编译和 GPU 检查尚未执行。该队列也有五分钟 timer。

## 17:49 四配置诊断完成；补齐真实刷新覆盖

同一 build10、相同三项 >=128K 请求，四配置均输出 884 token；输入字节、
输出内容/usage、除实验开关和端口外的配置全部一致，另存
`deepswe_two_recovery01/diagnostic_four_way_identity.json`。

| 配置 | 加权 decode tok/s | 每验证批次 verify ms | 中途刷新 |
| --- | ---: | ---: | ---: |
| 两项关闭 | 63.299 | 44.226 | 0 |
| 仅延期投机 | 63.393 | 44.157 | 0 |
| 仅 plan-cache | 63.676 | 43.945 | 0 |
| 两项开启 | 63.498 | 44.058 | 0 |

双开相对关闭仅约 **+0.31%**；单 cache 约 +0.60%，全部差异小于 1%，
不据此认定有显著收益。四组整卡采样峰值都为 **46,649,901,056 字节**。
完整 45 请求的三组重放正在执行；17:47 基线已完成 15/45，不能用此时的
部分样本当最终分桶平均值。完整诊断汇总为 `two_optimizations_diagnostic_complete01.json`。

代码与冻结输入审查显示，K128/G64 的初始刷新点是 **192512 token**。
原 45 请求采样主要集中在 128K、196K 等桶起点，大多跳过这个刷新点附近，
不能期待其自动覆盖历史工具未闭合造成的串行尾段。原基准保持原样；另冻结
独立的真实前缀事件集合 `deepswe_refresh_event_requests01`：

- 只从已经选定的三项任务中，按历史长度选择刷新点前 4096 token 内最后三轮，
  另加该任务最初的短请求建立正常 harness epoch；不使用成绩、吞吐或后续
  completion 长度筛选。Quill 源轨迹无符合该窗口的请求，明确记为未覆盖。
- Effect 选择 ordinal 117/118/119，输入 191227/192200/192362 token，距初始
  刷新点 1285/312/150 token；DynamoDB 选择 ordinal98，输入190218，剩2294。
  加两项初始请求，共六个请求，原生 CPU renderer 已确认计数与哈希。
- 两个实验臂都使用 Q8_0/FP8、K128/G64、原61440刷新阈值、T1/k20/p.95、
  自然停止，统一 max_tokens=8192，以容纳原先观察到的超过4096的工具输出。
  此单独覆盖集合不混入原45请求吞吐，也不强制模型继续生成到阈值。

`qw3-dflash-refresh-event-replay01.service` 已排队，待当前 A/B 和 pipeline
诊断结束后运行。它要求完整三组重放的计数/预算/资源证据通过，再用固定
build10、cache开启比较延期投机关/开，seed1000和1001，各自独立服务，顺序
off/on/on/off。实际观察不到 deferred_batches 时明确记为覆盖缺失；采样
输出变化后仍需逐事件审核，不能直接把吞吐比称为消除同一串行段的加速比。
不执行外部工具，不宣称 SWE 任务通过。该事件队列也有独立五分钟 timer。

## 17:57 完整基线遇到显存限制，转入分配诊断

`deepswe_two_recovery01/full_reference` 完成 **24/45** 请求后，在 Effect
ordinal125（197110 输入）触发整卡显存监测限制。首次违规记录为
**48,252,125,184 字节**，停止前采样峰值 **48,566,697,984 字节**，超出
48,000,000,000 字节目标。资源记录35905个样本，最大间隔248.32ms，
无 foreign-process 或 observer 错误。原失败目录保留，不能将前24个样本
算作完整45请求通过，也不能沿用较短诊断的46.65GB承诺整场低于48GB。

末尾原生日志显示：197K请求已经完成私有查询和语义选择，选取4096块；
从135808的P检查点开始重放61302-token后缀。额外分配的具体归属尚待诊断，
日志位置本身不能证明是KV、临时工作区还是状态快照。

build12 已编译并通过CPU策略测试，但流水线深度1在1K operator的第一个
候选形状触发 `copy_to_host async: an illegal memory access was encountered`。
因此该候选没有性能结论；工作源码已撤回此分支和相应probe改动，build12
源码及失败记录保留。不能把这个错误描述成通过正确性但无收益。

补充事件队列按前置条件停止，错误为完整重放尚未完成；没有启动GPU请求。
原六个CPU冻结的事件输入仍可复用，显存问题解决后再以新运行目录接续。

为定位额外分配，build13仅在`kernels_cuda.cu`新增默认关闭的
`QW3_CUDA_TENSOR_ALLOC_TRACE`：记录大于等于1MiB的owned CudaTensor
分配/释放、标签、字节数和当时CUDA设备占用。与原有workspace/arena trace
共同使用；不改变数据精度、张量生命周期、K/G预算或数值运算。
此日志并不覆盖直接由backend/权重对象管理的所有cudaMalloc，不能冒充完整
allocator统计。插桩吞吐不用于性能比较。

`qw3-dflash-memory-build13.service` 正在独立构建；
`qw3-dflash-memory-diagnostic01.service` 等待它完成并取得GPU锁，随后重放
Effect ordinal0/64/125（1479/135846/197110输入）作为缩减的暖前缀复现。
仍使用prefill chunk2048、K128/G64、原采样、两优化关闭、48GB observer。
若这个较小序列不能复现，须扩展到原先的完整前序；不能把缩减复现通过当作
完整测试已解决。诊断脚本已通过CPU语法与分配累计/峰值解析检查，GPU待执行。
