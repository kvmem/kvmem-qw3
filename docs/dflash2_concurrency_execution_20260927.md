# DFlash 2 / KVMem 并发优化执行计划

2026-09-27。用户要求按计划逐步完成优化、测试，并验证最终提升。
本文件记录执行顺序、验收标准和实际进度；源码分析见
[优化依据](dflash2_concurrency_optimization_plan_20260927.md)。

用户后续要求跳过不必要的效率测试，专注优化和优化后的提升。因此每个候选只保留
必要正确性检查和种子1000的 C2/C4 首轮长历史对照；有明确收益后才补第二种子与最终
独立复测。已接近结束的完整基线继续保留，不为每个候选重新测一遍。

## 固定范围

- 主线：Qwen3.8-27B **Q8_0 权重 + FP8 KV**，每请求 **K128K/G64K**。
- 先 C1/C2/C4，不把更换精度、缩小 K/G 或更长输出当作优化收益。
- 先测吞吐与延迟，记录整卡显存和主机内存；48 GB 不作为本阶段中止条件。
- 普通非 KVMem 路径只做必要回归。现有单请求验收产物不覆盖，不引入 BF16 目标权重。
- 130 tok/s 仅供参考；低收益或失败候选也需记录原因，不为达到数字放宽正确性。
- 原生冷请求、统一解码诊断、会话复用/实际 agent 任务分别报告。

## 阶段与交付物

| 阶段 | 修改和交付 | 验收后才进入下一步 | 当前状态 |
|---|---|---|---|
| S0 基线 | 两种子、12 条长历史输入，C1/C2/C4；保留全部原始计时、token ID 和资源采样 | 6 个正式轮次结束，检查样本/配置/二进制身份、干扰和错误 | 已完成并审计：72条正式请求，另有2条预检；无进程/资源错误 |
| S1a prefill 调度 | 独立 prefill 每块交还调度器；允许已就绪子集验证；限制连续 decode 轮数，公平推进 prefill | 必要短场景 off/on 与混合请求交错检查，然后长历史 C4 首轮收益筛选 | 两种策略短测及长C4输出一致；最早请求优先最大停顿0.778秒、TTFT中位129.675秒，但整组230.980秒未提速，后续优先复用与计算效率 |
| S1b 查询/刷新调度 | 私有查询、重放、刷新准备分阶段执行；事务提交保留原子性 | 一请求刷新、其他请求持续输出；取消、回滚、页表、DeltaNet 状态不串扰 | guided C2／refresh C4共48条对应响应一致，分别观察55／80轮验证与query让出交错；私有query内取消和长任务仍待验证 |
| S2 联合验证 | 宽 Q8/SwiGLU 执行候选；修复 FP8 ragged attention 的批组成数值稳定性 | 固定输入数值/状态验证；同工作量计算成本下降；长请求端到端复验 | 宽Q8/SwiGLU未见整体收益；INT8 MMA合批验证明显加快，C2/C4暖服务吞吐观察值+15%/+17%，但输出改变、长历史数值门槛失败；质量canary数学7/8→7/8、代码6/8→4/8（新增2题长度截断），候选未验收。继续处理小批次退化及数值问题，不扩大吞吐矩阵 |
| S3 会话复用 | 每会话前缀租约、KV/原始历史、recurrent checkpoint 和草稿状态；新增后缀增量处理 | 冷/热策略分别对照；连续工具轮次、取消、回收、刷新提交；HTTP/bounded 生命周期符合要求 | CPU40项及短GPU会话32条通过；固定同一controller后，顺序/双请求交错的4条长历史响应文本与usage一致；冷/热分叉仍保留作数值问题，bounded及最终质量验证未完成 |
| S4 集成验收 | 组合已通过且有收益的改动；必要时才处理剩余草稿/启动开销 | 独立重复 C1/C2/C4、长短混合、持续补位、刷新场景；输出最终收益和限制 | 待前序阶段完成 |

最新推进：小批次MMA已接入默认关闭的实验路径，21项集成分派/memcheck和32条C4刷新
transcript检查通过。两条长历史逐层诊断共994次同输入投影均在算子容限内，但hidden
差异逐层放大，全层MMA仍不采用。正在检查只加速最终词表投影的候选：先要求原数值
门槛通过、所有body hidden逐位一致，再执行单组C4筛选；不重跑完整性能矩阵。
bounded共享arena的executor接入已准备，服务准入、全局资源配额和GPU生命周期验证
仍未完成。阶段范围及最终验收要求不变。

每个阶段单独封存源码/二进制、配置、开关、输入身份、测试结果和收益。
顺序上的依赖是“先拿到可解释的阶段结果”；若一个候选无收益或存在错误，保存失败、
定位原因或放弃该候选，再继续剩余有价值的项目，不把失败候选混入组合结果。

完整基线结果见 [长历史基线报告](dflash2_long_concurrency_baseline_20260927.md)，
审计证据为 `benchmark/dflash2/long_concurrency01/baseline_audit.json`。
两种子共同显示长 prefill 阻塞及联合验证收益不足；在所有请求都开始输出之后，
C2 后段吞吐约85.6–85.7 tok/s，C4约75.3–78.0 tok/s。后段的实际批宽会变化，不能
把它当固定C的稳态吞吐。跨并发输出也会分叉，后续必须定位算术差异和状态正确性。

## S1a 已准备的实现和队列

- 开关：`QW3_CONTINUOUS_KVMEM_COOPERATIVE_PREFILL`，默认关闭。
- 每块仍调用原来的独立 executor，不启用未通过一致性检查的联合 prefill。
- 公平性参数：`QW3_CONTINUOUS_KVMEM_PREFILL_DECODE_ROUNDS`，候选默认每4轮已就绪验证
  后推进一个独立 prefill 块。初版保留原分块大小；512/1024/2048 策略调参另做 A/B。
- 保留“先恢复所有上轮 logits 使用者，再复用共享批处理缓冲区”的顺序。
- 新 CPU 策略测试覆盖持续 decode 下 prefill 不饥饿、prefill 不独占、槽位取消/补充。
- GPU 短测探针新增 off/on 对照与不同长度输入；覆盖 DFlash、MTP、none、普通 QW3、
  CPU/GPU 采样、取消、补位、guided query 和刷新。

冻结源码：`benchmark/dflash2/cooperative_prefill_review01/`。
后台服务：`qw3-dflash-cooperative-prefill02.service`。
执行脚本：`benchmark/dflash2/run_cooperative_prefill02.py`。
不可变计划、构建命令及源码快照：`benchmark/dflash2/cooperative_prefill02/plan.json`、`source/`。
实时状态：`benchmark/dflash2/cooperative_prefill02/manifest.json`。
原 `cooperative_prefill01` 在构建前、尚未使用 GPU 时被精简队列替代，原计划及终止原因保留。

任务等待原始基线全部成功结束，随后等待 GPU 独占空闲并取得实验锁，才开始独立 CPU
构建、CTest 和 CUDA 条件分支构建。新二进制保留原 CUDA 核心归档，仅替换本次修改的
host 调度翻译单元；不覆盖现有服务、基线 CLI 或核心归档。

GPU 队列先做8组短测 off/on，随后做 C1/C2 长历史256-token上限的 off/on。
初始检查要求短测回调与长测 token ID 完全一致；任何不一致先停队列定位，
不得直接当作可接受随机变化。通过后先执行与原基线相同输入/策略的种子1000 C2/C4，
有收益后再补第二种子和最终独立复测，不自动为每个小改动执行完整矩阵。
这次自动队列只覆盖 S1a；不会自动把后续未实现的阶段标为完成。

## S1b 私有 query 候选

### S1a 刷新检查失败后的处理

`cooperative_prefill02` 已停止：前五组共120条短响应通过off/on一致性；
`short_refresh_c4` 的CPU采样第二遍中，请求2从第24个回调片段开始分叉。
其私有刷新查询文本也不同，尚不能判断是批形状数值差异还是状态恢复缺陷。
资源观察没有外来GPU进程、观察错误或资源异常。guided/dense短测、长历史pilot与
正式C2/C4候选均未执行，因此没有新的吞吐提升结论。

依赖它的query及wide-SwiGLU队列已按预设规则终止，未执行GPU阶段。失败证据和
原冻结快照保持不变。`refresh_diagnosis01` 使用相同核心归档和头文件，仅修改诊断
探针以记录全部差异、请求对应的刷新时刻，并缩小到不含取消的CPU采样两遍。
先比较关闭优化的重复结果与off/on；此诊断不作为性能测量，也不放宽验收标准。

按用户要求复用已封存基线，跳过无关效率矩阵；正确性通过后只对有价值的候选做
小范围吞吐筛选，有收益才增加独立重复和最终验证。

在等待 GPU 时准备独立候选，尚未启用到 S1a 的冻结测试中：

- 开关 `QW3_CONTINUOUS_KVMEM_COOPERATIVE_QUERY` 默认关闭，且必须同时开启独立
  prefill 调度。私有提示词、thinking token、完整 query 推进单元结束后才让出。
- 私有 snapshot、query 内容、页池 transient 标记都属于当前请求；让出前将下一
  argmax 保存在栈上。其他请求可以推进；不在 forward 内部、恢复和 KV 重选提交中让出。
- 取消在让出前后检查，沿原有异常路径撤销 query probe 并恢复快照。
- 日志记录 `continuous_controller_query_yield`，验证日志记录 `query_yields`，后续
  检查确有其他请求在 query 期间继续推进。含让出等待的请求阶段墙钟不可充当独占计算成本。
- 探针增加仅 query 开关的 off/on 对照，以及只有一请求执行私有查询/刷新的混合场景。
  需要先证明状态正确性和交错推进；暂不重跑不会触发私有 query 的12输入长历史矩阵。

冻结源码和二进制：`benchmark/dflash2/cooperative_query_build01/`；构建脚本
`benchmark/dflash2/build_cooperative_query01.py`。CPU语法检查、CUDA条件分支编译/链接及
`cooperative_request_test`（新增暂停事务独立回滚检查）已通过。尚无该候选的GPU正确性
或吞吐结论；仍未实现可分段的重选搬运及S3会话复用。

针对性GPU队列已提交到 `qw3-dflash-cooperative-query01.service`，状态记录在
`benchmark/dflash2/cooperative_query01/manifest.json`。它等待 S1a 队列成功完成；若
S1a 失败则停止自身并记录原因。仅运行 guided C1、mixed guided C2、mixed refresh C4
（含补位与同行请求取消）三组 off/on 检查，要求每请求回调一致，且并发场景日志确认
在私有 query 让出期间执行了其他请求的验证。这不覆盖“私有 query 内取消”的GPU验证，
也不能代替真实长刷新任务的吞吐测量。

## S2 宽 Q8 FFN 候选

源码检查确认稳定验证的 gate/up 仍按8行分块，未使用已有宽 Q8 fanout；down投影已经
经过 `math.matmul`，不要把它误报成尚未支持宽投影。新增可选 gate/up 宽路径，使用
现有 FP32 scratch 和逐元素 SwiGLU，不新增 CUDA kernel、不更改精度或单请求计算。
详见 [候选说明](dflash2_wide_swiglu_candidate_20260927.md)。

独立构建 `benchmark/dflash2/wide_swiglu_build01` 已通过 CPU 语法、CUDA host 编译和
算子/短测/长测探针链接。`qw3-dflash-wide-swiglu01.service` 等待 S1a 完成及 query 队列
让出实验锁，再执行21项定向算子一致性、短C4的32条off/on响应、两个候选的单组长C4。
两个候选为“已有宽线性投影”和“宽线性投影+宽SwiGLU”，只取原始12条输入中的前4条，
复用S1a对应的第一组结果，不重新跑基线或每候选完整矩阵。输出ID改变时先停止定位。
此次是候选筛选；没有独立重复，不据此直接作最终提升结论。

## S3 状态归属准备

共享 backend 的暖启动记录原来包含20项独立字段：历史匹配、P/M checkpoint、重放
checkpoint环、query stash坐标和可恢复标志。这些信息需要与所属executor一起保留；
仅延长KV页的寿命会让不同会话共用元数据，不能完成安全复用。

已新增 `src/kvmem_prefix_state.hpp`，将这些字段组成不可复制、可移动的独立状态对象。
12个helper增加显式的状态/executor上下文；`generate_mtp`可以绑定一个调用者持有的
稳定对象。普通路径保留使用backend自身状态的包装函数，算法与checkpoint准入规则
不变；原来的legacy两遍query replay限制也保留。当前没有调用者为CB传入持久状态，
因此这一步尚未开启多会话复用。

源码转换前后及差异：`benchmark/dflash2/prefix_state_review01/`。
冻结构建：`benchmark/dflash2/prefix_state_build01/`；完整CPU构建和39项CTest通过，
CUDA host、算子/短测/长测探针编译链接通过。尚无该重构的GPU会话回归结果，现有S1/S2
队列继续执行各自原始冻结二进制，不将此重构混入前序归因。

下一步须将“executor＋前缀状态”作为会话驻留对象一起租出/归还，禁止同一会话同时
写入；冷请求准入前回收空闲驻留对象，取消/错误不得发布可复用状态；并接入完整历史
请求的缓存标识，与现有start/append/finish片段会话API区分。之后才能实现并验证
HTTP路由、bounded repository/GC、CPU/NVMe配额及连续工具轮次的生命周期。

工作区另已准备会话驻留与完整历史HTTP请求的接入草稿：实验开关默认关闭，
`kvmem_prefix_cache_key`区别于片段会话API；同一key禁止重叠写入，executor与前缀
状态成对租出/归还，取消或失败不发布缓存，冷准入回收空闲驻留对象。该草稿已在
`session_resident_cpu01` 完成CPU构建和40项CTest；尚未完成CUDA重建/GPU验证，
不能称为已支持会话复用；bounded与continuous同时开启的限制
仍保留。公共GenerationOptions布局已有变化，后续需完整重建核心及调用者，禁止
将新头文件与此前冻结归档混用。

## FP8 合批数值稳定性诊断

`refresh_diagnosis01` 的OFF/OFF与随机入队OFF/ON各8条响应均通过，表明原刷新
分叉不是每次必现；不能因一次未复现而撤销失败。后续固定请求2在ON侧最先入队，
使其能独立执行短验证尾部，另用独立attention算子排除刷新/采样逻辑的影响。

源码确认原ragged planner随批平均Q长度选择CTA_Q=16/64，也按批组成调整KV分片。
`stable_ragged01` 算子实测：page16下，256/2048历史合批相对单独执行分别有
31,778/33,198个FP32输出元素不同；新固定分块候选均为0。这里统计的是两种请求
排列的元素差异，不能直接等同于模型误差幅度或任务质量下降。

第一版固定分块在128K/C4遭遇临时工作区上限而停止；未启动模型复现。第二版使用
实际query行数与KV分片数构造紧凑中间结果布局，避免为每个分片预留整块64行。
开关 `QW3_EXPERIMENTAL_FP8_STABLE_RAGGED_PREFILL=1` 默认关闭，仅针对Q24/KV4、
head256、page16/32、每请求1..8行的FP8验证。固定CTA_Q=64、KV分片1024token；
其余形状沿用原路径，固定路径分配失败时明确失败，不静默改变算术。

第二版冻结源码/二进制：`stable_ragged02`；GPU检查：`stable_ragged_checks03`。
先检查256/2K/128K与page16/32的混合行数、请求重排、输出边界及memcheck，
再比较固定入队顺序下原路径与固定分块的刷新off/on。此阶段不测算子速度。
第二版完成24项算子比较（原路径/固定路径各12项；固定路径差异全部为0），输出边界检查通过，CUDA
memcheck报告0错误。固定请求2最先入队后，旧路径再次在请求2的第24个片段分叉；
固定分块路径的8条对应响应完全一致。两边资源观察均无外来GPU进程或观察错误。
这确认并修复了该复现用例的批组成数值不稳定问题；仍需集成路径与长历史验证。
固定分块可能改变旧路径的输出，不能直接将其与旧配置的吞吐比解释为等输出加速。

## 精简后的集成队列

`integrated_concurrency_build02` 从完整冻结源码重建CUDA核心、CLI/server、短测、
长测及算子探针，包含新的GenerationOptions布局。此前build01仅因未指定nvcc路径
而在配置阶段失败，没有执行GPU操作；build02显式指定CUDA13编译器。

build02已完整构建成功。用户再次要求优先多并发优化收益，原
`qw3-dflash-integrated-concurrency01.service` 在GPU启动前取消，清单确认 `stages=[]`，
原计划保留。新 `qw3-dflash-integrated-concurrency02.service` 只保留本轮实际启用路径的
prefill刷新C4、宽SwiGLU算子和C4检查，然后立即测长C4。
私有query和会话复用检查后移，这两项在本轮长测中均关闭；不因未启用功能阻塞性能筛选。
检查失败即停止，不继续计时。

随后仅执行原种子1000输入前4条的一组长历史C4前后对照，保留自然EOS及4096上限。
两侧都启用固定FP8分块，以隔离数值变化；对照关闭独立prefill与宽Q8/SwiGLU，候选
开启这三项，暂不混入会话复用和不会触发的私有查询让出。直接比较完整输出ID、服务
墙钟吞吐、输出窗口吞吐、TTFT和停顿，不额外执行短输出长历史pilot或完整C1/C2/C4
矩阵。此为组合候选筛选，有效益后再分析单项贡献和做最终独立重复。

清单：`benchmark/dflash2/integrated_concurrency02/manifest.json`。截至本节记录时
完整队列已完成。四请求输出ID完全一致，但组完成时间229.755→230.441秒，未见吞吐
提升；最大停顿150.50→3.56秒的同时，TTFT中位数126.57→197.39秒。
详见 [首轮长C4对照](dflash2_concurrency_focused_ab_20260927.md)。

后台每15秒更新运行清单、每5分钟输出心跳；GPU外来进程、错误或检查失败会停止
自有测试进程并保存证据。构建期间不与原正式基线并行，以免污染计时。

## 首轮结果后的推进

- `concurrency_lifecycle01` 在长C4结束后检查私有query让出与会话驻留，使用完整冻结
  build02。guided C2 的16条、refresh C4 的32条对应响应一致；分别记录108／124次
  query让出，其中55／80轮伴随同行验证。refresh覆盖同行请求取消，不代表私有query
  内取消也已验证。会话重复/轮换各16条响应通过，两份完整transcript相同，暖请求
  确认11次正的前缀复用（736／768 token）；仍只是K以内相同提示词的检查。
  整个原生队列80条响应通过，证据汇总 `native_lifecycle_review.json`。
- `http_session_concurrency01` 等待上述检查完成，再启动独立本地测试HTTP服务。
  `tests/dflash_http_session_probe.py` 使用真实Quill/Effect两任务连续两轮，保持
  K128K/G64K、Q8_0/FP8；仅256输出上限的首轮筛选，记录HTTP墙钟与TTFT。
  SSE事件不是逐token时间，不能冒充原生decode测量。该队列不启用bounded。
  当前测试服务及回放客户端已启动，正在执行轮换key的参考轮次，尚无暖请求性能结论。
- 新增 `QW3_CONTINUOUS_KVMEM_PREFILL_OLDEST_FIRST`，默认关闭，配合独立prefill。
  每次给最早请求一个prefill块，仍受4轮decode预算约束；取消后选择下一个最早请求，
  槽位复用不改变请求年龄。`concurrency_oldest_build01` 的策略CPU测试及host编译/链接
  已通过，公共ABI和CUDA内核未改变；原始build02及正在运行队列没有覆盖。
  `concurrency_oldest01` 排在HTTP检查后，先必要刷新一致性，再测相同长C4候选，复用
  本轮完整参考输出；宽Q8/SwiGLU关闭以隔离调度策略。仍须以完整组耗时、TTFT和停顿
  联合判断，不能只报告窗口吞吐。

## 如何判定“最终提升”

1. **正确性先行。** CPU 和适用 GPU 回归无错误；每请求状态隔离、取消/补位/刷新恢复
   通过。调度阶段先要求相同算术下的确定性对照；如发现算术随批形状变化，必须先定位。
   改变数值路径的后续优化还需独立数值与质量评价。
2. **同时报告三类时间。** 含 prefill 的服务墙钟吞吐/请求完成时间；明确共同时间窗口的
   decode 吞吐；每请求 TTFT、token间隔 P50/P95/P99 和最大停顿。重叠请求的 decode
   耗时不能相加当总吞吐分母。
3. **防止只移动等待。** S1a 若缩短首 token 后的停顿，却延迟首 token，要明确报告这个
   权衡；不能只用输出窗口吞吐上涨宣称整体加速。长任务历史复用也不能只在候选侧开启。
4. **控制输出长度影响。** 保留每请求 token ID、输出长度、自然 EOS/截断和接受率。
   自由生成长度不同的 tok/s 比值只能作为工作负载观测，另用固定验证输入的阶段回放
   判断计算收益，并报告相同请求集合的完成墙钟时间。
5. **保留独立对照与重复。** 单项候选先小样本筛选；最后保留的配置至少3次独立计时，
   覆盖两个种子及 C1/C2/C4。基线和候选交替运行，记录测量波动；正收益须超过波动。
   C1 若出现持续超过5%的下降，先定位并调整，不能只用 C4 的收益掩盖。
6. **最终交付。** 一张同配置前后对照表、各项开关的独立贡献、失败/无收益候选清单、
   精度/状态检查、显存和适用范围。只有完成真实会话/工具链验证后，才报告 SWE agent
   服务收益；原生探针结果不替代 SWE 任务成功率。

本计划采用“实测收益决定保留哪些优化”的结束标准；不预先承诺倍数或130 tok/s。

## 2026-09-27 后续执行调整：优先持续多请求

词表投影MMA候选已完成C4筛选：8条完整输出完全一致，暖组服务吞吐51.56→51.46
tok/s，无确认收益，停止扩展该候选测试。全层MMA尚未通过数值/质量门槛，默认关闭。

下一项已经实现controller空槽在同行prefill悬挂时接纳请求，并增加独立会话续发的
长历史回放。冻结构建已完成，`concurrency_replenish01`先检查补位和取消正确性，再
运行唯一一组C4持续会话A/B；不重跑C1/C2/C4全效率矩阵。
见[补位开发与验证记录](dflash2_concurrency_replenishment_20260927.md)。
bounded并发服务入口与资源管理仍是后续必须完成的阶段，目标保持active。

持续stream补位A/B现已结束，8请求完整输出一致，339.90→340.56秒，没有整体收益，
本项不扩展效率矩阵。随后已将bounded并发服务入口、固定owner额度、共享GPU页
预留和请求独立warm state接入默认关闭的实验路径；NVMe并发临时文件竞争已修复。
构建与相关CPU检查完成，bounded C4刷新/取消及驻留复用的32条GPU响应重复一致，
真实K128K/G64K C4正在运行。相同资源配置、单客户端顺序提交的完整输出对照已排队。
HTTP入口已构建未部署；GC压力、HTTP和最终性能重复仍待完成。
