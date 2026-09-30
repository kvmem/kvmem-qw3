# 9B 后续记忆评测计划：复用 27B 实验协议

2026-09-20 18:11 已完成固定39题评分及原评分器独立核验：Compaction+RAG
21/39（53.85%），同ID KVMem17/39（43.59%）。摘要失败8题、最终回答length
2题均保留分母；不是100题完整结果。最后1题使用同步CUDA规避运行崩溃，
仅系统执行方式有披露差异，不将该题时延用于效率比较，底层bug尚未定位。
18:12已自动恢复原Q8摘要阶段，随后继续100题完整答题与评分；39题已有答案
直接复用，已核验后续进程没有CUDA_LAUNCH_BLOCKING。结果与逐题CSV见当前
512K目录`partial39_report.md`、`partial39_completion.json`、`partial39_scores.csv`。

2026-09-20 17:22 用户要求先评分已处理的39题。这39题只是完成前置摘要：
35题成功、4题模型生成失败。已固定39个ID及摘要哈希，暂停后续Q8摘要生成，
临时切换既定vLLM0.21/FP8/MTP4后端，完成这批题的补充摘要、答题及原评分，
失败保留在分母中。`run_partial39.py` 完成独立评分核验及同ID KVMem对比后，
自动接回原campaign，继续剩余摘要并复用已生成答案，不重新采样。
只是调整执行顺序，模型/采样/提示/检索/容量/评分协议均不改变；切换后端的
冷启动会影响缓存及时延，阶段性39题结果不等同于100题完整结果。
结果写入当前512K目录`partial39_completion.json`和`partial39_scores.csv`。
已有40题第1轮的有效checkpoint保留，停止时未完成请求不计模型失败。
取消的定时器不恢复。

17:56 在这批39题已保存36题评分时，vLLM报告CUDA非法内存访问并退出，
37/38的在途请求断流，39尚未开始。故障记录保存在
`incidents/partial39_cuda_20260920/`。17:58按原后端命令重启，保留36题答案、
评分及有效摘要，从未完成的3题继续；未将基础设施故障算作模型失败，
未修改采样或内核配置。异步错误栈不能确定最初出错的CUDA内核。

18:01和18:03，最后一题又在原vLLM执行模式下发生非法访问；已完成38题、
20题正确。冻结请求的同步CUDA诊断可正常完成，诊断输出独立保存、不评分。
18:09对唯一未完成的正式回答启用`CUDA_LAUNCH_BLOCKING=1`作为运行规避，
模型、权重/KV精度、MTP、提示、采样/seed、thinking/输出预算与评分不变。
这是需要披露的系统执行差异，该题耗时不用于效率比较；不宣称底层bug已修复。
`partial39_execution_override.json`记录范围和原因，原38题答案哈希核验不变。
同步标志仅传给这次FP8子进程，不会带入恢复后的正常campaign。

2026-09-20 14:49 用户进一步授权取消摘要的10K thinking限制。512K对照组前置Q8
使用`--reasoning-budget -1`，后续FP8摘要请求不发送`thinking_token_budget`；
thinking保持开启，思考加正文的总输出仍为32768。其余采样、228336摘要输入、
最终回答的8192 thinking限制、检索与评分不变。上一版15题成功摘要及2题失败
单独保留，不混入新统计；整组100题在
`benchmark/agentlongbench_9b_matched27b_20260916/512k/compact_rag_nothinkcap_20260920/`
重新生成。已完成KVMem和≤256K不重跑。1M固定第三方摘要协议不变。
本条覆盖下文“thinking仍10000”的历史要求。详情见
`docs/agentlongbench_9b_summary_nothinkcap_20260920.md`。

2026-09-20 13:39 用户明确授权后续摘要不用temperature=0，摘要输出提高到32768。
512K KVMem已完成100题，原评分37.00%，保持原结果；旧Compaction+RAG前置摘要
3题成功、9题循环耗尽25000预算，已归档且不混入新统计。新512K对照目录为
`benchmark/agentlongbench_9b_matched27b_20260916/512k/compact_rag_samplingfix_20260920/`。
全部100题重建摘要；前置Q8与后续FP8摘要均采用温度1、top_p=.95、top_k20、
min_p0、presence_penalty1.5、repetition_penalty1；thinking仍10000。
为容纳32K输出及包装/安全余量，摘要输入由232000调至228336；这是必要容量
调整，切段边界变化明确披露。最终回答参数、64K输入、检索及评分不变。
1M仍复用锁定的第三方摘要，没有新增摘要生成；后续新生成摘要不得再用贪心。
详情见 `docs/agentlongbench_9b_summary_fix_20260920.md`。此条覆盖下文历史协议
中的摘要采样/预算要求，其余约束继续生效。定时器保持禁用。

2026-09-20 07:25 接续：≤256K两组均已完成250题及原评分/ID独立核验。
KVMem平均任务得分50.1428%，完全正确120/250；修正采样的Compaction+RAG
得分25.7995%，完全正确64/250，11个生成失败（8摘要、3回答）保留在分母。
新版Compaction+RAG产物仍为`le256k/compact_rag_samplingfix_20260919/`。
凌晨03:34切换512K时，模型已加载但绑定18124端口失败，尚无512K答案。
现已给campaign增加有界的端口可绑定检查，再启动模型；TCP退出等待期间
仍遵守paused标记。3项端口检查测试通过。已恢复512K KVMem，模型/协议
参数和已完成结果未改动，后续仍是512K对照组、1M两组。
故障和恢复记录：`incidents/stage_bind_20260920/`、`launch_resume_20260920.json`。
具体是否TIME_WAIT触发是基于服务切换时序和不同socket复用选项的推断，
原日志能直接确认的是端口绑定失败；未将其计为模型失败。

2026-09-19 19:43 生成失败修正：用户要求调整后继续。≤256K Compaction+RAG
旧46个终态保留为诊断版本，新正式目录为同实验根目录下
`le256k/compact_rag_samplingfix_20260919/`，整组250题摘要和回答重新生成。
新目录采用Qwen3.5推荐采样缓解thinking循环，原thinking开关、输出上限、
提示/检索/预算/评分保留；原35题提示差异继续标注。三个固定失败请求的新采样
诊断均自然结束，详见 `docs/agentlongbench_9b_sampling_fix_20260919.md`。
后续512K/1M仍按各自协议；KVMem和27B不重跑，不能再声称双方采样参数完全相同。
恢复顺序实验进程，旧版结果不得混入，取消过的30分钟定时器继续保持禁用。

2026-09-19 接续：用户再次授权完成剩余9B测试，并明确同意≤256K对照组中
无法恢复历史变体的35题使用仓库原始Codex摘要提示，单独标注协议差异。
其余215题摘要提示逐字对齐，250题原回答提示和原逐题评分全部重新核验。
≤256K对照组启动器为实验根目录的`run_le256_baseline.py`，复用冻结的原摘要、
RAG提示、SSE客户端和评分函数；保留逐题原采样、thinking及请求输出上限。
原4个directmax摘要请求的30000输出上限也保留，不自行改成剩余context预算。
仅复用模型无关的原检索块，不复用27B摘要；复用检索的旧耗时不计为新实测耗时。
已完成的250题9B KVMem保留并复算通过，不重跑27B、LongMemEval或MemoryAgentBench。
剩余五个cell共550次最终回答按既定顺序执行，每个cell完整评分和ID核验后切换。
沿用已存在的单个campaign进程，取消过的30分钟定时检测不重新启用。
授权和审计记录：`le256k/compact_rag/protocol_deviation_approval.json`、
`reference_lock_before_20260919.json`、`reference_lock.json`及`resume_20260919.json`。

2026-09-16 19:00 接续：用户明确授权执行其余测试。AgentLongBench 新队列为
`benchmark/agentlongbench_9b_matched27b_20260916/`；当前已启动 ≤256K 的9B KVMem，
前两题正式验证通过，已有连续答案和评分。三档仍为250/100/50题，每档两种方法。
400题KVMem原提示哈希和27B逐题评分均复核通过。512K和1M的Compaction+RAG
已恢复2026-08-11的v1脚本，分别复现100/50题原回答提示和评分；不使用后来为
latency实验修改的v2流程。512K前置两轮Q8摘要的200个旧提示也已逐字复现，
新实验将由9B重新生成这些摘要，再切原vLLM0.21/FP8/MTP4；1M保留原固定第三方
摘要，使用原vLLM0.22/FP8/MTP4。两组复用已校验的模型无关检索产物，旧检索
耗时只是来源元数据，不能算作新的端到端实测时延。

尚待解决的协议缺口：≤256K旧Compaction+RAG的250个回答提示均能复现，但其中
35题摘要使用额外的防循环提示，当前脚本只能复现其余215题的摘要提示哈希。
35题具体ID、来源与预期哈希保存在该组`compact_rag/reference_lock.json`。
已请求用户确认：继续恢复原提示，或明确接受并标注改用仓库原始提示的差异。
用户未确认前该cell保持`ready:false`；当前KVMem正常继续，按既定阶段顺序，
完成后等候此协议缺口解决，不跳过基线进入下个长度档。其余五个cell已准备启动器。
新队列是本次授权的顺序实验进程；被取消的每30分钟检测仍保持disabled/inactive。

状态（2026-09-16 更新）：LongMemEval-S 修正队列已完成 102/102 推理和配对评分，
KVMem 78/102，Compaction+RAG 73/102。用户已取消每30分钟检测，定时器保持禁用。
用户随后明确授权启动其他数据集；MemoryAgentBench >256K 的 9B KVMem 已完成
30 contexts / 1316 问及全部评分。用户进一步授权启动 Compaction+RAG，当前启动器
为 `benchmark/memoryagentbench_9b_matched27b_20260916/run_compact_rag.py`，输出在
同目录的 `compact_rag/`。后续各方法仍按以下逐项冻结协议执行，不恢复定时检测。

2026-09-16 16:47：Compaction+RAG 在1100问后生成 InfBench 摘要时两次触发
vLLM CUDA 非法内存访问。现已加入仅针对 FlashInfer decode plan 的 pinned host
缓冲区复用等待；MTP4、异步调度、精度、预算及摘要/回答参数保持原值。
同一失败请求完整25000-token输出与同步诊断逐字一致，正式续跑已落盘第1101问，
之前1100份答案哈希不变。补丁哈希、验证和故障边界保存在
`compact_rag/backend_repair.json` 及 `compact_rag/incidents/cuda_illegal_access_20260916_161442/`；
诊断耗时不计入正式结果，后续效率报告保留这一后端修复边界。

本次 MemoryAgentBench 保存了 `reference_lock.json` 和 `protocol_validation.json`，
首个正式 context 的原始输入、问题、答案元数据与 token IDs 均与27B参照一致。
启动前发现 semantic-chunk bootstrap 的 replay 初始化会把新增历史索引边界截回
64K；已修正初始化次序，71,680-token archive 与两个独立 query 回归通过。
失败属于引擎基础设施错误，未产生正式答案，不计为模型错误；详见实验目录
`failure_and_repair.json`。修复不修改预算、检索/完整 replay、采样或输出上限。
Compaction+RAG 原摘要/RAG源码已按历史SHA恢复，30 contexts / 1316 问的身份及
模型无关检索产物已校验。进一步追溯实际摘要血缘发现，27B正式结果在26个context
保留了33轮由原Q8/MTP0服务器生成的前置摘要，再由vLLM FP8/MTP4继续压缩和回答；
另4个context没有保留导入轮次。9B复现这一两阶段流程，按原停止条件重新生成摘要，
不会复用27B生成的文字。前置摘要阶段已实际发出232000-token请求。
FP8阶段权重为社区 `lovedheart/Qwen3.5-9B-FP8`，revision
`5d77dcb2e2c606bc261b5b8e946a67781f18d733`；128x128动态FP8与原27B格式一致，
全部权重分片SHA256已验证。实际vLLM启动和MTP就绪检查在前置摘要阶段完成后执行。
详见实验目录 `compact_rag/protocol_lock.json` 和 `initial_backend_provenance.json`。

2026-09-16用户要求先诊断生成失败并修复。旧LongMemEval-S队列在79个终态处
保留为失败诊断版本；当前正式修正队列改为
`/home/chaidi/qw3/benchmark/longmemeval_9b_legacy27b_samplingfix_20260916/`。
修正仅为摘要和回答采用官方Qwen3.5通用thinking采样（温度1、presence penalty1.5
等），旧摘要/RAG/提示流程不变，输出和thinking仍不增设上限。
102个Compaction+RAG样本全部重新生成，KVMem参照不重跑，两个版本不混算。
详见 `docs/longmemeval_9b_generation_fix_20260916.md`；完成门槛针对新目录，
不能将旧版本79题与新版本拼接。后续各数据集仍按各自冻结协议核对，不自动
把此处的采样修正推广到尚未验证的其他单元。

## 目标与范围

以 Qwen3.5-9B 替换历史 Qwen3.6-27B，评测 KVMem 与 Compaction+RAG。
按每个数据集、每个方法的**具体已完成实验**复用参数与代码，不把当前
LongMemEval-S 的配置推广到其他数据集，也不为了统一表格而修改原预算。
不重跑 27B，不在本轮追加 4B。Sliding Window、Compact-only、Full Context
不加入此次核心队列；如后续扩大方法范围，单独列出新增工作量。

所有历史读者答案只作参照。模型相关的 KV/archive 和原由 27B 生成的摘要
必须重新构建；独立于读者模型的检索索引及固定第三方摘要可在身份校验后复用。

## 顺序、样本与预算

| 阶段 | 数据范围 | 每个方法的题数 | KVMem active budget | KVMem block | Compaction+RAG 输入规则 |
|---|---|---:|---:|---:|---|
| 0 | 当前 LongMemEval-S 旧协议重跑 | 102 | 已有 KVMem 结果另存参照 | 32 | 原 MiniLM top-22；不新增 32K 裁剪 |
| 1 | MemoryAgentBench >256K 已完成组 | 30 个 context / 1316 问 | 65536 | 32 | 旧正式版本：64K 完整输入上限 |
| 2 | AgentLongBench ≤256K long-250 | 250 | 32768 | 32 | 原 top-30、1024/128；保留实际输入长度 |
| 3 | AgentLongBench 512K normal-100 | 100 | 65536 | 32 | 旧正式版本：64K 完整输入上限 |
| 4 | AgentLongBench 1M canonical-first-50 | 50 | 102400 | 128，32-token scoring slices 须由日志复核 | 旧正式版本：100K 完整输入上限 |

AgentLongBench 的 ≤256K、512K、1M 是历史长度分组，不是三档统一的
active window。≤256K 组含不同长度样本，不能改成新抽取的纯 256K 集。
1M 文件本身不止 50 条，必须使用正式结果覆盖的 canonical first-50。

阶段 1–4 共 1716 个问题，每题两个方法，共 **3432 次最终回答生成**，
另有摘要生成、archive 构建、检索建索引和评分。MemoryAgentBench 不是只跑
30 次回答；同一 context 的多个问题必须作为独立分支，不累积之前的问答。

## 锁定的 27B 参照产物

下列目录都在 `/data/chaidi/kvmem_eval/results/` 下，除非注明其他路径。
执行前复制并哈希保存实际 run config、manifest、启动命令及对应源码版本。

### MemoryAgentBench

- KVMem：`memoryagentbench_over256k_raw_query_k64_b32_semantic_chunk_20260812_110226/`。
  已有 1316 问和完整 special judge；采用 raw-question score span、K64/G32、
  B32、FP8 archive / FP16 index-query、adaptive gain .10/.06、sink512/recent0、
  2048-token semantic chunks、immutable refresh=8、CPU-only archive。
- Compaction+RAG：
  `memoryagentbench_over256k_compact_rag_notail_cap64k_vllm021_20260812/methods/compact-rag/`。
  目录名仍有 vllm021，实际升级及 serving environment 必须由运行记录确定，
  不能从目录名推断。保存的参数规定 FP8 KV、MTP4、无 thinking、temp .6/
  top-p .95/top-k20、官方 source-specific 输出上限、32/8 Jina chunks、
  2200 候选、动态 64K 输入预算、完整历史压缩且无 raw tail。
- 样本选择：`scripts/kvmem_eval/memoryagentbench_over256k_contexts.jsonl`。
  核对 split/source/dataset_row/question_index 四元组，30 contexts、1316 问。
- 官方模板固定时间、chunking、context padding、frozen branch、prefix cache
  priming/guard 和逐问题重选策略均继承对应原实验，不按新默认值替换。

### AgentLongBench ≤256K

- KVMem：`agentlongbench_kvmem_32k_b32_lmeparams_thinking4k_20260716/`，250/250 完成。
  Mean-K，FP16 KV，K32/G32/B32，thinking4K，输出32768，temp .6/top-p .95，
  query replay 和原 query span、原 tier placement 不变。
  `seed:null` 要追溯原服务默认值，不能直接改成 1000。
- Compaction+RAG：
  `/home/chaidi/AgentLongBench_Motivation/outputs/compactrag/`，250 条 manifest。
  原 top-30、1024-token chunks/128 overlap、原 Codex 摘要流程和提示。
  已见实际记录为 temp0/top-p.9、输出20000，不能统一成 KVMem 的 .6/32768。
  最终输入并非严格32768，须保留原装配与实际 token 计数。
- 数据：`/data/chaidi/kvmem_eval/data/agentlongbench_250/{samples,manifest}.jsonl`。
  不使用按方法分歧筛出的 52 条诊断集，也不使用未完成的早期 125 条结果。

### AgentLongBench 512K

- KVMem：
  `agentlongbench_512k_k64_g32_semantic_chunk2048_fullquery_recent0_adaptive_b32_fp8_mtp4_gaporder_20260806/`，100/100 完成。
  保留历史 manifest 的执行顺序；不得根据新结果重新排序或挑题。
  K64/G32/B32、adaptive、FP8 KV、2048 semantic chunks、thinking8192、
  输出32768、temp .6/top-p .95、seed20260722、ctx655360。
- Compaction+RAG：
  `agentlongbench_512k_compact_rag_notail_cap64k_vllm_fp8_mtp4_20260811/`。
  配置指纹 `1930f5a16738eaa2dd64ec78c6c0404e256cafd764b854bb32f98a1ca0dd7334`。
  232000 摘要输入、25000 摘要输出、原摘要 thinking10000 和循环覆盖流程；
  无 tail，Jina32/8、2200 候选、动态64K，回答 thinking8192、输出32768、
  temp .6/top-p .95/seed20260722。原由27B生成的前置摘要和后续摘要均用9B重建，
  不能通过 `source_compact_root` 隐式读取旧27B摘要。
- 数据：`/data/chaidi/kvmem_eval/data/agentlongbench_512k_normal100/samples.jsonl`。
  与原 selected_samples manifest 校验100个 stable IDs。

### AgentLongBench 1M

- KVMem：
  `agentlongbench_1m_k100_g32_semantic_chunk2048_fullquery_recent0_adaptive_b128_fp8_mtp4_full50_20260810_174552/`，50/50 完成。
  K100/G32/B128、adaptive、FP8 KV、2048 semantic chunks、thinking8192、
  输出32768、temp .6/top-p .95/seed20260722、ctx1310720。
  不采用名称含 full50 但实际仅完成5条的20260814 B32性能运行；
  guided-query direct 的50条属于另一策略，不混入主参照。
- Compaction+RAG：
  `agentlongbench_1m_ds_summary_rag_notail_cap100k_vllm022_fp8_mtp4_seq2_20260811/`。
  配置指纹 `3719f68d2b034d2da4e832e4a57b19d3d65aef013f1efcffc5a653745bdc10c2`，
  与 `paper_utility_compact_rag_validated_v2_20260812.json` 一致。
  Jina32/8、3600 候选、动态100K输入，无tail，回答thinking8192、输出32768、
  temp .6/top-p .95/seed20260722。
- **第三方摘要例外**：原正式协议使用固定的 DeepSeek-v4-pro 摘要，
  `/home/chaidi/AgentLongBench-Long/summary_plus_tail.jsonl`，SHA256
  `13fe87e76137ff903e5322b0529fb98a80230321d9bb5cd08af9bb05df3e8dc3`。
  仅读取 `summary`，不用 tail/combined context；RAG仍覆盖完整原历史。
  仅替换回答模型时保留这个原固定输入，不改成9B摘要或重新调用DeepSeek。
- 数据：`/home/chaidi/AgentLongBench-Long/DeepseekMillion/{samples,manifest}.jsonl`，
  使用上述摘要与正式27B答案覆盖的50个ID。

## “只替换模型”的执行约束

每个 method × dataset 的配置独立冻结。生成参数、资源配置及默认值都需要
在启动命令与真实请求两层核对，不能只比较 Python 配置中的显式键。

允许差异：模型权重路径/标识及其模型自身的结构；经验证的 tokenizer 路径替换；
输出目录、日志路径、无语义影响的端口。其余差异必须先解决，不能带着未解释
差异生成一批答案后再称为严格模型替换。

必须锁定：

1. 输入版本、ID、顺序、system/user 提示、时间戳、分轮方式和 query span。
2. 权重精度、KV/index/query精度、active/gen/logical context、block/sub-block、
   sink/recent、scorer、adaptive阈值、索引位置、压力回收、query replay、MTP。
3. temperature/top-p/top-k/min-p/penalties/seed；thinking开关及预算；
   输出上限及“省略字段”对应的真实默认值；stop/EOS、超时与重试规则。
4. 摘要提示、触发边界、次数/循环终止、输出限制、尾部、模型来源；RAG embedding
   版本、编码截断长度、chunk边界、overlap、top-k、去重及最终排列。
5. 服务后端及版本、prefix cache/frozen branching、并行度、CPU/NVMe放置与缓存上限。
   历史KVMem ≤256K是FP16 KV，不得因为当前9B支持FP8就自动改成FP8。
   旧基线若实际使用vLLM/不同权重精度，应先复原对应配置，不能一律替换成qw3。
6. 官方判分代码及版本、每个子任务主指标、LLM judge的模型/提示/参数。
   MemoryAgentBench的特殊评分旧用DeepSeek-v4-pro；不能默认沿用本轮Flash。
   若原judge已不可调用，需要单独记录，并对两种模型已有答案统一补评分，
   原评分与统一补评分别展示，不重跑27B推理。

对于9B支持必须使用新引擎修复的情况，保存补丁与版本差异、验证旧策略路径没有
替换。若旧服务/默认值无法恢复，该cell标记“严格对齐未完成”，不静默降级为近似实验。

## 执行门槛与进度管理

0. 当前LongMemEval-S的102个Compaction+RAG样本全部达到终态并完成评分核算；
   终态包括有效答案或已验证的模型生成失败。失败保留在主指标分母，不要求靠
   重采样补成102个有效答案；网络中断不视为模型失败。审核空答案、截断、
   失败样本和实际参数，保留已有结果。GPU上的后续任务不与它并跑。
1. 为上述8个method×dataset单元建立 `reference_lock.json`：复制实际配置、原请求
   样例、源码版本、数据与摘要哈希；记录所有尚待核实字段。先做CPU准备和容量检查。
2. 配置比较只允许模型身份及必要路径差异；资源不足或9B暂不支持原精度时先处理
   支持问题，不擅自改budget/precision/replay以通过。
3. 每个单元跑协议验证：ALB用原队列最前2题；MAB选确定性固定的代表context，
   覆盖短输出、长输出和同context两个独立问题。验证GPU缓存真实命中、query replay
   与scorer无fallback、摘要/检索装配正确、输出限制生效、判分产物可读。
   参数不变时这些结果计入正式集；若修正协议则另存验证结果并重新生成。
4. 依表格阶段顺序执行；每阶段先KVMem再Compaction+RAG，各方法保持原单元并行度，
   不在同GPU上同时运行两个模型服务。只在完整样本/context边界切换。
5. 每个问题落盘答案、参数、耗时、终止原因及评分；校验ID/hash后断点续跑，
   重试保留原失败记录，不能按正确性挑选重试结果。
6. 上一阶段两种方法推理和评分完成后再启动下一阶段。后台定时器为
   `qw3-memory-benchmark-monitor.timer`，每30分钟通过受支持的 `codex queue`
   将检测排入当前任务。后续启动器、配置锁定和协议验证仍须由任务实际完成，
   不能跳过这些步骤直接执行历史27B启动脚本。

### 已安装的定时检测

- 脚本：`/home/chaidi/qw3/scripts/kvmem_eval/scheduled_memory_benchmark_check.py`。
- 状态与提示：`/home/chaidi/qw3/benchmark/memory_benchmark_monitor_20260915/`。
- systemd 用户单元：`/home/chaidi/.config/systemd/user/qw3-memory-benchmark-monitor.{timer,service}`。
- 2026-09-15 22:52 CST 手动触发验证成功，消息已进入当前任务队列；下一次23:22 CST。
- 这是本机定时器，不是 Codex App 的原生 Scheduled tasks 条目；需要本机开机，
  Codex 共享 app-server 可用才能实际执行任务。排队成功不等于新实验已启动。
- 无异常或阶段切换时保持静默。用户暂停优先；`paused` 文件阻止后续派发。
  全部阶段评分核验完成后写 `all_complete.json` 并禁用定时器。
- 查看：`systemctl --user status qw3-memory-benchmark-monitor.timer`；
  停止检测：`systemctl --user disable --now qw3-memory-benchmark-monitor.timer`。

## 输出

### 固定评分与汇报口径（2026-09-16 用户确认）

所有后续数据集必须复用对应27B参照的指标计算、答案后处理、判分规则、分母、
样本过滤、类别权重和汇总方式。每个阶段的 `reference_lock.json` 必须另附
`metric_contract`：原评分脚本及哈希、原judge提示/模型/参数、主指标字段、聚合层级、
失败处理与同ID检查。通过既有27B逐题分数复算原汇总的CPU核验后，才能将新结果
称为对齐结果；不必重新调用judge或重新生成27B答案。AgentLongBench三个长度组
分别追溯各自原评分入口，不用MemoryAgentBench的聚合方式替代。

耗时、TTFT、吞吐、token用量等效率指标也遵循同一要求：锁定原计时起止点、
首reasoning/首content定义、摘要/预热/重试是否计入、样本过滤、均值或中位数以及
加权方式。不能把纯decode吞吐替代端到端吞吐，或把新定义的统计量直接放进旧表。
原数据缺少必要字段时标记不可计算，不估算后混作实测值。

MemoryAgentBench主指标固定为原 `overall_macro_category_mean`：先在每个source
内按题平均，再在各category内按source等权平均，最后四个category等权平均。
分项继续沿用原口径：EventQA/Ruler/Fact用substring EM，Recsys用Recall@5，
LongMemEval用原LLM judge正确率，InfBench用原摘要rubric F1。保留原来的
context-macro、question-weighted指标时必须明确标为辅助指标，不能替代主指标。
不再另取1100道二值题的准确率作为整体结果或主对比；不排除空答案/已验证的模型
生成失败，不把基础设施故障计为模型错误，不改变原指标分母。

本轮9B K64/B32主分为26.37%，对齐27B K64/B32的40.47%；旧27B K200/B512的
40.99%单独注明配置，不能混作相同配置的模型对比。禁止把不同聚合或配置的数值
直接比较而不说明。后续Compaction+RAG也按同一主指标报告。
当前冻结与核验入口为实验目录的 `metric_contract.json`、`report_legacy_metrics.py`；
已用原27B逐题分数复现旧40.99%及其两项辅助汇总。Compaction+RAG评分完成后自动
调用该入口核验并生成 `legacy_metric_report.{json,md}`。

- 分数据集、长度档、子任务的9B方法对比，以及同ID的历史27B参照。
- MemoryAgentBench报告官方分项与macro aggregation，不把不同任务指标直接称作
  一个统一准确率；AgentLongBench同时保存官方得分和exact match。
- 时间分开记：history/archive构建、摘要、索引、query检索/selection/replay、
  TTFT、decode、完整请求。正式表格严格沿用相应27B单元的原公式与加权方式；
  新增诊断统计需另列定义，不能替代原指标。
- 单列失败、空输出、length终止、超时、fallback和覆盖率；不把不完整结果混作完整集。
- 时间估算在每阶段验证样本后，根据实测构建和每问成本给出范围；当前不沿用
  LongMemEval-S速度推算1316问的MAB，也不预先宣称全部能在一晚跑完。
